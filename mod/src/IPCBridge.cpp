#include "TrajectorySim.hpp"

#include <Geode/Geode.hpp>
#include <Geode/modify/CCDirector.hpp>
#include <Geode/modify/CCScheduler.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

using namespace geode::prelude;

// POSIX shared memory buffer layout for IPC between C++ mod and Python
struct SharedData {
  volatile int32_t frameIdx;          // 60Hz frame counter
  volatile int32_t frameReadyBin;     // 1 when C++ writes frame, 0 when Python consumed
  volatile float ttdRelease;          // Raw frames to death (Release)
  volatile float ttdHold;             // Raw frames to death (Hold)
  volatile float ttdImpulse;          // Raw frames to death (Impulse)
  volatile float maxHorizon;          // Dynamic screen horizon (60Hz frames)
  uint8_t frameBuffer[640 * 480 * 3]; // 921,600 bytes
};

static SharedData *data = nullptr;
static int lastFrameIdx = -1;
static int fileDescriptor = -1;
static bool s_ipcConnected = false;

/// Unmap the shared memory buffer.
static void closeShm() {
  if (!data)
    return;

  munmap(data, sizeof(SharedData));
  close(fileDescriptor);
  fileDescriptor = -1;
  data = nullptr;
  s_ipcConnected = false;
}

/// Retrieve the shared memory buffer.
static void initShm() {
  if (data)
    return;

  fileDescriptor = shm_open("/GDMem", O_RDWR, 0666);
  if (fileDescriptor != -1) {
    data = (SharedData *)mmap(NULL, sizeof(SharedData), PROT_READ | PROT_WRITE, MAP_SHARED, fileDescriptor, 0);
    if (data == MAP_FAILED) {
      data = nullptr;
      close(fileDescriptor);
      fileDescriptor = -1;
      s_ipcConnected = false;
    } else {
      s_ipcConnected = true;
    }
  } else {
    s_ipcConnected = false;
  }
}

class $modify(MyPlayLayer, PlayLayer) {
  bool init(GJGameLevel *level, bool useReplay, bool dontCreateObjects) {
    if (!PlayLayer::init(level, useReplay, dontCreateObjects)) {
      return false;
    }

    closeShm();
    initShm();
    TrajectorySim::init(this);
    lastFrameIdx = -1;
    return true;
  }

  void resetLevel() {
    PlayLayer::resetLevel();
    if (!TrajectorySim::isSimulating()) {
      closeShm();
      initShm();
      TrajectorySim::init(this);
      lastFrameIdx = -1;
    }
  }

  void onQuit() {
    TrajectorySim::quit();
    closeShm();
    PlayLayer::onQuit();
  }

  void processRecording() {
    if (!s_ipcConnected || !data) {
      return;
    }

    if (data->frameReadyBin == -1) {
      closeShm();
      return;
    }

    if (m_hasCompletedLevel) {
      data->frameReadyBin = -1;
      closeShm();
      return;
    }

    if (!m_started || m_playerDied || m_isPaused || !m_player1 || m_player1->m_isDead) {
      return;
    }

    // GD 2.208 advances m_currentProgress by 8 per 60Hz frame (2 per 240Hz tick)
    int frame60Idx = (m_gameState.m_currentProgress / 2) / 4;
    if (frame60Idx == lastFrameIdx)
      return;

    // Lockstep handshake: wait for Python to consume previous frame (or signal session end)
    while (data->frameReadyBin == 1) {
      if (data->frameReadyBin == -1) {
        closeShm();
        return;
      }
      std::this_thread::yield();
    }

    lastFrameIdx = frame60Idx;
    data->frameIdx = frame60Idx;

    // Run trajectory simulation to calculate frames to death and max horizon
    auto sim = TrajectorySim::simulate(this);
    data->ttdRelease = sim.ttdRelease;
    data->ttdHold = sim.ttdHold;
    data->ttdImpulse = sim.ttdImpulse;
    data->maxHorizon = sim.maxHorizon;

    // Capture 640x480 screen pixels from Cocos2d-x frame buffer at 60Hz
    glReadPixels(0, 0, 640, 480, GL_RGB, GL_UNSIGNED_BYTE, (void *)data->frameBuffer);

    std::atomic_thread_fence(std::memory_order_release);
    data->frameReadyBin = 1;
  }

  void postUpdate(float dt) {
    PlayLayer::postUpdate(dt);
    this->processRecording();
  }
};

static inline bool isLiveGameplayActive() {
  auto pl = PlayLayer::get();
  return s_ipcConnected && pl && !pl->m_isPaused && pl->m_started && !pl->m_playerDied && !pl->m_hasCompletedLevel;
}

class $modify(MyGJBaseGameLayer, GJBaseGameLayer) {
  void update(float dt) {
    if (isLiveGameplayActive()) {
      dt = 1.0f / 60.0f;
    }
    GJBaseGameLayer::update(dt);
  }

  void handleButton(bool down, int button, bool isPlayer1) {
    if (button == (int)PlayerButton::Jump || button == 1) {
      TrajectorySim::handleButtonPress(down, isPlayer1);
    }
    GJBaseGameLayer::handleButton(down, button, isPlayer1);
  }
};

class $modify(MyDirector, cocos2d::CCDirector) {
  void calculateDeltaTime() {
    CCDirector::calculateDeltaTime();
    if (isLiveGameplayActive()) {
      m_fDeltaTime = 1.0f / 60.0f;
      m_fActualDeltaTime = 1.0f / 60.0f;
    }
  }
};

class $modify(MyScheduler, cocos2d::CCScheduler) {
  void update(float dt) {
    if (isLiveGameplayActive()) {
      dt = 1.0f / 60.0f;
    }
    CCScheduler::update(dt);
  }
};
