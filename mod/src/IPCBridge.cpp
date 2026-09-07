#include <Geode/Geode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace geode::prelude;

struct MacroEvent {
  int32_t frame;
  int32_t down;
};

// POSIX shared memory buffer layout for IPC between C++ mod and Python
struct SharedData {
  volatile int32_t frameIdx;          // 60Hz frame counter
  volatile int32_t frameReadyBin;     // 1 when C++ writes frame, 0 when Python consumed
  volatile int32_t macroCount;        // Number of 240Hz macro events loaded
  uint8_t frameBuffer[640 * 480 * 3]; // 921,600 bytes
  MacroEvent macroBuffer[50000];      // 400,000 bytes
};

static SharedData *data = nullptr;
static int lastFrameIdx = -1;
static int fileDescriptor = -1;
static size_t s_macroIndex = 0;

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
    }
  }
}

/// Unmap the shared memory buffer on level exit.
static void closeShm() {
  if (!data)
    return;

  munmap(data, sizeof(SharedData));
  close(fileDescriptor);
  fileDescriptor = -1;
  data = nullptr;
}

class $modify(MyPlayLayer, PlayLayer) {
  bool init(GJGameLevel *level, bool useReplay, bool dontCreateObjects) {
    if (!PlayLayer::init(level, useReplay, dontCreateObjects)) {
      return false;
    }

    initShm();
    lastFrameIdx = -1;
    s_macroIndex = 0;
    return true;
  }

  void resetLevel() {
    PlayLayer::resetLevel();
    lastFrameIdx = -1;
    s_macroIndex = 0;
  }

  void onQuit() {
    closeShm();
    PlayLayer::onQuit();
  }

  void processRecording() {
    if (!m_started || m_playerDied || m_isPaused || m_hasCompletedLevel || !m_player1 || m_player1->m_isDead) {
      return;
    }

    if (!data) {
      initShm();
      if (!data)
        return;
    }

    // GD 2.208 advances m_currentProgress by 8 per 60Hz frame (2 per 240Hz tick)
    int frame60Idx = (m_gameState.m_currentProgress / 2) / 4;
    if (frame60Idx == lastFrameIdx)
      return;
    lastFrameIdx = frame60Idx;
    data->frameIdx = frame60Idx;

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

class $modify(MyGJBaseGameLayer, GJBaseGameLayer) {
  void simulateClick(PlayerButton button, bool down, bool player2) {
    auto performButton = down ? &PlayerObject::pushButton : &PlayerObject::releaseButton;
    bool swapControls = GameManager::get()->getGameVariable(GameVar::Flip2PlayerControls);
    player2 = swapControls ? !player2 : player2;

    if (m_levelSettings->m_twoPlayerMode) {
      PlayerObject *plr = player2 ? m_player2 : m_player1;
      if (plr)
        (plr->*performButton)(button);
    } else {
      if (m_player1)
        (m_player1->*performButton)(button);

      if (m_gameState.m_isDualMode && m_player2) {
        (m_player2->*performButton)(button);
      }
    }

    m_effectManager->playerButton(down, !player2);

    if (down) {
      m_clicks++;
      if (button == PlayerButton::Jump)
        m_jumping = true;
    } else {
      if (button == PlayerButton::Jump)
        m_jumping = false;
    }
  }

  void processBot() {
    if (!data || data->macroCount <= 0)
      return;

    int32_t progress = m_gameState.m_currentProgress / 2;
    while (s_macroIndex < (size_t)data->macroCount && data->macroBuffer[s_macroIndex].frame <= progress) {
      const auto &ev = data->macroBuffer[s_macroIndex++];
      this->simulateClick(PlayerButton::Jump, ev.down != 0, false);
    }
  }

  void processQueuedButtons(float dt, bool clearInputQueue) {
    GJBaseGameLayer::processQueuedButtons(dt, clearInputQueue);
    this->processBot();
  }
};
