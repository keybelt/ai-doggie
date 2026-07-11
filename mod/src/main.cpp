#include <Geode/Geode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

using namespace geode::prelude;

struct SharedData {
  volatile int32_t frameIdx;
  volatile int32_t currActionBin;
  volatile int32_t frameReadyBin;
  volatile int32_t actionReadyBin;
  uint8_t frameBuffer[640 * 480 * 3]; // 921,600 bytes
};

SharedData *data = nullptr;
bool isJumping = false;
int lastFrameIdx = -1;
std::string shmName = "GDMem";
int fileDescriptor = -1;

/// Retrieve the data from the shared memory.
void initShm() {
  if (data)
    return;

  // 0_RDWR is read/write, 0666 is read/write for owner, group, and others.
  fileDescriptor = shm_open(("/" + shmName).c_str(), O_RDWR, 0666);

  if (fileDescriptor != -1) {
    data = (SharedData *)mmap(NULL, sizeof(SharedData), PROT_READ | PROT_WRITE, MAP_SHARED, fileDescriptor, 0);
    if (data == MAP_FAILED) {
      data = nullptr;
      close(fileDescriptor);
      fileDescriptor = -1;
    }
  }
};

/// Unmap the shared memory and reset values.
void closeShm() {
  if (!data)
    return;

  munmap(data, sizeof(SharedData));
  close(fileDescriptor);
  fileDescriptor = -1;

  data = nullptr;
}

/// Injects shared memory logic into game loop.
class $modify(MyPlayLayer, PlayLayer) {
  bool init(GJGameLevel *level, bool useReplay, bool dontCreateObjects) {
    if (!PlayLayer::init(level, useReplay, dontCreateObjects)) {
      return false;
    }

    initShm();
    isJumping = false;
    lastFrameIdx = -1;
    return true;
  }

  void resetLevel() {
    PlayLayer::resetLevel();
    isJumping = false;
    lastFrameIdx = -1;
  }

  void onQuit() {
    closeShm();
    PlayLayer::onQuit();
  }
};

/// Override all the jumping logic.
class $modify(MyGJBaseGameLayer, GJBaseGameLayer) {
  void simulateClick(PlayerButton button, bool down, bool player2) {
    auto isClick = down ? &PlayerObject::pushButton : &PlayerObject::releaseButton;

    if (m_levelSettings->m_twoPlayerMode && m_gameState.m_isDualMode) {
      PlayerObject *plr = player2 ? m_player2 : m_player1;
      if (plr)
        (plr->*isClick)(button);
    } else {
      if (m_player1)
        (m_player1->*isClick)(button);

      if (m_gameState.m_isDualMode && m_player2) {
        (m_player2->*isClick)(button);
      }
    }

    m_effectManager->playerButton(down, !player2);

    if (down) {
      m_clicks++;
      if (button == PlayerButton::Jump)
        m_jumping = true;
    }
  }

  void processClick() {
    if (!m_player1)
      return;

    if (!data) {
      initShm();
      if (!data)
        return;
    }

    // 2.208 made m_currentProgress count twice as fast, for now we just divide it by 2
    int frameIdx = m_gameState.m_currentProgress / 2;
    if (frameIdx == lastFrameIdx)
      return;
    lastFrameIdx = frameIdx;
    data->frameIdx = frameIdx;

    if (frameIdx % 4 == 0) {
      // Capture 640x480 screen pixels from Cocos2d-x frame buffer
      glReadPixels(0, 0, 640, 480, GL_RGB, GL_UNSIGNED_BYTE, (void *)data->frameBuffer);

      data->actionReadyBin = 0;
      std::atomic_thread_fence(std::memory_order_release);
      data->frameReadyBin = 1;

      auto start = std::chrono::steady_clock::now();

      // TODO: make timeout indefinite for macro recording, only enforce 60hz for inference
      while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(8)) {
        if (data->actionReadyBin != 0) {
          break;
        }
        std::this_thread::yield();
      }
    }

    bool shouldJump = ((data->currActionBin >> (frameIdx % 4)) & 1);

    if (shouldJump && !isJumping) {
      simulateClick(PlayerButton::Jump, true, false);
      isJumping = true;
    } else if (!shouldJump && isJumping) {
      simulateClick(PlayerButton::Jump, false, false);
      isJumping = false;
    }
  }

  void processQueuedButtons(float dt, bool clearInputQueue) {
    GJBaseGameLayer::processQueuedButtons(dt, clearInputQueue);
    this->processClick();
  }
};
