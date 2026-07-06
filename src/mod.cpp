#include <Geode/Geode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include <chrono>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace geode::prelude;

struct SharedData {
  volatile int32_t frameIdx;
  volatile int32_t currActionBin;
  volatile int32_t frameReadyBin;
  volatile int32_t actionReadyBin;
  volatile int32_t width;
  volatile int32_t height;
  uint8_t frameBuffer[1920 * 1200 * 3]; // 6,912,000 bytes
};

SharedData *data = nullptr;
bool isJumping = false;
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
    return true;
  }

  void resetLevel() {
    PlayLayer::resetLevel();
    isJumping = false;
  }

  void onQuit() {
    closeShm();
    PlayLayer::onQuit();
  }
};

/// Override all the jumping logic.
class $modify(MyGJBaseGameLayer, GJBaseGameLayer) {
  void sendClick(PlayerButton button, bool down, bool player2) {
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

    int frameIdx = m_gameState.m_currentProgress / 2; // i got this from eclipse or smth idk why /2 but oh well
    data->frameIdx = frameIdx;

    bool isValid = true;

    if (frameIdx % 4 == 0) {
      GLint viewport[4];
      glGetIntegerv(GL_VIEWPORT, viewport);
      int width = viewport[2];
      int height = viewport[3];

      if (width > 1920)
        width = 1920;
      if (height > 1200)
        height = 1200;

      data->width = width;
      data->height = height;

      // Capture screen pixels from Cocos2d-x frame buffer
      glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, (void *)data->frameBuffer);

      data->actionReadyBin = 0;
      data->frameReadyBin = 1;

      auto start = std::chrono::steady_clock::now();
      bool timedOut = true;
      // 16ms timeout to prevent game freeze if Python crashes/stops
      while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(16)) {
        if (data->actionReadyBin != 0) {
          timedOut = false;
          break;
        }
      }
      isValid = !timedOut;
    }

    if (isValid) {
      bool shouldJump = ((data->currActionBin >> (frameIdx % 4)) & 1);

      if (shouldJump && !isJumping) {
        sendClick(PlayerButton::Jump, true, false);
        isJumping = true;
      } else if (!shouldJump && isJumping) {
        sendClick(PlayerButton::Jump, false, false);
        isJumping = false;
      }
    }
  }

  void processQueuedButtons(float dt, bool clearInputQueue) {
    GJBaseGameLayer::processQueuedButtons(dt, clearInputQueue);
    this->processClick();
  }
};
