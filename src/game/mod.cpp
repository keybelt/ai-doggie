#include <Geode/Geode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace geode::prelude;

struct SharedData {
  int32_t frameIdx;
  int32_t currActionBin;
  int32_t frameReadyBin;
  int32_t actionReadyBin;
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

    int frameIdx = m_gameState.m_currentProgress / 2;

    bool isRecordingMode = (data->actionReadyBin != -1);
    bool isValid = true;

    if (isRecordingMode) {
      data->currActionBin = 0;
      data->frameIdx = frameIdx;
      data->actionReadyBin = 0;
      data->frameReadyBin = 1;

      int timeout = 4000000;
      while (data->actionReadyBin == 0 && timeout > 0) {
        timeout--;
      }
      isValid = (timeout > 0);
    }

    if (isValid) {
      bool shouldJump = (data->currActionBin == 1);

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
