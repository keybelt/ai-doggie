#include <Geode/Geode.hpp>
#include <Geode/modify/EffectGameObject.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>

#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd>
#include <utility>

using namespace geode::prelude;

struct SharedData {
  volatile int32_t frameIdx;   // 60Hz frame counter
  volatile int32_t currAction; // 0 = Release, 1 = Jump
  volatile int32_t frameReadyBin;
  volatile int32_t actionReadyBin;
  volatile int32_t ttdRelease;
  volatile int32_t ttdHold;
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

  // O_RDWR is read/write, 0666 is read/write for owner, group, and others.
  fileDescriptor = shm_open(("/" + shmName).c_str(), O_RDWR, 0666);

  if (fileDescriptor != -1) {
    data = (SharedData *)mmap(NULL, sizeof(SharedData), PROT_READ | PROT_WRITE, MAP_SHARED, fileDescriptor, 0);
    if (data == MAP_FAILED) {
      data = nullptr;
      close(fileDescriptor);
      fileDescriptor = -1;
    }
  }
}

/// Unmap the shared memory and reset values.
void closeShm() {
  if (!data)
    return;

  munmap(data, sizeof(SharedData));
  close(fileDescriptor);
  fileDescriptor = -1;

  data = nullptr;
}

namespace TrajectorySim {
static PlayerObject *s_clonePlayer = nullptr;
static bool s_simulating = false;
static bool s_simulationDead = false;

inline bool isSimulating() { return s_simulating; }

inline bool handleSimulationDeath(PlayerObject *player) {
  if (s_simulating && player == s_clonePlayer) {
    s_simulationDead = true;
    return true;
  }
  return false;
}

void init(PlayLayer *pl) {
  if (!pl || s_clonePlayer)
    return;

  s_clonePlayer = PlayerObject::create(1, 1, pl, pl, true);
  if (!s_clonePlayer)
    return;

  s_clonePlayer->retain();
  s_clonePlayer->setPosition({0, 105});
  s_clonePlayer->setVisible(false);
  if (pl->m_objectLayer) {
    pl->m_objectLayer->addChild(s_clonePlayer);
  }
}

void cleanup() {
  if (s_clonePlayer) {
    s_clonePlayer->removeFromParent();
    s_clonePlayer->release();
    s_clonePlayer = nullptr;
  }
}

int simulateBranch(PlayLayer *pl, PlayerObject *realPlayer, bool isHold, int horizon, float dt) {
  if (!s_clonePlayer || !realPlayer || !pl)
    return horizon;

  s_simulationDead = false;

  s_clonePlayer->copyAttributes(realPlayer);
  s_clonePlayer->setPosition(realPlayer->getPosition());
  s_clonePlayer->m_yAccel = realPlayer->m_yAccel;
  s_clonePlayer->m_xAccel = realPlayer->m_xAccel;
  s_clonePlayer->m_isUpsideDown = realPlayer->m_isUpsideDown;
  s_clonePlayer->m_gravityMod = realPlayer->m_gravityMod;
  s_clonePlayer->m_isOnGround = realPlayer->m_isOnGround;
  s_clonePlayer->m_playerSpeed = realPlayer->m_playerSpeed;
  s_clonePlayer->m_vehicleSize = realPlayer->m_vehicleSize;

  if (isHold) {
    s_clonePlayer->pushButton(PlayerButton::Jump);
  } else {
    s_clonePlayer->releaseButton(PlayerButton::Jump);
  }

  for (int step = 0; step < horizon; ++step) {
    if (s_clonePlayer->m_collisionLogTop)
      s_clonePlayer->m_collisionLogTop->removeAllObjects();
    if (s_clonePlayer->m_collisionLogBottom)
      s_clonePlayer->m_collisionLogBottom->removeAllObjects();
    if (s_clonePlayer->m_collisionLogLeft)
      s_clonePlayer->m_collisionLogLeft->removeAllObjects();
    if (s_clonePlayer->m_collisionLogRight)
      s_clonePlayer->m_collisionLogRight->removeAllObjects();

    pl->checkCollisions(s_clonePlayer, dt, false);
    if (s_simulationDead) {
      return step;
    }

    s_clonePlayer->update(dt);
  }

  return horizon;
}

std::pair<int32_t, int32_t> computeTTD(PlayLayer *pl, int horizon = 120) {
  if (!pl || !pl->m_player1 || !s_clonePlayer) {
    return {horizon, horizon};
  }

  s_simulating = true;
  float dt = 1.0f / 240.0f;
  if (pl->m_gameState.m_timeWarp > 0.0f) {
    dt = (1.0f / 240.0f) / pl->m_gameState.m_timeWarp;
  }

  int32_t ttdRelease = simulateBranch(pl, pl->m_player1, false, horizon, dt);
  int32_t ttdHold = simulateBranch(pl, pl->m_player1, true, horizon, dt);

  s_simulating = false;
  return {ttdRelease, ttdHold};
}
} // namespace TrajectorySim

/// Injects shared memory logic and simulator lifecycle into game loop.
class $modify(MyPlayLayer, PlayLayer) {
  bool init(GJGameLevel *level, bool useReplay, bool dontCreateObjects) {
    if (!PlayLayer::init(level, useReplay, dontCreateObjects)) {
      return false;
    }

    initShm();
    TrajectorySim::init(this);
    isJumping = false;
    lastFrameIdx = -1;
    return true;
  }

  void resetLevel() {
    PlayLayer::resetLevel();
    isJumping = false;
    lastFrameIdx = -1;
  }

  void destroyPlayer(PlayerObject *player, GameObject *gameObject) {
    if (TrajectorySim::handleSimulationDeath(player)) {
      return;
    }
    PlayLayer::destroyPlayer(player, gameObject);
  }

  void flipGravity(PlayerObject *player, bool p1, bool p2) {
    if (TrajectorySim::isSimulating()) {
      if (player) {
        player->m_isUpsideDown = !player->m_isUpsideDown;
        player->m_gravityMod = -player->m_gravityMod;
      }
      return;
    }
    PlayLayer::flipGravity(player, p1, p2);
  }

  void onQuit() {
    TrajectorySim::cleanup();
    closeShm();
    PlayLayer::onQuit();
  }
};

class $modify(MyPlayerObject, PlayerObject) {
  void ringJump(RingObject *ring, bool p1) {
    if (TrajectorySim::isSimulating()) {
      if (ring) {
        this->m_yAccel = ring->m_jumpBoost;
      }
      return;
    }
    PlayerObject::ringJump(ring, p1);
  }
};

class $modify(MyEffectGameObject, EffectGameObject) {
  void triggerObject(GJBaseGameLayer *layer, int p1, const gd::vector<int> *p2) {
    if (TrajectorySim::isSimulating())
      return;
    EffectGameObject::triggerObject(layer, p1, p2);
  }
};

/// Override jumping and collision logic.
class $modify(MyGJBaseGameLayer, GJBaseGameLayer) {
  bool canBeActivatedByPlayer(PlayerObject *p0, EffectGameObject *p1) {
    if (TrajectorySim::isSimulating())
      return false;
    return GJBaseGameLayer::canBeActivatedByPlayer(p0, p1);
  }

  void collisionCheckObjects(PlayerObject *player, gd::vector<GameObject *> *vec, int objectsCount, float dt) {
    if (TrajectorySim::isSimulating()) {
      gd::vector<GameObject *> extra;
      extra.reserve(objectsCount);
      for (int i = 0; i < objectsCount; i++) {
        GameObject *obj = vec->at(i);
        if (obj->m_objectType == GameObjectType::Solid || 
            obj->m_objectType == GameObjectType::Hazard ||
            obj->m_objectType == GameObjectType::AnimatedHazard || 
            obj->m_objectType == GameObjectType::Slope ||
            obj->m_objectType == GameObjectType::JumpPad ||
            obj->m_objectType == GameObjectType::Modifier) {
          extra.push_back(obj);
        }
      }
      GJBaseGameLayer::collisionCheckObjects(player, &extra, extra.size(), dt);
      return;
    }
    GJBaseGameLayer::collisionCheckObjects(player, vec, objectsCount, dt);
  }

  void playerTouchedRing(PlayerObject *player, RingObject *ring) {
    if (TrajectorySim::isSimulating())
      return;
    GJBaseGameLayer::playerTouchedRing(player, ring);
  }

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

    // GD 2.208 advances m_currentProgress by 8 per 60Hz frame (2 per 240Hz tick)
    int frame60Idx = (m_gameState.m_currentProgress / 2) / 4;
    if (frame60Idx == lastFrameIdx)
      return;
    lastFrameIdx = frame60Idx;
    data->frameIdx = frame60Idx;

    // Compute Time-To-Death (TTD) for Release and Hold
    auto [ttdRelease, ttdHold] = TrajectorySim::computeTTD(PlayLayer::get(), 120);
    data->ttdRelease = ttdRelease;
    data->ttdHold = ttdHold;

    // Capture 640x480 screen pixels from Cocos2d-x frame buffer at 60Hz
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

    bool shouldJump = (data->currAction == 1);
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
