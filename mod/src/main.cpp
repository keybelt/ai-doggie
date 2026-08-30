#include <Geode/Geode.hpp>
#include <Geode/modify/CCNode.hpp>
#include <Geode/modify/EffectGameObject.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/GameObject.hpp>
#include <Geode/modify/HardStreak.hpp>
#include <Geode/modify/LevelEditorLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

using namespace geode::prelude;

struct MacroEvent {
  int32_t frame;
  int32_t down;
};

struct SharedData {
  volatile int32_t frameIdx;          // 60Hz frame counter
  volatile int32_t frameReadyBin;     // 1 when C++ writes frame, 0 when Python consumed
  volatile int32_t ttdRelease;        // Time to death for release
  volatile int32_t ttdHold;           // Time to death for hold
  volatile int32_t macroCount;        // Number of 240Hz macro events loaded
  uint8_t frameBuffer[640 * 480 * 3]; // 921,600 bytes
  MacroEvent macroBuffer[50000];      // 400,000 bytes
};

SharedData *data = nullptr;
int lastFrameIdx = -1;
std::string shmName = "GDMem";
int fileDescriptor = -1;
static size_t s_macroIndex = 0;

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
static bool s_isHold = false;
static float s_frameDt = 1.0f / 240.0f;
static std::vector<RingObject *> s_activatedRings;
static std::vector<EffectGameObject *> s_activatedEffects;

inline bool isSimulating() { return s_simulating; }
inline bool isHold() { return s_isHold; }
inline PlayerObject *getClonePlayer() { return s_clonePlayer; }

inline void setFrameDelta(float dt) {
  PlayLayer *pl = PlayLayer::get();
  if (pl && pl->m_gameState.m_timeWarp > 0.0f) {
    s_frameDt = dt / pl->m_gameState.m_timeWarp;
  } else {
    s_frameDt = dt;
  }
}

inline bool handleSimulationDeath(PlayerObject *player) {
  if (s_simulating && player == s_clonePlayer) {
    s_simulationDead = true;
    return true;
  }
  return false;
}

inline void trackActivatedRing(RingObject *ring) {
  if (s_simulating && ring) {
    s_activatedRings.push_back(ring);
  }
}

inline void restoreActivatedRings() {
  for (auto *ring : s_activatedRings) {
    ring->m_activated = false;
    ring->m_activatedByPlayer1 = false;
    ring->m_activatedByPlayer2 = false;
  }
  s_activatedRings.clear();
}

inline void trackActivatedEffect(EffectGameObject *effect) {
  if (s_simulating && effect) {
    s_activatedEffects.push_back(effect);
  }
}

inline void restoreActivatedEffects() {
  for (auto *effect : s_activatedEffects) {
    effect->m_activated = false;
    effect->m_activatedByPlayer1 = false;
    effect->m_activatedByPlayer2 = false;
  }
  s_activatedEffects.clear();
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
  s_clonePlayer->m_isDead = false;
  s_isHold = isHold;

  s_clonePlayer->copyAttributes(realPlayer);
  s_clonePlayer->m_gravityMod = realPlayer->m_gravityMod;
  s_clonePlayer->m_isOnGround = realPlayer->m_isOnGround;
  s_clonePlayer->setVisible(false);

  if (isHold) {
    s_clonePlayer->pushButton(PlayerButton::Jump);
  } else {
    s_clonePlayer->releaseButton(PlayerButton::Jump);
  }

  int result = horizon;
  for (int step = 0; step < horizon; ++step) {
    if (s_clonePlayer->m_collisionLogTop)
      s_clonePlayer->m_collisionLogTop->removeAllObjects();
    if (s_clonePlayer->m_collisionLogBottom)
      s_clonePlayer->m_collisionLogBottom->removeAllObjects();
    if (s_clonePlayer->m_collisionLogLeft)
      s_clonePlayer->m_collisionLogLeft->removeAllObjects();
    if (s_clonePlayer->m_collisionLogRight)
      s_clonePlayer->m_collisionLogRight->removeAllObjects();
    s_clonePlayer->m_touchedRings.clear();

    pl->checkCollisions(s_clonePlayer, dt, false);
    if (s_simulationDead || s_clonePlayer->m_isDead) {
      result = step;
      break;
    }

    s_clonePlayer->update(dt);
  }

  restoreActivatedRings();
  restoreActivatedEffects();
  return result;
}

std::pair<int32_t, int32_t> computeTTD(PlayLayer *pl, int horizon) {
  if (!pl || !pl->m_player1 || !s_clonePlayer) {
    return {horizon, horizon};
  }

  s_simulating = true;
  int32_t ttdRelease = simulateBranch(pl, pl->m_player1, false, horizon, s_frameDt);
  int32_t ttdHold = simulateBranch(pl, pl->m_player1, true, horizon, s_frameDt);
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
    lastFrameIdx = -1;
    s_macroIndex = 0;
    return true;
  }

  void resetLevel() {
    PlayLayer::resetLevel();
    lastFrameIdx = -1;
    s_macroIndex = 0;
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
        player->flipGravity(p1, true);
      }
      return;
    }
    PlayLayer::flipGravity(player, p1, p2);
  }

  void playEndAnimationToPos(cocos2d::CCPoint p0) {
    if (TrajectorySim::isSimulating())
      return;
    PlayLayer::playEndAnimationToPos(p0);
  }

  void onQuit() {
    TrajectorySim::cleanup();
    closeShm();
    PlayLayer::onQuit();
  }
};

class $modify(MyLevelEditorLayer, LevelEditorLayer) {
  bool init(GJGameLevel *level, bool unk) {
    bool result = LevelEditorLayer::init(level, unk);
    TrajectorySim::cleanup();
    return result;
  }
};

class $modify(MyHardStreak, HardStreak) {
  void addPoint(cocos2d::CCPoint p0) {
    if (TrajectorySim::isSimulating())
      return;
    HardStreak::addPoint(p0);
  }
};

class $modify(MyGameObject, GameObject) {
  void playShineEffect() {
    if (TrajectorySim::isSimulating())
      return;
    GameObject::playShineEffect();
  }
};

class $modify(MyEffectGameObject, EffectGameObject) {
  void triggerObject(GJBaseGameLayer *layer, int p1, const gd::vector<int> *p2) {
    if (TrajectorySim::isSimulating())
      return;
    EffectGameObject::triggerObject(layer, p1, p2);
  }
};

class $modify(MyCCNode, cocos2d::CCNode) {
  cocos2d::CCAction *runAction(cocos2d::CCAction *action) {
    if (TrajectorySim::isSimulating())
      return nullptr;
    return cocos2d::CCNode::runAction(action);
  }
};

class $modify(MyPlayerObject, PlayerObject) {
  void playSpiderDashEffect(cocos2d::CCPoint from, cocos2d::CCPoint to) {
    if (TrajectorySim::isSimulating())
      return;
    PlayerObject::playSpiderDashEffect(from, to);
  }

  void incrementJumps() {
    if (TrajectorySim::isSimulating())
      return;
    PlayerObject::incrementJumps();
  }

  void update(float dt) {
    PlayerObject::update(dt);
    if (PlayLayer::get() && !TrajectorySim::isSimulating()) {
      TrajectorySim::setFrameDelta(dt);
    }
  }

  void ringJump(RingObject *ring, bool p1) {
    if (TrajectorySim::isSimulating()) {
      TrajectorySim::trackActivatedRing(ring);
      PlayerObject::ringJump(ring, p1);
      return;
    }
    PlayerObject::ringJump(ring, p1);
  }
};

/// Override jumping and input processing.
class $modify(MyGJBaseGameLayer, GJBaseGameLayer) {
  void toggleDualMode(GameObject *object, bool dual, PlayerObject *player, bool noEffects) {
    if (TrajectorySim::isSimulating())
      return;
    GJBaseGameLayer::toggleDualMode(object, dual, player, noEffects);
  }

  void playerTouchedRing(PlayerObject *player, RingObject *ring) {
    if (TrajectorySim::isSimulating()) {
      player->m_touchedRings.insert(ring->m_uniqueID);
      if (TrajectorySim::isHold() && !ring->m_activated) {
        player->ringJump(ring, true);
      }
      return;
    }
    GJBaseGameLayer::playerTouchedRing(player, ring);
  }

  void playerTouchedTrigger(PlayerObject *player, EffectGameObject *trigger) {
    if (TrajectorySim::isSimulating()) {
      TrajectorySim::trackActivatedEffect(trigger);
      if (trigger && !trigger->m_activatedByPlayer1 && trigger->m_speedModType > 0) {
        // Native GD speed portal constants from EffectGameObject::updateSpeedModType:
        // 1 = 0.5x (Slow), 2 = 1.0x (Normal), 3 = 2.0x (Fast), 4 = 3.0x (Very Fast), 5 = 4.0x (Fastest)
        float speed = 0.9f;
        switch (trigger->m_speedModType) {
        case 1:
          speed = 0.7f; // 0.5x speed
          break;
        case 2:
          speed = 0.9f; // 1.0x speed
          break;
        case 3:
          speed = 1.1f; // 2.0x speed
          break;
        case 4:
          speed = 1.3f; // 3.0x speed
          break;
        case 5:
          speed = 1.6f; // 4.0x speed
          break;
        default:
          speed = 0.9f;
          break;
        }
        player->updateTimeMod(speed, true);
      }
      GJBaseGameLayer::playerTouchedTrigger(player, trigger);
      return;
    }
    GJBaseGameLayer::playerTouchedTrigger(player, trigger);
  }

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
    }
  }

  void processBot() {
    if (!data || data->macroCount <= 0)
      return;

    // gd 2.208 quirk makes currentProgress count twice as fast
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

  void processRecording() {
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
    auto [ttdRelease, ttdHold] = TrajectorySim::computeTTD(PlayLayer::get(), 240);
    data->ttdRelease = ttdRelease;
    data->ttdHold = ttdHold;

    // Capture 640x480 screen pixels from Cocos2d-x frame buffer at 60Hz
    glReadPixels(0, 0, 640, 480, GL_RGB, GL_UNSIGNED_BYTE, (void *)data->frameBuffer);

    std::atomic_thread_fence(std::memory_order_release);
    data->frameReadyBin = 1;
  }

  void updateCamera(float dt) {
    this->processRecording();
    GJBaseGameLayer::updateCamera(dt);
  }
};
