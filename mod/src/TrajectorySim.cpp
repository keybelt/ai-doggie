#include "TrajectoryDrawer.hpp"

#include <Geode/Geode.hpp>
#include <Geode/modify/AchievementNotifier.hpp>
#include <Geode/modify/CCNode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>

using namespace geode::prelude;

// Simulation state
static bool s_simulating = false;
static bool s_simulationDead = false;
static bool s_player1Pressed = false;
static bool s_player2Pressed = false;
static size_t s_simFrameCount = 0;
static float s_rawDt = 1.0f / 240.0f;
static float s_frameDt = 1.0f / 240.0f;
static constexpr size_t SIM_ITERATIONS = 300;

static void initSimulation(PlayLayer *pl) {
  s_simFrameCount = 0;
  TrajectoryDrawer::get()->init(pl);
}

static void quitSimulation() {
  s_simulating = false;
  s_player1Pressed = false;
  s_player2Pressed = false;
  s_simFrameCount = 0;
  TrajectoryDrawer::get()->quit();
}

static void handleButtonPress(bool down, bool isPlayer1) {
  if (isPlayer1) {
    s_player1Pressed = down;
  } else {
    s_player2Pressed = down;
  }
}

static TrajectoryBranch simulateBranch(PlayLayer *pl, bool down) {
  s_simulationDead = false;
  PlayerObject *p1 = pl->m_player1;
  PlayerObject *p2 = (pl->m_gameState.m_isDualMode ? pl->m_player2 : nullptr);
  TrajectoryBranch branch;
  if (!p1)
    return branch;

  branch.p1.reserve(SIM_ITERATIONS + 1);
  branch.p1.push_back(p1->getPosition());
  if (p2) {
    branch.p2.reserve(SIM_ITERATIONS + 1);
    branch.p2.push_back(p2->getPosition());
  }

  for (size_t i = 0; i < SIM_ITERATIONS; ++i) {
    pl->checkCollisions(p1, s_frameDt, false);
    if (s_simulationDead || p1->m_isDead)
      break;

    if (p2) {
      pl->checkCollisions(p2, s_frameDt, false);
      if (s_simulationDead || p2->m_isDead)
        break;
    }

    if (down) {
      // Tap every frame: push then release so orbs see a fresh press each step
      p1->pushButton(PlayerButton::Jump);
      if (p2)
        p2->pushButton(PlayerButton::Jump);
    } else if (i == 0) {
      p1->releaseButton(PlayerButton::Jump);
      if (p2)
        p2->releaseButton(PlayerButton::Jump);
    }

    p1->update(s_frameDt);
    branch.p1.push_back(p1->getPosition());

    if (p2) {
      p2->update(s_frameDt);
      branch.p2.push_back(p2->getPosition());
    }
  }

  return branch;
}

static void simulate(PlayLayer *pl) {
  if (!pl || !pl->m_player1 || s_simulating)
    return;
  if (!pl->m_started || pl->m_isPaused || pl->m_playerDied || pl->m_hasCompletedLevel || pl->m_player1->m_isDead) {
    TrajectoryDrawer::get()->hide();
    return;
  }

  // Only simulate every 4 frames (60Hz on 240Hz physics)
  if (s_simFrameCount++ % 4 != 0) {
    return;
  }

  float warp = (pl->m_gameState.m_timeWarp > 0.f) ? pl->m_gameState.m_timeWarp : 1.f;
  s_frameDt = (s_rawDt > 0.f ? s_rawDt : (1.0f / 240.0f)) / warp;

  PlayerObject *p1 = pl->m_player1;
  PlayerObject *p2 = (pl->m_gameState.m_isDualMode ? pl->m_player2 : nullptr);

  bool wasPractice = pl->m_isPracticeMode;
  pl->m_isPracticeMode = true;

  // 1. Snapshot via practice checkpoint
  CheckpointObject *cp = pl->createCheckpoint();
  if (!cp) {
    pl->m_isPracticeMode = wasPractice;
    return;
  }
  cp->retain();
  if (cp->m_physicalCheckpointObject) {
    cp->m_physicalCheckpointObject->setVisible(false);
  }
  pl->storeCheckpoint(cp);

  // Snapshot visual states
  bool p1Effects = p1->m_playEffects;
  bool p2Effects = p2 ? p2->m_playEffects : false;

  // Suppress visual effects on player during simulation
  p1->m_playEffects = false;
  if (p2)
    p2->m_playEffects = false;

  s_simulating = true;

  // 2. Step forward headlessly & reset level to checkpoint
  TrajectoryData data;
  data.releaseBranch = simulateBranch(pl, false); // Release branch (Red)
  pl->resetLevel();
  pl->loadLastCheckpoint();

  data.holdBranch = simulateBranch(pl, true); // Hold branch (Green)
  pl->resetLevel();
  pl->loadLastCheckpoint();

  // 3. Clean up snapshot checkpoint
  pl->removeCheckpoint(false);
  cp->release();
  pl->m_isPracticeMode = wasPractice;

  // Restore live player state
  p1->m_playEffects = p1Effects;
  if (p2)
    p2->m_playEffects = p2Effects;

  s_simulating = false;

  TrajectoryDrawer::get()->render(pl, data);
}

// ================= Minimal Hooks & Suppression =================

class $modify(TrajectoryPLHook, PlayLayer) {
  bool init(GJGameLevel *level, bool useReplay, bool dontCreateObjects) {
    if (!PlayLayer::init(level, useReplay, dontCreateObjects))
      return false;
    initSimulation(this);
    if (m_attemptLabel)
      m_attemptLabel->setVisible(false);
    return true;
  }

  void resetLevel() {
    PlayLayer::resetLevel();
    if (m_attemptLabel)
      m_attemptLabel->setVisible(false);
    if (!s_simulating) {
      initSimulation(this);
    }
  }

  void destroyPlayer(PlayerObject *player, GameObject *object) {
    if (s_simulating) {
      s_simulationDead = true;
      if (player)
        player->m_isDead = true;
      return;
    }
    PlayLayer::destroyPlayer(player, object);
  }

  void onQuit() {
    quitSimulation();
    PlayLayer::onQuit();
  }
};

class $modify(TrajectoryBGLHook, GJBaseGameLayer) {
  void updateCamera(float dt) {
    s_rawDt = dt;
    simulate(PlayLayer::get());
    GJBaseGameLayer::updateCamera(dt);
  }

  void handleButton(bool down, int button, bool isPlayer1) {
    if (button == (int)PlayerButton::Jump || button == 1) {
      handleButtonPress(down, isPlayer1);
    }
    GJBaseGameLayer::handleButton(down, button, isPlayer1);
  }
};

class $modify(TrajectoryNodeHook, cocos2d::CCNode) {
  void addChild(cocos2d::CCNode *child, int zOrder, int tag) {
    if (s_simulating)
      return;
    CCNode::addChild(child, zOrder, tag);
  }
};

class $modify(TrajectoryAchievementHook, AchievementNotifier) {
  void notifyAchievement(char const *title, char const *desc, char const *icon, bool quest) {
    // Suppress all achievement popups
  }
};

class $modify(TrajectoryPOHook, PlayerObject) {
  void loadFromCheckpoint(PlayerCheckpoint *cp) {
    PlayerObject::loadFromCheckpoint(cp);
    m_isDead = false;

    bool isPressed = m_isSecondPlayer ? s_player2Pressed : s_player1Pressed;
    isPressed ? this->pushButton(PlayerButton::Jump) : this->releaseButton(PlayerButton::Jump);
  }
};
