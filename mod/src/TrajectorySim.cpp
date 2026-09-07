#include "TrajectorySim.hpp"
#include "TrajectoryDrawer.hpp"

#include <Geode/Geode.hpp>
#include <Geode/modify/AchievementNotifier.hpp>
#include <Geode/modify/CCNode.hpp>
#include <Geode/modify/PlayerObject.hpp>

using namespace geode::prelude;

namespace TrajectorySim {
static bool s_simulating = false;
static bool s_simulationDead = false;
static bool s_player1Pressed = false;
static bool s_player2Pressed = false;
// Geometry Dash 2.2 runs physics at 240 TPS, where dt is measured in 60Hz frame units:
// 60.0f / 240.0f = 0.25f per tick
static constexpr float BASE_FRAME_DT = 0.25f;
static constexpr size_t SIM_ITERATIONS = 240;

bool isSimulating() { return s_simulating; }

void init(PlayLayer *pl) { TrajectoryDrawer::get()->init(pl); }

void quit() {
  s_simulating = false;
  s_player1Pressed = false;
  s_player2Pressed = false;
  TrajectoryDrawer::get()->quit();
}

void handleButtonPress(bool down, bool isPlayer1) {
  if (isPlayer1) {
    s_player1Pressed = down;
  } else {
    s_player2Pressed = down;
  }
}

void handleSimulationDeath(PlayerObject *player) {
  s_simulationDead = true;
  if (player) {
    player->m_isDead = true;
  }
}

enum class TrajectoryMode {
  Release,
  Hold,
  Impulse,
};

static TrajectoryBranch simulateBranch(PlayLayer *pl, TrajectoryMode mode, float dt) {
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
    if (mode == TrajectoryMode::Hold) {
      p1->pushButton(PlayerButton::Jump);
      if (p2)
        p2->pushButton(PlayerButton::Jump);
    } else if (mode == TrajectoryMode::Impulse) {
      if (i == 0) {
        p1->pushButton(PlayerButton::Jump);
        if (p2)
          p2->pushButton(PlayerButton::Jump);
      } else if (i == 1) {
        p1->releaseButton(PlayerButton::Jump);
        if (p2)
          p2->releaseButton(PlayerButton::Jump);
      }
    } else if (i == 0) { // TrajectoryMode::Release
      p1->releaseButton(PlayerButton::Jump);
      if (p2)
        p2->releaseButton(PlayerButton::Jump);
    }

    p1->update(dt);
    pl->checkCollisions(p1, dt, false);
    if (s_simulationDead || p1->m_isDead)
      break;

    if (p2) {
      p2->update(dt);
      pl->checkCollisions(p2, dt, false);
      if (s_simulationDead || p2->m_isDead)
        break;
    }

    branch.p1.push_back(p1->getPosition());

    if (p2) {
      branch.p2.push_back(p2->getPosition());
    }
  }

  return branch;
}

void simulate(PlayLayer *pl) {
  if (!pl || !pl->m_player1 || s_simulating)
    return;
  if (!pl->m_started || pl->m_isPaused || pl->m_playerDied || pl->m_hasCompletedLevel || pl->m_player1->m_isDead) {
    return;
  }

  float warp = (pl->m_gameState.m_timeWarp > 0.f) ? pl->m_gameState.m_timeWarp : 1.f;
  float dt = BASE_FRAME_DT * warp;

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
  data.releaseBranch = simulateBranch(pl, TrajectoryMode::Release, dt); // Release branch (Red)
  pl->resetLevel();
  pl->loadLastCheckpoint();

  data.holdBranch = simulateBranch(pl, TrajectoryMode::Hold, dt); // Hold branch (Green)
  pl->resetLevel();
  pl->loadLastCheckpoint();

  data.impulseBranch = simulateBranch(pl, TrajectoryMode::Impulse, dt); // Impulse branch (Yellow)
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
} // namespace TrajectorySim

// ================= Suppression & State Restoration Hooks =================

class $modify(TrajectoryNodeHook, cocos2d::CCNode) {
  void addChild(cocos2d::CCNode *child, int zOrder, int tag) {
    if (TrajectorySim::isSimulating())
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

    bool isPressed = m_isSecondPlayer ? TrajectorySim::s_player2Pressed : TrajectorySim::s_player1Pressed;
    isPressed ? this->pushButton(PlayerButton::Jump) : this->releaseButton(PlayerButton::Jump);
  }
};
