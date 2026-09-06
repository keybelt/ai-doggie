#include "TrajectorySimulator.hpp"

#include <Geode/modify/AchievementNotifier.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/GameObject.hpp>
#include <Geode/modify/HardStreak.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>
#include <Geode/modify/RingObject.hpp>

TrajectorySimulator *TrajectorySimulator::get() {
  static TrajectorySimulator instance;
  return &instance;
}

void TrajectorySimulator::init(PlayLayer *pl) {
  if (!pl)
    return;

  if (!m_drawNode) {
    m_drawNode = cocos2d::CCDrawNode::create();
    m_drawNode->retain();
  }

  if (m_drawNode->getParent()) {
    m_drawNode->removeFromParent();
  }

  if (pl->m_debugDrawNode && pl->m_debugDrawNode->getParent()) {
    pl->m_debugDrawNode->getParent()->addChild(m_drawNode);
    m_drawNode->setZOrder(pl->m_debugDrawNode->getZOrder());
  } else {
    pl->addChild(m_drawNode, 100);
  }

  m_simFrameCount = 0;
  m_drawNode->setVisible(false);
}

void TrajectorySimulator::quit() {
  m_simulating = false;
  m_player1Pressed = false;
  m_player2Pressed = false;
  m_simFrameCount = 0;
  if (m_drawNode) {
    m_drawNode->removeFromParent();
    m_drawNode->release();
    m_drawNode = nullptr;
  }
}

void TrajectorySimulator::hide() {
  if (m_drawNode) {
    m_drawNode->setVisible(false);
    m_drawNode->clear();
  }
}

void TrajectorySimulator::handleButton(bool down, bool isPlayer1) {
  if (isPlayer1) {
    m_player1Pressed = down;
  } else {
    m_player2Pressed = down;
  }
}



void TrajectorySimulator::simulateBranch(PlayLayer *pl, bool down) {
  m_simulationDead = false;
  PlayerObject *p1 = pl->m_player1;
  PlayerObject *p2 = (pl->m_gameState.m_isDualMode ? pl->m_player2 : nullptr);
  if (!p1)
    return;

  cocos2d::ccColor4F col = down ? cocos2d::ccColor4F{0.f, 1.f, 0.1f, 1.f} : cocos2d::ccColor4F{1.f, 0.f, 0.1f, 1.f};

  for (size_t i = 0; i < m_iterations; ++i) {
    cocos2d::CCPoint prev1 = p1->getPosition();
    cocos2d::CCPoint prev2 = p2 ? p2->getPosition() : cocos2d::CCPointZero;

    pl->checkCollisions(p1, m_frameDt, false);
    if (m_simulationDead || p1->m_isDead)
      break;

    if (p2) {
      pl->checkCollisions(p2, m_frameDt, false);
      if (m_simulationDead || p2->m_isDead)
        break;
    }

    if (down) {
      // Tap every frame: push then release so orbs see a fresh press each step
      p1->pushButton(PlayerButton::Jump);
      if (p2) p2->pushButton(PlayerButton::Jump);
    } else if (i == 0) {
      p1->releaseButton(PlayerButton::Jump);
      if (p2) p2->releaseButton(PlayerButton::Jump);
    }

    p1->update(m_frameDt);
    if (p2)
      p2->update(m_frameDt);

    m_drawNode->drawSegment(prev1, p1->getPosition(), 0.65f, col);
    if (p2)
      m_drawNode->drawSegment(prev2, p2->getPosition(), 0.65f, col);
  }
}

void TrajectorySimulator::simulate(PlayLayer *pl) {
  if (!pl || !pl->m_player1 || m_simulating)
    return;
  if (!pl->m_started || pl->m_isPaused || pl->m_playerDied || pl->m_hasCompletedLevel || pl->m_player1->m_isDead) {
    hide();
    return;
  }

  // Only simulate every 4 frames (60Hz on 240Hz physics)
  if (m_simFrameCount++ % 4 != 0) {
    return;
  }

  if (!m_drawNode || !m_drawNode->getParent()) {
    init(pl);
  }

  m_simulating = true;
  m_drawNode->setVisible(true);
  m_drawNode->clear();

  float warp = (pl->m_gameState.m_timeWarp > 0.f) ? pl->m_gameState.m_timeWarp : 1.f;
  m_frameDt = (m_rawDt > 0.f ? m_rawDt : (1.0f / 240.0f)) / warp;

  PlayerObject *p1 = pl->m_player1;
  PlayerObject *p2 = (pl->m_gameState.m_isDualMode ? pl->m_player2 : nullptr);

  bool wasPractice = pl->m_isPracticeMode;
  pl->m_isPracticeMode = true;

  // 1. Snapshot via practice checkpoint
  CheckpointObject *cp = pl->createCheckpoint();
  if (!cp) {
    pl->m_isPracticeMode = wasPractice;
    m_simulating = false;
    return;
  }
  cp->retain();
  if (cp->m_physicalCheckpointObject) {
    cp->m_physicalCheckpointObject->setVisible(false);
  }
  pl->storeCheckpoint(cp);

  // 2. Step forward headlessly & reset level to checkpoint
  simulateBranch(pl, false); // Release branch (Red)
  pl->resetLevel();
  pl->loadLastCheckpoint();

  simulateBranch(pl, true); // Hold branch (Green)
  pl->resetLevel();
  pl->loadLastCheckpoint();

  // 3. Clean up snapshot checkpoint
  pl->removeCheckpoint(false);
  cp->release();
  pl->m_isPracticeMode = wasPractice;

  // Revive player from simulation death
  p1->m_isDead = false;
  if (p2)
    p2->m_isDead = false;
  pl->m_playerDied = false;

  // Restore live player button state
  m_player1Pressed ? p1->pushButton(PlayerButton::Jump) : p1->releaseButton(PlayerButton::Jump);
  if (p2) {
    m_player2Pressed ? p2->pushButton(PlayerButton::Jump) : p2->releaseButton(PlayerButton::Jump);
  }

  m_simulating = false;
}

// ---------------- Minimal Hooks ----------------

class $modify(TrajectoryPLHook, PlayLayer) {
  bool init(GJGameLevel *level, bool useReplay, bool dontCreateObjects) {
    if (!PlayLayer::init(level, useReplay, dontCreateObjects))
      return false;
    TrajectorySimulator::get()->init(this);
    if (m_attemptLabel)
      m_attemptLabel->setVisible(false);
    return true;
  }

  void resetLevel() {
    PlayLayer::resetLevel();
    if (m_attemptLabel)
      m_attemptLabel->setVisible(false);
    if (!TrajectorySimulator::get()->isSimulating()) {
      TrajectorySimulator::get()->init(this);
    }
  }

  void destroyPlayer(PlayerObject *player, GameObject *object) {
    if (TrajectorySimulator::get()->isSimulating()) {
      TrajectorySimulator::get()->handleDeath();
      if (player)
        player->m_isDead = true;
      return;
    }
    PlayLayer::destroyPlayer(player, object);
  }

  void onQuit() {
    TrajectorySimulator::get()->quit();
    PlayLayer::onQuit();
  }
};

class $modify(TrajectoryBGLHook, GJBaseGameLayer) {
  void updateCamera(float dt) {
    TrajectorySimulator::get()->setFrameDt(dt);
    TrajectorySimulator::get()->simulate(PlayLayer::get());
    GJBaseGameLayer::updateCamera(dt);
  }

  void handleButton(bool down, int button, bool isPlayer1) {
    if (button == (int)PlayerButton::Jump || button == 1) {
      TrajectorySimulator::get()->handleButton(down, isPlayer1);
    }
    GJBaseGameLayer::handleButton(down, button, isPlayer1);
  }

  cocos2d::CCParticleSystemQuad* spawnParticle(char const* plist, int zOrder, cocos2d::tCCPositionType positionType, cocos2d::CCPoint position) {
    if (TrajectorySimulator::get()->isSimulating())
      return nullptr;
    return GJBaseGameLayer::spawnParticle(plist, zOrder, positionType, position);
  }
};

class $modify(TrajectoryHardStreakHook, HardStreak) {
  void addPoint(cocos2d::CCPoint p0) {
    if (TrajectorySimulator::get()->isSimulating())
      return;
    HardStreak::addPoint(p0);
  }
};

class $modify(TrajectoryAchievementHook, AchievementNotifier) {
  void notifyAchievement(char const *title, char const *desc, char const *icon, bool quest) {
    // suppress all achievement popups
  }
};

class $modify(TrajectoryGOHook, GameObject) {
  void playShineEffect() {
    if (TrajectorySimulator::get()->isSimulating())
      return;
    GameObject::playShineEffect();
  }
};

class $modify(TrajectoryRingHook, RingObject) {
  void spawnCircle() {
    if (TrajectorySimulator::get()->isSimulating())
      return;
    RingObject::spawnCircle();
  }
};

class $modify(TrajectoryPOHook, PlayerObject) {
  void playSpiderDashEffect(cocos2d::CCPoint from, cocos2d::CCPoint to) {
    if (TrajectorySimulator::get()->isSimulating())
      return;
    PlayerObject::playSpiderDashEffect(from, to);
  }

  void incrementJumps() {
    if (TrajectorySimulator::get()->isSimulating())
      return;
    PlayerObject::incrementJumps();
  }

  void spawnCircle() {
    if (TrajectorySimulator::get()->isSimulating())
      return;
    PlayerObject::spawnCircle();
  }

  void spawnCircle2() {
    if (TrajectorySimulator::get()->isSimulating())
      return;
    PlayerObject::spawnCircle2();
  }
};

