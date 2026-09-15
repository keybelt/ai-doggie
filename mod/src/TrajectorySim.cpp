#include "TrajectorySim.hpp"
#include "TrajectoryDrawer.hpp"

#include <Geode/Geode.hpp>
#include <Geode/modify/AchievementNotifier.hpp>
#include <Geode/modify/CCDrawNode.hpp>
#include <Geode/modify/CCNode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>
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

  branch.p1.push_back(p1->getPosition());
  if (p2) {
    branch.p2.push_back(p2->getPosition());
  }

  cocos2d::CCNode *parent = p1->getParent() ? p1->getParent() : pl->m_objectLayer;
  cocos2d::CCAffineTransform toWorld =
      parent ? parent->nodeToWorldTransform() : cocos2d::CCAffineTransformMakeIdentity();
  cocos2d::CCSize winSize = cocos2d::CCDirector::sharedDirector()->getWinSize();

  auto isPastScreenHorizontal = [&](PlayerObject *player) -> bool {
    cocos2d::CCPoint screenPos = cocos2d::CCPointApplyAffineTransform(player->getPosition(), toWorld);
    if (player->m_isGoingLeft) {
      return screenPos.x < 0.0f;
    }
    return screenPos.x > winSize.width;
  };

  for (size_t i = 0;; ++i) {
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

    bool p1Done = isPastScreenHorizontal(p1) || p1->m_isDead;
    bool p2Done = !p2 || isPastScreenHorizontal(p2) || p2->m_isDead;
    if (p1Done && p2Done) {
      break;
    }
  }

  return branch;
}

static cocos2d::CCDrawNode *s_dummyDrawNode = nullptr;

cocos2d::CCDrawNode *getDummyDrawNode() {
  if (!s_dummyDrawNode) {
    s_dummyDrawNode = cocos2d::CCDrawNode::create();
    s_dummyDrawNode->retain();
    s_dummyDrawNode->setVisible(false);
  }
  return s_dummyDrawNode;
}

struct DebugDrawGuard {
  PlayLayer *pl;
  cocos2d::CCDrawNode *realNode;
  cocos2d::CCDrawNode *dummyNode;

  DebugDrawGuard(PlayLayer *p)
      : pl(p), realNode(p ? p->m_debugDrawNode : nullptr), dummyNode(getDummyDrawNode()) {
    if (pl && realNode) {
      pl->m_debugDrawNode = dummyNode;
    }
  }

  ~DebugDrawGuard() {
    if (pl && realNode) {
      pl->m_debugDrawNode = realNode;
    }
    if (dummyNode) {
      dummyNode->clear();
    }
  }
};

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

  TrajectoryData data;
  {
    // RAII guard swaps pl->m_debugDrawNode with an unparented dummy CCDrawNode for the
    // entire duration of simulation, ensuring realNode is completely untouched.
    DebugDrawGuard drawGuard(pl);
    s_simulating = true;

    // 2. Step forward headlessly & reset level to checkpoint
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

    // Crucial: loadLastCheckpoint restored m_position, but did NOT recompute m_orientedBox!
    // Without this, m_orientedBox remains stuck at the simulated death coordinates, causing
    // Eclipse's rotated player hitbox to render at the death point (phantom dummy hitbox)
    // while the live player loses the fill alpha contributed by the rotated box.
    p1->updateOrientedBox();
    if (p2) {
      p2->updateOrientedBox();
    }

    s_simulating = false;
  }

  // Refresh Eclipse hitboxes for the live frame:
  // updateVisibility(0.0f) triggers Eclipse's ShowHitboxesPLHook::updateVisibility ->
  // visitHitboxes(), which clears Eclipse's drawNode and redraws all hitboxes cleanly
  // at the restored live player position and newly updated orientedBox.
  pl->updateVisibility(0.0f);

  TrajectoryDrawer::get()->render(pl, data);
}
} // namespace TrajectorySim

// ================= Suppression & State Restoration Hooks =================

class $modify(TrajectoryCCDNHook, cocos2d::CCDrawNode) {
  static void onModify(auto &self) {
    (void)self.setHookPriority("cocos2d::CCDrawNode::drawPolygon", geode::Priority::First);
    (void)self.setHookPriority("cocos2d::CCDrawNode::drawRect", geode::Priority::First);
    (void)self.setHookPriority("cocos2d::CCDrawNode::drawCircle", geode::Priority::First);
    (void)self.setHookPriority("cocos2d::CCDrawNode::drawSegment", geode::Priority::First);
    (void)self.setHookPriority("cocos2d::CCDrawNode::drawLines", geode::Priority::First);
    (void)self.setHookPriority("cocos2d::CCDrawNode::clear", geode::Priority::First);
    (void)self.setHookPriority("cocos2d::CCDrawNode::setVisible", geode::Priority::First);
  }

  void setVisible(bool visible) {
    if (TrajectorySim::isSimulating() && this != TrajectorySim::getDummyDrawNode()) {
      return;
    }
    CCDrawNode::setVisible(visible);
  }

  void clear() {
    if (TrajectorySim::isSimulating() && this != TrajectorySim::getDummyDrawNode()) {
      return;
    }
    CCDrawNode::clear();
  }

  bool drawPolygon(cocos2d::CCPoint *vertex, unsigned int count, const cocos2d::ccColor4F &fillColor,
                   float borderWidth, const cocos2d::ccColor4F &borderColor, cocos2d::BorderAlignment alignment) {
    if (TrajectorySim::isSimulating() && this != TrajectorySim::getDummyDrawNode()) {
      return true;
    }
    return CCDrawNode::drawPolygon(vertex, count, fillColor, borderWidth, borderColor, alignment);
  }

  bool drawRect(const cocos2d::CCRect &rect, const cocos2d::ccColor4F &fillColor, float borderWidth,
                const cocos2d::ccColor4F &borderColor, cocos2d::BorderAlignment alignment) {
    if (TrajectorySim::isSimulating() && this != TrajectorySim::getDummyDrawNode()) {
      return true;
    }
    return CCDrawNode::drawRect(rect, fillColor, borderWidth, borderColor, alignment);
  }

  bool drawRect(const cocos2d::CCPoint &from, const cocos2d::CCPoint &to, const cocos2d::ccColor4F &fillColor,
                float borderWidth, const cocos2d::ccColor4F &borderColor, cocos2d::BorderAlignment alignment) {
    if (TrajectorySim::isSimulating() && this != TrajectorySim::getDummyDrawNode()) {
      return true;
    }
    return CCDrawNode::drawRect(from, to, fillColor, borderWidth, borderColor, alignment);
  }

  bool drawCircle(const cocos2d::CCPoint &position, float radius, const cocos2d::ccColor4F &color,
                  float borderWidth, const cocos2d::ccColor4F &borderColor, unsigned int segments) {
    if (TrajectorySim::isSimulating() && this != TrajectorySim::getDummyDrawNode()) {
      return true;
    }
    return CCDrawNode::drawCircle(position, radius, color, borderWidth, borderColor, segments);
  }

  bool drawSegment(const cocos2d::CCPoint &from, const cocos2d::CCPoint &to, float radius,
                   const cocos2d::ccColor4F &color) {
    if (TrajectorySim::isSimulating() && this != TrajectorySim::getDummyDrawNode()) {
      return true;
    }
    return CCDrawNode::drawSegment(from, to, radius, color);
  }

  bool drawLines(cocos2d::CCPoint *points, unsigned int count, float radius, const cocos2d::ccColor4F &color) {
    if (TrajectorySim::isSimulating() && this != TrajectorySim::getDummyDrawNode()) {
      return true;
    }
    return CCDrawNode::drawLines(points, count, radius, color);
  }
};

class $modify(TrajectoryPLHook, PlayLayer) {
  static void onModify(auto &self) {
    (void)self.setHookPriority("PlayLayer::updateProgressbar", geode::Priority::First);
    (void)self.setHookPriority("PlayLayer::updateVisibility", geode::Priority::First);
    (void)self.setHookPriority("PlayLayer::destroyPlayer", geode::Priority::First);
  }

  void updateProgressbar() {
    if (TrajectorySim::isSimulating()) {
      return;
    }
    PlayLayer::updateProgressbar();
  }

  void updateVisibility(float dt) {
    if (TrajectorySim::isSimulating()) {
      return;
    }
    PlayLayer::updateVisibility(dt);
  }

  void destroyPlayer(PlayerObject *player, GameObject *object) {
    if (TrajectorySim::isSimulating()) {
      TrajectorySim::handleSimulationDeath(player);
      return;
    }
    PlayLayer::destroyPlayer(player, object);
  }
};

class $modify(TrajectoryPOHook, PlayerObject) {
  static void onModify(auto &self) {
    (void)self.setHookPriority("PlayerObject::playerDestroyed", geode::Priority::First);
    (void)self.setHookPriority("PlayerObject::loadFromCheckpoint", geode::Priority::First);
  }

  void playerDestroyed(bool p0) {
    if (TrajectorySim::isSimulating()) {
      TrajectorySim::handleSimulationDeath(this);
      return;
    }
    PlayerObject::playerDestroyed(p0);
  }

  void loadFromCheckpoint(PlayerCheckpoint *cp) {
    PlayerObject::loadFromCheckpoint(cp);
    m_isDead = false;
    this->setPosition(this->m_position);

    bool isPressed = m_isSecondPlayer ? TrajectorySim::s_player2Pressed : TrajectorySim::s_player1Pressed;
    isPressed ? this->pushButton(PlayerButton::Jump) : this->releaseButton(PlayerButton::Jump);
  }
};

class $modify(TrajectoryNodeHook, cocos2d::CCNode) {
  void addChild(cocos2d::CCNode *child, int zOrder, int tag) {
    if (TrajectorySim::isSimulating()) {
      if (child) {
        child->setVisible(false);
      }
    }
    CCNode::addChild(child, zOrder, tag);
  }
};

class $modify(TrajectoryAchievementHook, AchievementNotifier) {
  void notifyAchievement(char const *title, char const *desc, char const *icon, bool quest) {
    // Suppress all achievement popups
  }
};

class $modify(TrajectoryBGLHook, GJBaseGameLayer) {
  static void onModify(auto &self) {
    (void)self.setHookPriority("GJBaseGameLayer::updateDebugDraw", geode::Priority::First);
  }

  void updateDebugDraw() {
    if (TrajectorySim::isSimulating()) {
      return;
    }
    GJBaseGameLayer::updateDebugDraw();
  }

  void updateTimeMod(float speed, bool players, bool noEffects) {
    if (TrajectorySim::isSimulating()) {
      if (m_player1) {
        m_player1->updateTimeMod(speed, true);
      }
      if (m_gameState.m_isDualMode && m_player2) {
        m_player2->updateTimeMod(speed, true);
      }
      return;
    }
    GJBaseGameLayer::updateTimeMod(speed, players, noEffects);
  }
};
