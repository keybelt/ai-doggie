#include "TrajectorySim.hpp"

#include <Geode/Geode.hpp>
#include <Geode/modify/AchievementNotifier.hpp>
#include <Geode/modify/CCNode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/PlayerObject.hpp>

#include <algorithm>
#include <cmath>

using namespace geode::prelude;

namespace TrajectorySim {
static bool s_simulating = false;
static bool s_simulationDead = false;

// Geometry Dash 2.2 runs physics at 240 TPS, where dt is measured in 60Hz frame units:
// 60.0f / 240.0f = 0.25f per tick
static constexpr float BASE_FRAME_DT = 0.25f;

// Simulation ticks per 60Hz video frame (240 TPS / 60 FPS = 4 sub-ticks per frame).
// Dividing raw 240 TPS iterations by TICKS_PER_FRAME converts telemetry directly into 60Hz video frame units.
static constexpr float TICKS_PER_FRAME = 4.0f;

bool isSimulating() { return s_simulating; }

void init(PlayLayer *pl) {}

void quit() { s_simulating = false; }

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

static float simulateBranch(PlayLayer *pl, TrajectoryMode mode, float dt) {
    s_simulationDead = false;
    PlayerObject *p1 = pl->m_player1;
    PlayerObject *p2 = (pl->m_gameState.m_isDualMode ? pl->m_player2 : nullptr);
    if (!p1)
        return 0.0f;

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
        if (s_simulationDead || p1->m_isDead) {
            return static_cast<float>(i) / TICKS_PER_FRAME;
        }

        if (p2) {
            p2->update(dt);
            pl->checkCollisions(p2, dt, false);
            if (s_simulationDead || p2->m_isDead) {
                return static_cast<float>(i) / TICKS_PER_FRAME;
            }
        }

        bool p1Done = isPastScreenHorizontal(p1);
        bool p2Done = !p2 || isPastScreenHorizontal(p2);
        if (p1Done && p2Done) {
            return static_cast<float>(i + 1) / TICKS_PER_FRAME;
        }
    }
}

SimResult simulate(PlayLayer *pl) {
    SimResult result;
    if (!pl || !pl->m_player1 || s_simulating)
        return result;
    if (!pl->m_started || pl->m_isPaused || pl->m_playerDied || pl->m_hasCompletedLevel || pl->m_player1->m_isDead) {
        return result;
    }

    float warp = (pl->m_gameState.m_timeWarp > 0.f) ? pl->m_gameState.m_timeWarp : 1.f;
    float dt = BASE_FRAME_DT * warp;

    PlayerObject *p1 = pl->m_player1;
    PlayerObject *p2 = (pl->m_gameState.m_isDualMode ? pl->m_player2 : nullptr);

    // Preserve live holding buttons (the only input state not stored in PlayerCheckpoint)
    auto p1HoldingButtons = p1->m_holdingButtons;
    gd::map<int, bool> p2HoldingButtons;
    if (p2) {
        p2HoldingButtons = p2->m_holdingButtons;
    }

    bool wasPractice = pl->m_isPracticeMode;
    pl->m_isPracticeMode = true;

    // 1. Snapshot via practice checkpoint
    CheckpointObject *cp = pl->createCheckpoint();
    if (!cp) {
        pl->m_isPracticeMode = wasPractice;
        return result;
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

    // 2. Step forward headlessly & reset level to checkpoint for each action
    result.ttdRelease = simulateBranch(pl, TrajectoryMode::Release, dt);
    pl->resetLevel();
    pl->loadLastCheckpoint();
    p1->m_holdingButtons = p1HoldingButtons;
    p1->m_isDead = false;
    if (p2) {
        p2->m_holdingButtons = p2HoldingButtons;
        p2->m_isDead = false;
    }

    result.ttdHold = simulateBranch(pl, TrajectoryMode::Hold, dt);
    pl->resetLevel();
    pl->loadLastCheckpoint();
    p1->m_holdingButtons = p1HoldingButtons;
    p1->m_isDead = false;
    if (p2) {
        p2->m_holdingButtons = p2HoldingButtons;
        p2->m_isDead = false;
    }

    result.ttdImpulse = simulateBranch(pl, TrajectoryMode::Impulse, dt);
    pl->resetLevel();
    pl->loadLastCheckpoint();
    p1->m_holdingButtons = p1HoldingButtons;
    p1->m_isDead = false;
    if (p2) {
        p2->m_holdingButtons = p2HoldingButtons;
        p2->m_isDead = false;
    }

    // 3. Clean up snapshot checkpoint
    pl->removeCheckpoint(false);
    cp->release();
    pl->m_isPracticeMode = wasPractice;

    // Restore live player state
    p1->m_holdingButtons = p1HoldingButtons;
    p1->m_isDead = false;
    if (p2) {
        p2->m_holdingButtons = p2HoldingButtons;
        p2->m_isDead = false;
    }

    p1->m_playEffects = p1Effects;
    if (p2)
        p2->m_playEffects = p2Effects;

    p1->updateOrientedBox();
    if (p2) {
        p2->updateOrientedBox();
    }

    s_simulating = false;

    // Refresh Eclipse hitboxes for the live frame
    pl->updateVisibility(0.0f);

    return result;
}
} // namespace TrajectorySim

// ================= Suppression & State Restoration Hooks =================

class $modify(TrajectoryPLHook, PlayLayer) {
    static void onModify(auto &self) {
        (void)self.setHookPriority("PlayLayer::updateVisibility", geode::Priority::First);
        (void)self.setHookPriority("PlayLayer::destroyPlayer", geode::Priority::First);
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
