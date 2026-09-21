#pragma once

#include <Geode/Geode.hpp>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace geode::prelude;

constexpr int FRAME_WIDTH = 640;
constexpr int FRAME_HEIGHT = 480;
constexpr int FRAME_CHANNELS = 3;
constexpr int FRAME_BUFFER_SIZE = FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS;
constexpr int MAX_ACTIONS = 1024;
constexpr float END_WALL_DIST_TOLERANCE = 500.0f;

struct SharedData {
    volatile int32_t dataReadyBin;          // 1=ready for Python, 0=consumed by Python, -1=closed
    volatile float ftd;                     // Raw frames to death (target)
    volatile float vx;                      // Player horizontal velocity
    volatile float vy;                      // Player vertical velocity (m_yAccel)
    volatile float gravityDir;              // 1.0=normal, -1.0=inverted
    volatile int32_t isHolding;             // 1 if jump held at spawn, 0 otherwise
    volatile int32_t actionLength;          // Number of actions in actionsBuffer
    int8_t actionsBuffer[MAX_ACTIONS];      // Recorded action sequence (0=release, 1=jump)
    uint8_t frameBuffer[FRAME_BUFFER_SIZE]; // RGB pixels of I_0 at spawn (921,600 bytes)
};

class DataCollector {
  public:
    static void setData(SharedData *d) { s_data = d; }

    static void resetPassState() {
        s_isBackward = false;
        s_isPerturbed = false;
        s_rolloutActive = false;
        s_edgeX = 0.0f;
        s_frame = 0;
        s_maxFrames = 0;
        s_perturbFrame = -1;
        s_macroTape.clear();
        s_checkpointFrameDeltas.clear();
    }

    static bool isBackward() { return s_isBackward; }

    static void setBackward(bool b) { s_isBackward = b; }

    static void forward(PlayLayer *pl) {
        auto p1 = pl->m_player1;

        // Record native 240Hz button state directly into s_macroTape
        int8_t isHeld = p1->m_holdingButtons[static_cast<int>(PlayerButton::Jump)] ? 1 : 0;
        s_macroTape.push_back(isHeld);

        if (s_edgeX == 0.0f) {
            addCheckpoint(pl);
            s_edgeX = getScreenEdgeGameX(pl);
            s_frame = 0;
            return;
        }

        s_frame++;

        if ((pl->m_levelLength - p1->getPositionX()) <= END_WALL_DIST_TOLERANCE) {
            pl->removeCheckpoint(false);
            s_isBackward = true;
            s_data->dataReadyBin = 2;
            return;
        }

        bool reachedEdge = p1->m_isGoingLeft ? (p1->getPositionX() <= s_edgeX) : (p1->getPositionX() >= s_edgeX);

        if (reachedEdge || s_frame >= (MAX_ACTIONS - 1)) {
            addCheckpoint(pl);
            s_checkpointFrameDeltas.push_back(s_frame);
            s_frame = 0;
            s_edgeX = getScreenEdgeGameX(pl);
        }
    }

    static void onPlayerDied(PlayLayer *pl) {
        if (s_rolloutActive && s_isBackward) {
            finishRollout(pl, static_cast<float>(s_frame));
        }
    }

    static void finishRollout(PlayLayer *pl, float ftd) {
        s_rolloutActive = false;

        if (s_data) {
            s_data->ftd = ftd;
            s_data->actionLength = s_maxFrames;
            s_data->dataReadyBin = 1;

            while (s_data && s_data->dataReadyBin == 1) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }

        if (!s_isPerturbed) {
            s_isPerturbed = true;
        } else {
            pl->removeCheckpoint(false);
            s_checkpointFrameDeltas.pop_back();
            s_isPerturbed = false;

            // Discard the last backward segment (CP 0 spawn): quit before rolling it
            if (s_checkpointFrameDeltas.size() <= 1) {
                s_isBackward = false;
                resetPassState();
                pl->onQuit();
                return;
            }
        }

        startRollout(pl);
        s_rolloutActive = true;
    }

    static void backward(PlayLayer *pl) {
        if (!s_rolloutActive) {
            startRollout(pl);
            s_rolloutActive = true;
        }

        if (stepRollout(pl)) {
            finishRollout(pl, static_cast<float>(s_maxFrames));
        }
    }

  private:
    static float getScreenEdgeGameX(PlayLayer *pl) {
        float cameraX = pl->m_gameState.m_cameraPosition.x;
        float winWidth = cocos2d::CCDirector::sharedDirector()->getWinSize().width;
        float zoom = pl->m_gameState.m_cameraZoom;
        float cameraWidth = winWidth / (zoom > 0.0f ? zoom : 1.0f);
        return pl->m_player1->m_isGoingLeft ? cameraX : (cameraX + cameraWidth);
    }

    static void addCheckpoint(PlayLayer *pl) {
        CheckpointObject *cp = pl->createCheckpoint();
        if (cp) {
            if (cp->m_physicalCheckpointObject) {
                cp->m_physicalCheckpointObject->setVisible(false);
            }
            pl->storeCheckpoint(cp);
        }
    }

    static void startRollout(PlayLayer *pl) {
        pl->resetLevel();
        pl->loadLastCheckpoint();
        s_frame = 0;
        s_maxFrames = std::clamp(s_checkpointFrameDeltas.back(), 1, MAX_ACTIONS - 1);
        s_perturbFrame = s_isPerturbed ? (rand() % s_maxFrames) : -1;
    }

    static bool stepRollout(PlayLayer *pl) {
        auto p1 = pl->m_player1;
        auto p2 = pl->m_gameState.m_isDualMode ? pl->m_player2 : nullptr;

        // Capture I_0 and initial telemetry at frame 0 of golden rollout
        if (!s_isPerturbed && s_frame == 0 && s_data) {
            glReadPixels(0, 0, FRAME_WIDTH, FRAME_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, (void *)s_data->frameBuffer);
            s_data->vx = p1->m_isGoingLeft ? -p1->m_playerSpeed : p1->m_playerSpeed;
            s_data->vy = static_cast<float>(p1->m_yVelocity);
            s_data->gravityDir = p1->m_isUpsideDown ? -1.0f : 1.0f;
            s_data->isHolding = p1->m_holdingButtons[static_cast<int>(PlayerButton::Jump)] ? 1 : 0;
        }

        // Current 240Hz tick index = m_currentProgress / 2
        size_t tick240 = static_cast<size_t>(pl->m_gameState.m_currentProgress / 2);
        bool shouldHold = (s_macroTape[tick240] == 1);

        // Apply perturbation if this is the perturbation frame
        if (s_isPerturbed && s_frame == s_perturbFrame) {
            shouldHold = !shouldHold;
        }

        bool currentlyHeld = p1->m_holdingButtons[static_cast<int>(PlayerButton::Jump)];
        if (shouldHold && !currentlyHeld) {
            p1->pushButton(PlayerButton::Jump);
            if (p2)
                p2->pushButton(PlayerButton::Jump);
        } else if (!shouldHold && currentlyHeld) {
            p1->releaseButton(PlayerButton::Jump);
            if (p2)
                p2->releaseButton(PlayerButton::Jump);
        }

        // Record the live button state into shared memory for Python
        if (s_data) {
            s_data->actionsBuffer[s_frame] = p1->m_holdingButtons[static_cast<int>(PlayerButton::Jump)] ? 1 : 0;
        }
        s_frame++;

        return s_frame >= s_maxFrames;
    }

    inline static SharedData *s_data = nullptr;
    inline static bool s_isBackward = false;
    inline static bool s_isPerturbed = false;
    inline static bool s_rolloutActive = false;
    inline static float s_edgeX = 0.0f;
    inline static int s_frame = 0;
    inline static int s_maxFrames = 0;
    inline static int s_perturbFrame = -1;

    inline static std::vector<int8_t> s_macroTape;
    inline static std::vector<int> s_checkpointFrameDeltas;
};
