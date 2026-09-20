#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace geode::prelude;

constexpr const char *SHM_NAME = "/GDMem";
constexpr int FRAME_WIDTH = 640;
constexpr int FRAME_HEIGHT = 480;
constexpr int FRAME_CHANNELS = 3;
constexpr int FRAME_BUFFER_SIZE = FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS;

constexpr int MAX_ACTIONS = 256;
constexpr float END_WALL_DIST_TOLERANCE = 1000.0f;

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

static SharedData *data = nullptr;
static int fileDescriptor = -1;

static void closeShm() {
    if (!data)
        return;
    munmap(data, sizeof(SharedData));
    close(fileDescriptor);
    fileDescriptor = -1;
    data = nullptr;
}

static void initShm() {
    if (data)
        return;

    fileDescriptor = shm_open(SHM_NAME, O_RDWR, 0666);
    if (fileDescriptor != -1) {
        data = (SharedData *)mmap(NULL, sizeof(SharedData), PROT_READ | PROT_WRITE, MAP_SHARED, fileDescriptor, 0);
        if (data == MAP_FAILED) {
            data = nullptr;
            close(fileDescriptor);
            fileDescriptor = -1;
        }
    }
}

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

static bool s_isBackward = false;
static bool s_isPerturbed = false;
static bool s_rolloutActive = false;
static float s_edgeX = 0.0f;
static int s_frame = 0;
static int s_lastFrameIdx = -1;
static int s_maxFrames = 0;
static int s_perturbFrame = -1;
static std::vector<int> s_checkpointFrames;

static void resetPassState() {
    s_isBackward = false;
    s_isPerturbed = false;
    s_rolloutActive = false;
    s_edgeX = 0.0f;
    s_frame = 0;
    s_lastFrameIdx = -1;
    s_maxFrames = 0;
    s_perturbFrame = -1;
    s_checkpointFrames.clear();
}

static void startRollout(PlayLayer *pl) {
    pl->loadLastCheckpoint();
    s_frame = 0;
    s_maxFrames = std::clamp(s_checkpointFrames.back(), 1, MAX_ACTIONS - 1);
    s_perturbFrame = s_isPerturbed ? (rand() % s_maxFrames) : -1;
}

static void forward(PlayLayer *pl) {
    if (s_edgeX == 0.0f) {
        addCheckpoint(pl);
        s_edgeX = getScreenEdgeGameX(pl);
        s_frame = 0;
        return;
    }

    s_frame++;

    auto p1 = pl->m_player1;

    if (pl->m_levelLength > 0.0f && (pl->m_levelLength - p1->getPositionX()) <= END_WALL_DIST_TOLERANCE) {
        if (s_frame > 0) {
            s_checkpointFrames.push_back(s_frame);
        }
        s_isBackward = true;
        return;
    }

    bool reachedEdge = p1->m_isGoingLeft ? (p1->getPositionX() <= s_edgeX) : (p1->getPositionX() >= s_edgeX);

    if (reachedEdge || s_frame >= (MAX_ACTIONS - 1)) {
        addCheckpoint(pl);
        s_checkpointFrames.push_back(s_frame);
        s_frame = 0;
        s_edgeX = getScreenEdgeGameX(pl);
    }
}

// Executes one in-game tick of the rollout. Returns true when the rollout completes.
static bool stepRollout(PlayLayer *pl, float &outFtd, int &outActionLength) {
    auto p1 = pl->m_player1;
    auto p2 = pl->m_gameState.m_isDualMode ? pl->m_player2 : nullptr;

    // Capture I_0 and initial telemetry at frame 0 of golden rollout
    if (!s_isPerturbed && s_frame == 0) {
        glReadPixels(0, 0, FRAME_WIDTH, FRAME_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, (void *)data->frameBuffer);
        data->vx = p1->m_isGoingLeft ? -p1->m_playerSpeed : p1->m_playerSpeed;
        data->vy = static_cast<float>(p1->m_yVelocity);
        data->gravityDir = p1->m_isUpsideDown ? -1.0f : 1.0f;
        data->isHolding = p1->m_holdingButtons[static_cast<int>(PlayerButton::Jump)] ? 1 : 0;
    }

    // Apply perturbation: flip button state at perturbFrame
    if (s_frame == s_perturbFrame) {
        bool isHeld = p1->m_holdingButtons[static_cast<int>(PlayerButton::Jump)];
        if (isHeld) {
            p1->releaseButton(PlayerButton::Jump);
            if (p2)
                p2->releaseButton(PlayerButton::Jump);
        } else {
            p1->pushButton(PlayerButton::Jump);
            if (p2)
                p2->pushButton(PlayerButton::Jump);
        }
    }

    // Record action directly into shared memory buffer
    data->actionsBuffer[s_frame] = p1->m_holdingButtons[static_cast<int>(PlayerButton::Jump)] ? 1 : 0;
    s_frame++;

    bool died = pl->m_playerDied;

    if (died || s_frame >= s_maxFrames) {
        outFtd = died ? static_cast<float>(s_frame) : static_cast<float>(s_maxFrames);
        outActionLength = s_maxFrames;
        return true;
    }

    return false;
}

static void backward(PlayLayer *pl) {
    if (s_checkpointFrames.empty()) {
        s_isBackward = false;
        resetPassState();
        pl->levelComplete();
        return;
    }

    if (!s_rolloutActive) {
        startRollout(pl);
        s_rolloutActive = true;
    }

    float ftd = 0.0f;
    int actionLength = 0;
    if (!stepRollout(pl, ftd, actionLength)) {
        return;
    }

    s_rolloutActive = false;

    // End of rollout: send packet to Python
    data->ftd = ftd;
    data->actionLength = actionLength;
    std::atomic_thread_fence(std::memory_order_release);
    data->dataReadyBin = 1;
    while (data->dataReadyBin == 1) {
        if (data->dataReadyBin == -1) {
            closeShm();
            return;
        }
        std::this_thread::yield();
    }

    // Advance to perturbed rollout or pop checkpoint
    if (!s_isPerturbed) {
        s_isPerturbed = true;
    } else {
        pl->removeCheckpoint(true);
        s_checkpointFrames.pop_back();
        s_isPerturbed = false;
        if (s_checkpointFrames.empty()) {
            s_isBackward = false;
            resetPassState();
            pl->levelComplete();
        }
    }
}

class $modify(MyPlayLayer, PlayLayer) {
    void setupSession() {
        m_isPracticeMode = true;
        resetPassState();
        closeShm();
        initShm();
    }

    bool init(GJGameLevel *level, bool useReplay, bool dontCreateObjects) {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) {
            return false;
        }
        setupSession();
        return true;
    }

    void resetLevel() {
        PlayLayer::resetLevel();
        setupSession();
    }

    void destroyPlayer(PlayerObject *player, GameObject *gameObject) {
        PlayLayer::destroyPlayer(player, gameObject);
        if (m_playerDied && !s_isBackward && data) {
            this->pauseGame(false);
        }
    }

    void onQuit() {
        closeShm();
        PlayLayer::onQuit();
    }

    void levelComplete() {
        if (data && !s_isBackward) {
            if (s_frame > 0) {
                s_checkpointFrames.push_back(s_frame);
            }
            s_isBackward = true;
            return;
        }
        PlayLayer::levelComplete();
    }

    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);
        if (!m_started || m_isPaused || !m_player1 || !data)
            return;
        if (data->dataReadyBin == -1) {
            closeShm();
            return;
        }

        // GD 2.2 advances m_currentProgress by 8 per 60Hz frame (2 per 240Hz tick)
        int frame60Idx = (m_gameState.m_currentProgress / 2) / 4;
        if (frame60Idx == s_lastFrameIdx) {
            return;
        }
        s_lastFrameIdx = frame60Idx;

        if (!s_isBackward) {
            forward(this);
        } else {
            backward(this);
        }
    }
};
