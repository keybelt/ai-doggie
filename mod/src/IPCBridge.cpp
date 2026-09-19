#include <Geode/Geode.hpp>
#include <Geode/modify/CCScheduler.hpp>
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
    cocos2d::CCAffineTransform worldToNode = pl->m_objectLayer->worldToNodeTransform();
    cocos2d::CCSize winSize = cocos2d::CCDirector::sharedDirector()->getWinSize();

    bool goingLeft = pl->m_player1->m_isGoingLeft;
    cocos2d::CCPoint screenEdge = cocos2d::CCPoint(goingLeft ? 0.0f : winSize.width, 0.0f);
    cocos2d::CCPoint gameEdge = cocos2d::CCPointApplyAffineTransform(screenEdge, worldToNode);

    return gameEdge.x;
}

static void addCheckpoint(PlayLayer *pl) {
    CheckpointObject *cp = pl->createCheckpoint();
    if (cp) {
        cp->retain();
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
static int s_perturbFrame = -1;
static std::vector<int> s_checkpointFrames;

static void resetPassState() {
    s_isBackward = false;
    s_isPerturbed = false;
    s_rolloutActive = false;
    s_edgeX = 0.0f;
    s_frame = 0;
    s_perturbFrame = -1;
    s_checkpointFrames.clear();
}

static void startRollout(PlayLayer *pl) {
    auto p1 = pl->m_player1;
    auto p2 = pl->m_gameState.m_isDualMode ? pl->m_player2 : nullptr;

    pl->loadLastCheckpoint();
    p1->m_isDead = false;
    if (p2)
        p2->m_isDead = false;

    s_frame = 0;
    s_edgeX = getScreenEdgeGameX(pl);

    int maxFrames = std::clamp(s_checkpointFrames.back(), 1, MAX_ACTIONS - 1);

    s_perturbFrame = s_isPerturbed ? (rand() % maxFrames) : -1;
}

static void forward(PlayLayer *pl) {
    auto p1 = pl->m_player1;
    auto p2 = pl->m_gameState.m_isDualMode ? pl->m_player2 : nullptr;

    // Pause the game if the player dies during the forward pass
    if (p1->m_isDead || (p2 && p2->m_isDead)) {
        if (!pl->m_isPaused) {
            pl->pauseGame(false);
        }
        return;
    }

    s_frame++;

    bool reachedEdge = p1->m_isGoingLeft ? (p1->getPositionX() <= s_edgeX) : (p1->getPositionX() >= s_edgeX);

    // Drop checkpoint initially or when reaching screen edge
    if (reachedEdge) {
        addCheckpoint(pl);
        s_checkpointFrames.push_back(s_frame);
        s_frame = 0;
        s_edgeX = getScreenEdgeGameX(pl);
    }

    // When level completes, prepare for backward pass
    if (pl->m_hasCompletedLevel) {
        s_checkpointFrames.push_back(s_frame);
        s_isBackward = true;
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

    bool died = p1->m_isDead || (p2 && p2->m_isDead);
    bool reachedEdge = p1->m_isGoingLeft ? (p1->getPositionX() <= s_edgeX) : (p1->getPositionX() >= s_edgeX);

    int maxFrames = std::clamp(s_checkpointFrames.back(), 1, MAX_ACTIONS - 1);

    if (died || reachedEdge || s_frame >= maxFrames) {
        outFtd = died ? static_cast<float>(s_frame) : static_cast<float>(maxFrames);
        outActionLength = maxFrames;
        return true;
    }

    return false;
}

static void backward(PlayLayer *pl) {
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
        pl->removeCheckpoint(false);
        s_checkpointFrames.pop_back();
        s_isPerturbed = false;
        if (pl->m_checkpointArray->count() == 0) {
            s_isBackward = false;
        }
    }
}

class $modify(MyPlayLayer, PlayLayer) {
    void setupSession() {
        cocos2d::CCDirector::sharedDirector()->setAnimationInterval(1.0 / 60.0);
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

    void onQuit() {
        closeShm();
        PlayLayer::onQuit();
    }

    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);
        if (!m_started || m_isPaused || !m_player1 || !data)
            return;
        if (data->dataReadyBin == -1) {
            closeShm();
            return;
        }

        if (!s_isBackward) {
            forward(this);
        } else {
            backward(this);
        }
    }
};

class $modify(MyScheduler, cocos2d::CCScheduler) {
    void update(float dt) {
        if (auto pl = PlayLayer::get(); pl && !pl->m_isPaused) {
            dt = 1.0f / 60.0f;
        }
        CCScheduler::update(dt);
    }
};
