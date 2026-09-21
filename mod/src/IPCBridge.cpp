#include "DataCollector.hpp"

#include <Geode/Geode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace geode::prelude;

constexpr const char *SHM_NAME = "/GDMem";
static SharedData *s_data = nullptr;
static int s_fd = -1;

static void closeShm() {
    if (!s_data)
        return;
    s_data->dataReadyBin = -1;
    munmap(s_data, sizeof(SharedData));
    close(s_fd);
    s_fd = -1;
    s_data = nullptr;
    DataCollector::setData(nullptr);
}

static void initShm() {
    if (s_data)
        return;

    s_fd = shm_open(SHM_NAME, O_RDWR, 0666);
    if (s_fd != -1) {
        s_data = (SharedData *)mmap(NULL, sizeof(SharedData), PROT_READ | PROT_WRITE, MAP_SHARED, s_fd, 0);
        if (s_data == MAP_FAILED) {
            s_data = nullptr;
            close(s_fd);
            s_fd = -1;
        }
    }
    DataCollector::setData(s_data);
}

static void setupSession(PlayLayer *pl) {
    initShm();
    if (!s_data)
        return;
    pl->m_isPracticeMode = true;
    DataCollector::resetPassState();
}

class $modify(MyBaseGameLayer, GJBaseGameLayer) {
    void processQueuedButtons(float dt, bool clearInputQueue) {
        GJBaseGameLayer::processQueuedButtons(dt, clearInputQueue);

        auto pl = PlayLayer::get();
        if (!pl || !s_data || pl->m_isPaused || !pl->m_player1)
            return;
        if (s_data->dataReadyBin == -1) {
            closeShm();
            return;
        }

        if (!DataCollector::isBackward()) {
            DataCollector::forward(pl);
        } else {
            DataCollector::backward(pl);
        }
    }
};

class $modify(MyPlayLayer, PlayLayer) {
    bool init(GJGameLevel *level, bool useReplay, bool dontCreateObjects) {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) {
            return false;
        }
        setupSession(this);
        return true;
    }

    void resetLevel() {
        bool wasBackward = DataCollector::isBackward();
        PlayLayer::resetLevel();
        if (!wasBackward) {
            setupSession(this);
        }
    }

    void destroyPlayer(PlayerObject *player, GameObject *gameObject) {
        if (!DataCollector::isBackward()) {
            PlayLayer::destroyPlayer(player, gameObject);
            if (m_playerDied) {
                this->pauseGame(false);
            }
        } else {
            DataCollector::onPlayerDied(this);
        }
    }

    void onQuit() {
        closeShm();
        PlayLayer::onQuit();
    }
};
