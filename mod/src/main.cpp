#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>

#ifdef GEODE_IS_WINDOWS
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

using namespace geode::prelude;

struct BotSharedData {
    int32_t current_frame;
    int32_t current_action;
    int32_t frame_ready;
    int32_t action_ready;
};

BotSharedData* sharedData = nullptr;
bool last_jump_state = false;

#ifdef GEODE_IS_WINDOWS
HANDLE hMapFile = NULL;
#else
int shm_fd = -1;
#endif

void initSharedMemory() {
    if (sharedData) return;

#ifdef GEODE_IS_WINDOWS
    hMapFile = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, "GDBotMem");
    if (!hMapFile) hMapFile = OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, "wnsm_GDBotMem");

    if (hMapFile) {
        sharedData = (BotSharedData*)MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(BotSharedData));
    }
#else
    shm_fd = shm_open("/GDBotMem", O_RDWR, 0666);
    if (shm_fd != -1) {
        sharedData = (BotSharedData*)mmap(NULL, sizeof(BotSharedData), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    }
#endif
}

void closeSharedMemory() {
    if (!sharedData) return;
#ifdef GEODE_IS_WINDOWS
    UnmapViewOfFile(sharedData);
    CloseHandle(hMapFile);
    hMapFile = NULL;
#else
    munmap(sharedData, sizeof(BotSharedData));
    close(shm_fd);
    shm_fd = -1;
#endif
    sharedData = nullptr;
}


class $modify(MyPlayLayer, PlayLayer) {
    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) return false;

        initSharedMemory();
        last_jump_state = false;

        return true;
    }

    void onQuit() {
        closeSharedMemory();
        PlayLayer::onQuit();
    }
};

class $modify(MyBGL, GJBaseGameLayer) {
    void simulateClick(PlayerButton button, bool down, bool player2) {
        auto performButton = down ? &PlayerObject::pushButton : &PlayerObject::releaseButton;

        if (m_levelSettings->m_twoPlayerMode && m_gameState.m_isDualMode) {
            PlayerObject* targetPlayer = player2 ? m_player2 : m_player1;
            if (targetPlayer) (targetPlayer->*performButton)(button);
        } else {
            if (m_player1) (m_player1->*performButton)(button);
            if (m_gameState.m_isDualMode && m_player2) {
                (m_player2->*performButton)(button);
            }
        }

        m_effectManager->playerButton(down, !player2);

        if (down) {
            m_clicks++;
            if (button == PlayerButton::Jump) m_jumping = true;
        }
    }

    void processBotInput() {
        if (!m_player1) return;

        if (!sharedData) {
            initSharedMemory();
            if (!sharedData) return;
        }

        int current_frame = m_gameState.m_currentProgress / 2;

        sharedData->action_ready = 0;
        sharedData->current_frame = current_frame;
        sharedData->frame_ready = 1;

        int timeout = 50000;
        while (sharedData->action_ready == 0 && timeout > 0) {
            timeout--;
        }

        if (timeout > 0) {
            bool should_jump = (sharedData->current_action == 1);

            if (should_jump && !last_jump_state) {
                simulateClick(PlayerButton::Jump, true, false);
                last_jump_state = true;
            } else if (!should_jump && last_jump_state) {
                simulateClick(PlayerButton::Jump, false, false);
                last_jump_state = false;
            }
        }
    }

#ifndef GEODE_IS_MACOS
    void processCommands(float dt, bool isHalfTick, bool isLastTick) {
        GJBaseGameLayer::processCommands(dt, isHalfTick, isLastTick);
        this->processBotInput();
    }
#else
    void processQueuedButtons(float dt, bool clearInputQueue) {
        GJBaseGameLayer::processQueuedButtons(dt, clearInputQueue);
        this->processBotInput();
    }
#endif
};