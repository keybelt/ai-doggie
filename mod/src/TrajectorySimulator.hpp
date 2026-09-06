#pragma once

#include <Geode/Geode.hpp>
#include <cocos2d.h>

using namespace geode::prelude;

class TrajectorySimulator {
public:
    static TrajectorySimulator* get();

    void init(PlayLayer* pl);
    void quit();
    void simulate(PlayLayer* pl);
    void hide();

    bool isSimulating() const { return m_simulating; }
    void handleDeath() { m_simulationDead = true; }
    void handleButton(bool down, bool isPlayer1);
    void setFrameDt(float dt) { m_rawDt = dt; }

private:
    void simulateBranch(PlayLayer* pl, bool down);

    cocos2d::CCDrawNode* m_drawNode = nullptr;
    bool m_simulating = false;
    bool m_simulationDead = false;
    bool m_player1Pressed = false;
    bool m_player2Pressed = false;
    size_t m_simFrameCount = 0;
    float m_rawDt = 1.0f / 240.0f;
    float m_frameDt = 1.0f / 240.0f;
    size_t m_iterations = 300;
};
