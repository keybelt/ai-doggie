#pragma once

#include <Geode/Geode.hpp>

using namespace geode::prelude;

namespace TrajectorySim {
void init(PlayLayer *pl);
void quit();
void handleButtonPress(bool down, bool isPlayer1);
void handleSimulationDeath(PlayerObject *player);
bool isSimulating();
void simulate(PlayLayer *pl);
} // namespace TrajectorySim
