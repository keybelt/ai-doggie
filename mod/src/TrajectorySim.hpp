#pragma once

#include <Geode/Geode.hpp>

using namespace geode::prelude;

namespace TrajectorySim {
struct SimResult {
  float ttdRelease = 0.0f;
  float ttdHold = 0.0f;
  float ttdImpulse = 0.0f;
  float maxHorizon = 0.0f;
};

void init(PlayLayer *pl);
void quit();
void handleSimulationDeath(PlayerObject *player);
bool isSimulating();
SimResult simulate(PlayLayer *pl);
} // namespace TrajectorySim
