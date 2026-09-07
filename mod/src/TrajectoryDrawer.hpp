#pragma once

#include <Geode/Geode.hpp>
#include <cocos2d.h>
#include <vector>

using namespace geode::prelude;

struct TrajectoryBranch {
  std::vector<cocos2d::CCPoint> p1;
  std::vector<cocos2d::CCPoint> p2;
};

struct TrajectoryData {
  TrajectoryBranch releaseBranch;
  TrajectoryBranch holdBranch;
};

class TrajectoryDrawer {
public:
  static TrajectoryDrawer *get();

  void init(PlayLayer *pl);
  void quit();
  void clear();
  void render(PlayLayer *pl, TrajectoryData const &data);

private:
  void drawBranch(TrajectoryBranch const &branch, cocos2d::ccColor4F color);

  cocos2d::CCDrawNode *m_drawNode = nullptr;
};
