#include "TrajectoryDrawer.hpp"

TrajectoryDrawer *TrajectoryDrawer::get() {
  static TrajectoryDrawer instance;
  return &instance;
}

void TrajectoryDrawer::init(PlayLayer *pl) {
  if (!pl)
    return;

  if (!m_drawNode) {
    m_drawNode = cocos2d::CCDrawNode::create();
    m_drawNode->retain();
  }

  if (m_drawNode->getParent()) {
    m_drawNode->removeFromParent();
  }

  if (pl->m_debugDrawNode && pl->m_debugDrawNode->getParent()) {
    pl->m_debugDrawNode->getParent()->addChild(m_drawNode);
    m_drawNode->setZOrder(pl->m_debugDrawNode->getZOrder());
  } else {
    pl->addChild(m_drawNode, 100);
  }

  m_drawNode->setVisible(false);
}

void TrajectoryDrawer::quit() {
  if (m_drawNode) {
    m_drawNode->removeFromParent();
    m_drawNode->release();
    m_drawNode = nullptr;
  }
}

void TrajectoryDrawer::clear() {
  if (m_drawNode) {
    m_drawNode->clear();
  }
}

void TrajectoryDrawer::drawBranch(TrajectoryBranch const &branch, cocos2d::ccColor4F color) {
  if (!m_drawNode)
    return;

  for (size_t i = 1; i < branch.p1.size(); ++i) {
    m_drawNode->drawSegment(branch.p1[i - 1], branch.p1[i], 0.65f, color);
  }
  for (size_t i = 1; i < branch.p2.size(); ++i) {
    m_drawNode->drawSegment(branch.p2[i - 1], branch.p2[i], 0.65f, color);
  }
}

void TrajectoryDrawer::render(PlayLayer *pl, TrajectoryData const &data) {
  if (!m_drawNode || !m_drawNode->getParent()) {
    init(pl);
  }

  if (!m_drawNode)
    return;

  m_drawNode->setVisible(true);
  m_drawNode->clear();

  // Release branch (Red)
  drawBranch(data.releaseBranch, cocos2d::ccColor4F{1.f, 0.f, 0.1f, 1.f});

  // Hold branch (Green)
  drawBranch(data.holdBranch, cocos2d::ccColor4F{0.f, 1.f, 0.1f, 1.f});

  // Impulse branch (Yellow)
  drawBranch(data.impulseBranch, cocos2d::ccColor4F{1.f, 0.9f, 0.f, 1.f});
}
