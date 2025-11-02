"""
OrbyGlasses - Path Planning Stub

This module was previously a Reinforcement Learning based path planner.
It has been disabled and replaced with a stub since the feature was not used.

The actual navigation is handled by:
- src/navigation/indoor_navigation.py (A* path planning)
- Indoor navigation is already integrated and active
"""

import logging
from typing import Dict, Optional


class PathPlanner:
    """
    Stub path planner that does nothing.

    This was previously an RL-based predictor but has been disabled
    since indoor_navigation.py provides A* path planning which is
    already integrated and working.
    """

    def __init__(self, config: Dict):
        """Initialize path planner stub."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.enabled = False  # Always disabled
        self.predictor = PathPredictorStub()

        # Don't log anything - this is intentionally disabled

    def predict_path(self, detections, slam_position):
        """
        Stub method - returns None.

        Actual path planning is handled by IndoorNavigator.
        """
        return None

    def update(self, detections, slam_position):
        """Stub method - does nothing."""
        pass


class PathPredictorStub:
    """Stub predictor that does nothing."""

    def __init__(self):
        """Initialize predictor stub."""
        pass

    def train(self):
        """Stub method - does nothing."""
        pass

    def predict(self, observation):
        """Stub method - returns None."""
        return None
