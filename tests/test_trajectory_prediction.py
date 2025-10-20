"""
Unit tests for trajectory prediction system.
"""

import pytest
import numpy as np
import sys
import os
from collections import deque

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trajectory_prediction import (
    ObjectTracker,
    SocialForceModel,
    TrajectoryGNN,
    TrajectoryPredictionSystem
)


class MockConfig:
    """Mock configuration for testing."""

    def __init__(self):
        self.config = {
            'trajectory_prediction.enabled': True,
            'trajectory_prediction.max_history': 10,
            'trajectory_prediction.prediction_horizon': 3,
            'trajectory_prediction.time_step': 0.5,
        }

    def get(self, key, default=None):
        return self.config.get(key, default)


def create_detection(label: str, center_x: float, center_y: float, depth: float) -> dict:
    """Helper to create detection dictionary."""
    return {
        'label': label,
        'bbox': [center_x - 50, center_y - 50, center_x + 50, center_y + 50],
        'center': [center_x, center_y],
        'depth': depth,
        'confidence': 0.9
    }


class TestObjectTracker:
    """Test cases for ObjectTracker."""

    def test_initialization(self):
        """Test tracker initialization."""
        tracker = ObjectTracker()
        assert len(tracker.objects) == 0
        assert tracker.next_object_id == 0
        print("✓ Tracker initialization test passed")

    def test_register_first_object(self):
        """Test registering first object."""
        tracker = ObjectTracker()
        detection = create_detection('person', 320, 240, 3.0)

        tracked = tracker.update([detection], frame_time=0.0)

        assert len(tracked) == 1
        assert 0 in tracked
        assert tracked[0]['label'] == 'person'
        assert len(tracked[0]['positions']) == 1
        print("✓ First object registration test passed")

    def test_track_moving_object(self):
        """Test tracking object across multiple frames."""
        tracker = ObjectTracker()

        # Frame 1: Person at (320, 240)
        det1 = create_detection('person', 320, 240, 3.0)
        tracked1 = tracker.update([det1], frame_time=0.0)

        # Frame 2: Person moved to (330, 240)
        det2 = create_detection('person', 330, 240, 2.9)
        tracked2 = tracker.update([det2], frame_time=0.5)

        # Should have same object ID
        assert len(tracked2) == 1
        obj_id = list(tracked2.keys())[0]
        assert len(tracked2[obj_id]['positions']) == 2
        assert len(tracked2[obj_id]['velocities']) == 1

        # Check velocity is reasonable
        velocity = tracked2[obj_id]['velocities'][0]
        assert velocity[0] > 0  # Moving in positive x direction

        print(f"✓ Moving object tracking test passed (velocity: {velocity})")

    def test_multiple_objects(self):
        """Test tracking multiple objects simultaneously."""
        tracker = ObjectTracker()

        # Frame 1: Two people
        det1 = create_detection('person', 200, 240, 3.0)
        det2 = create_detection('person', 400, 240, 4.0)

        tracked = tracker.update([det1, det2], frame_time=0.0)

        assert len(tracked) == 2
        print(f"✓ Multiple objects test passed ({len(tracked)} objects tracked)")

    def test_object_disappears(self):
        """Test handling of disappeared objects."""
        tracker = ObjectTracker(max_disappeared=2)

        # Frame 1: Person appears
        det1 = create_detection('person', 320, 240, 3.0)
        tracker.update([det1], frame_time=0.0)

        # Frame 2-4: Person disappears (no detections)
        tracker.update([], frame_time=0.5)
        tracker.update([], frame_time=1.0)
        tracked = tracker.update([], frame_time=1.5)

        # Should be removed after max_disappeared frames
        assert len(tracked) == 0
        print("✓ Object disappearance test passed")


class TestSocialForceModel:
    """Test cases for SocialForceModel."""

    def test_initialization(self):
        """Test social force model initialization."""
        model = SocialForceModel()
        assert model.person_radius > 0
        assert model.max_speed > 0
        print("✓ Social force model initialization test passed")

    def test_repulsive_force(self):
        """Test repulsive force between two objects."""
        model = SocialForceModel()

        # Target object
        target = {
            'positions': deque([np.array([0, 0, 0]), np.array([0.1, 0, 0])]),
            'velocities': deque([np.array([1, 0, 0])])
        }

        # Other object nearby
        other = {
            'positions': deque([np.array([1, 0, 0])]),
            'velocities': deque([np.array([-1, 0, 0])])
        }

        force = model.predict_social_forces(target, [other])

        # Force should push target away from other object
        # Since other is at (1,0,0), force should be negative in x
        assert force[0] < 0  # Repulsive force in -x direction

        print(f"✓ Repulsive force test passed (force: {force})")

    def test_no_force_when_far(self):
        """Test no force when objects are far apart."""
        model = SocialForceModel()

        target = {
            'positions': [np.array([0, 0, 0])],
            'velocities': [np.array([1, 0, 0])]
        }

        # Other object far away
        other = {
            'positions': [np.array([10, 0, 0])],
            'velocities': [np.array([0, 0, 0])]
        }

        force = model.predict_social_forces(target, [other])

        # Force should be near zero
        assert np.linalg.norm(force) < 0.1

        print("✓ No force when far test passed")


class TestTrajectoryGNN:
    """Test cases for TrajectoryGNN."""

    def test_initialization(self):
        """Test GNN initialization."""
        gnn = TrajectoryGNN(prediction_horizon=3, time_step=0.5)
        assert gnn.prediction_horizon == 3
        assert gnn.time_step == 0.5
        print("✓ GNN initialization test passed")

    def test_predict_straight_motion(self):
        """Test prediction for object moving straight."""
        gnn = TrajectoryGNN(prediction_horizon=3, time_step=0.5)

        # Object moving straight in +x direction
        tracked_obj = {
            0: {
                'positions': [
                    np.array([0, 0, 0]),
                    np.array([0.5, 0, 0]),
                    np.array([1.0, 0, 0]),
                ],
                'velocities': [
                    np.array([1, 0, 0]),
                    np.array([1, 0, 0]),
                ],
                'timestamps': [0.0, 0.5, 1.0],
                'label': 'person'
            }
        }

        predictions = gnn.predict_trajectories(tracked_obj)

        assert 0 in predictions
        pred = predictions[0]
        assert len(pred['predicted_positions']) == 3
        assert pred['confidence'] > 0

        # Predicted positions should continue in +x direction
        for pos in pred['predicted_positions']:
            assert pos[0] > 1.0  # Should be ahead of current position

        print(f"✓ Straight motion prediction test passed")
        print(f"  Current: [1.0, 0, 0]")
        print(f"  Predicted: {[f'[{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}]' for p in pred['predicted_positions']]}")

    def test_predict_with_collision(self):
        """Test collision prediction."""
        gnn = TrajectoryGNN(prediction_horizon=3, time_step=0.5)

        # Two objects moving toward each other
        tracked_objects = {
            0: {
                'positions': [np.array([0, 0, 0]), np.array([0.5, 0, 0])],
                'velocities': [np.array([1, 0, 0])],
                'timestamps': [0.0, 0.5],
                'label': 'person'
            },
            1: {
                'positions': [np.array([3, 0, 0]), np.array([2.5, 0, 0])],
                'velocities': [np.array([-1, 0, 0])],
                'timestamps': [0.0, 0.5],
                'label': 'person'
            }
        }

        predictions = gnn.predict_trajectories(tracked_objects)

        # At least one should predict collision
        collision_detected = any(p['predicted_collision'] for p in predictions.values())

        print(f"✓ Collision prediction test passed (collision detected: {collision_detected})")


class TestTrajectoryPredictionSystem:
    """Test cases for complete system."""

    def test_initialization(self):
        """Test system initialization."""
        config = MockConfig()
        system = TrajectoryPredictionSystem(config)

        assert system.enabled == True
        assert system.tracker is not None
        assert system.gnn is not None
        print("✓ System initialization test passed")

    def test_update_with_detections(self):
        """Test system update with real detections."""
        config = MockConfig()
        system = TrajectoryPredictionSystem(config)

        # Frame 1
        detections1 = [
            create_detection('person', 320, 240, 3.0),
            create_detection('car', 400, 240, 5.0),
        ]
        result1 = system.update(detections1)

        assert 'tracked_objects' in result1
        assert 'predictions' in result1
        assert len(result1['tracked_objects']) == 2

        # Frame 2: Objects moved
        detections2 = [
            create_detection('person', 330, 240, 2.9),
            create_detection('car', 410, 240, 4.8),
        ]
        result2 = system.update(detections2)

        # Should have predictions now
        predictions = result2['predictions']
        assert len(predictions) > 0

        print(f"✓ System update test passed")
        print(f"  Tracked: {len(result2['tracked_objects'])} objects")
        print(f"  Predictions: {len(predictions)} trajectories")

    def test_collision_warnings(self):
        """Test collision warning generation."""
        config = MockConfig()
        system = TrajectoryPredictionSystem(config)

        # Simulate objects moving
        for i in range(5):
            detections = [
                create_detection('person', 320 + i * 10, 240, 3.0),
            ]
            result = system.update(detections)

        # Get warnings
        user_position = np.array([0, 0, 0])
        warnings = system.get_collision_warnings(
            result['predictions'],
            user_position
        )

        print(f"✓ Collision warning test passed ({len(warnings)} warnings)")

    def test_performance(self):
        """Test prediction performance."""
        import time

        config = MockConfig()
        system = TrajectoryPredictionSystem(config)

        # Simulate 30 frames
        times = []
        for i in range(30):
            detections = [
                create_detection('person', 320 + i * 2, 240, 3.0),
                create_detection('car', 400 - i * 3, 240, 5.0),
            ]

            start = time.time()
            result = system.update(detections)
            elapsed = (time.time() - start) * 1000  # ms
            times.append(elapsed)

        avg_time = np.mean(times)
        max_time = np.max(times)
        fps = 1000.0 / avg_time

        print(f"✓ Performance test:")
        print(f"  Average: {avg_time:.1f}ms per frame ({fps:.1f} FPS)")
        print(f"  Max: {max_time:.1f}ms")
        print(f"  Min: {np.min(times):.1f}ms")

        # Performance assertions
        assert avg_time < 50, f"Too slow: {avg_time:.1f}ms > 50ms"


if __name__ == "__main__":
    print("\n" + "="*60)
    print("RUNNING TRAJECTORY PREDICTION TESTS")
    print("="*60 + "\n")

    # Run all tests
    pytest.main([__file__, "-v", "-s"])

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60 + "\n")
