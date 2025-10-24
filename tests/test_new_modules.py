"""
Tests for new modules: YOLO-World, Depth Anything V2, Simple SLAM
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestYOLOWorldDetector:
    """Tests for YOLO-World detector."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        try:
            from core.yolo_world_detector import YOLOWorldDetector
            self.YOLOWorldDetector = YOLOWorldDetector
            self.detector = YOLOWorldDetector()
        except ImportError as e:
            pytest.skip(f"YOLO-World not available: {e}")

    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector is not None
        assert self.detector.model is not None
        assert self.detector.confidence >= 0 and self.detector.confidence <= 1

    def test_detect_with_empty_frame(self):
        """Test detection with empty frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = self.detector.detect(frame)

        assert isinstance(detections, list)
        # Empty frame should have no detections
        assert len(detections) == 0

    def test_detect_with_random_frame(self):
        """Test detection with random frame."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = self.detector.detect(frame)

        assert isinstance(detections, list)
        # Check detection format if any detections found
        if len(detections) > 0:
            det = detections[0]
            assert 'bbox' in det
            assert 'label' in det
            assert 'confidence' in det
            assert 'center' in det


class TestDepthAnythingV2:
    """Tests for Depth Anything V2."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        try:
            from core.depth_anything_v2 import DepthAnythingV2
            self.DepthAnythingV2 = DepthAnythingV2
            # Use small model for testing
            config = {'models': {'depth': {'size': 'small', 'device': 'cpu'}}}
            self.estimator = DepthAnythingV2(config)
        except ImportError as e:
            pytest.skip(f"Depth Anything V2 not available: {e}")

    def test_initialization(self):
        """Test estimator initialization."""
        assert self.estimator is not None
        assert self.estimator.model is not None
        assert self.estimator.processor is not None

    def test_estimate_depth_shape(self):
        """Test depth estimation output shape."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = self.estimator.estimate_depth(frame)

        assert depth.shape == (480, 640)
        assert depth.dtype == np.float32 or depth.dtype == np.float64

    def test_estimate_depth_range(self):
        """Test depth values are in reasonable range."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = self.estimator.estimate_depth(frame)

        # Depth should be positive and reasonable (0-10m for indoor)
        assert np.all(depth >= 0)
        assert np.all(depth <= 20)  # Allow some margin

    def test_get_depth_at_bbox(self):
        """Test depth extraction at bounding box."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = self.estimator.estimate_depth(frame)

        bbox = [100, 100, 200, 200]
        depth_value = self.estimator.get_depth_at_bbox(depth, bbox, (640, 480))

        assert isinstance(depth_value, float)
        assert depth_value >= 0

    def test_visualize_depth(self):
        """Test depth visualization."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth = self.estimator.estimate_depth(frame)

        colored = self.estimator.visualize_depth(depth)

        assert colored.shape == (480, 640, 3)
        assert colored.dtype == np.uint8


class TestSimpleSLAM:
    """Tests for Simple SLAM."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        from navigation.simple_slam import SimpleSLAM
        self.SimpleSLAM = SimpleSLAM
        self.slam = SimpleSLAM()

    def test_initialization(self):
        """Test SLAM initialization."""
        assert self.slam is not None
        assert self.slam.orb is not None
        assert len(self.slam.position) == 3
        assert self.slam.pose.shape == (4, 4)

    def test_process_first_frame(self):
        """Test processing first frame."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = self.slam.process_frame(frame)

        assert result is not None
        assert 'position' in result
        assert 'pose' in result
        assert 'tracking_quality' in result
        assert 'num_matches' in result
        assert result['is_keyframe'] == True

    def test_process_multiple_frames(self):
        """Test processing multiple frames."""
        for i in range(5):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            result = self.slam.process_frame(frame)

            assert result is not None
            assert len(result['position']) == 3
            assert result['pose'].shape == (4, 4)

    def test_reset(self):
        """Test SLAM reset."""
        # Process a frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.slam.process_frame(frame)

        # Reset
        self.slam.reset()

        # Check state is reset
        assert np.allclose(self.slam.position, [0, 0, 0])
        assert np.allclose(self.slam.pose, np.eye(4))
        assert len(self.slam.map_points) == 0


class TestIntegration:
    """Integration tests for all modules."""

    def test_full_navigation_pipeline(self):
        """Test full pipeline: detection + depth + SLAM."""
        # Skip if modules not available
        try:
            from core.yolo_world_detector import YOLOWorldDetector
            from core.depth_anything_v2 import DepthAnythingV2
            from navigation.simple_slam import SimpleSLAM
        except ImportError:
            pytest.skip("Required modules not available")

        # Initialize modules
        config = {'models': {'depth': {'size': 'small', 'device': 'cpu'}}}
        detector = YOLOWorldDetector()
        depth_estimator = DepthAnythingV2(config)
        slam = SimpleSLAM()

        # Create test frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Process pipeline
        detections = detector.detect(frame)
        depth_map = depth_estimator.estimate_depth(frame)
        slam_result = slam.process_frame(frame, depth_map)

        # Verify outputs
        assert isinstance(detections, list)
        assert depth_map.shape == (480, 640)
        assert slam_result is not None
        assert 'position' in slam_result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
