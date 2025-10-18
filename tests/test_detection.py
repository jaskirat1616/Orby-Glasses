"""
Unit tests for detection module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import cv2
from detection import ObjectDetector, DepthEstimator, DetectionPipeline
from utils import ConfigManager


@pytest.fixture
def config():
    """Create test configuration."""
    config_dict = {
        'models': {
            'yolo': {
                'path': 'models/yolo/yolo11n.pt',
                'confidence': 0.5,
                'iou_threshold': 0.45,
                'device': 'cpu'
            },
            'depth': {
                'path': 'models/depth/depth_pro.pt',
                'device': 'cpu'
            }
        },
        'safety': {
            'min_safe_distance': 1.5
        }
    }
    config = ConfigManager.__new__(ConfigManager)
    config.config = config_dict
    return config


@pytest.fixture
def sample_frame():
    """Create a sample test frame."""
    # Create a simple test image (480x640x3)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw some shapes to simulate objects
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue square
    cv2.circle(frame, (400, 300), 50, (0, 255, 0), -1)  # Green circle

    return frame


class TestObjectDetector:
    """Tests for ObjectDetector class."""

    def test_initialization(self, config):
        """Test detector initialization."""
        detector = ObjectDetector(
            confidence=0.5,
            device='cpu'
        )
        assert detector is not None
        assert detector.confidence == 0.5
        assert detector.device == 'cpu'

    def test_device_validation(self, config):
        """Test device validation."""
        detector = ObjectDetector(device='invalid_device')
        assert detector.device == 'cpu'  # Should fallback to CPU

    def test_detect_returns_list(self, config, sample_frame):
        """Test that detect returns a list."""
        detector = ObjectDetector(device='cpu')
        detections = detector.detect(sample_frame)
        assert isinstance(detections, list)

    def test_detection_structure(self, config, sample_frame):
        """Test detection dictionary structure."""
        detector = ObjectDetector(device='cpu')
        detections = detector.detect(sample_frame)

        if len(detections) > 0:
            det = detections[0]
            assert 'bbox' in det
            assert 'label' in det
            assert 'confidence' in det
            assert 'class_id' in det
            assert 'center' in det
            assert len(det['bbox']) == 4
            assert len(det['center']) == 2


class TestDepthEstimator:
    """Tests for DepthEstimator class."""

    def test_initialization(self, config):
        """Test depth estimator initialization."""
        estimator = DepthEstimator(device='cpu')
        assert estimator is not None
        assert estimator.device == 'cpu'

    def test_estimate_depth_returns_array(self, config, sample_frame):
        """Test that depth estimation returns numpy array."""
        estimator = DepthEstimator(device='cpu')
        depth_map = estimator.estimate_depth(sample_frame)

        assert depth_map is not None
        assert isinstance(depth_map, np.ndarray)
        assert depth_map.shape[:2] == sample_frame.shape[:2]

    def test_depth_map_normalized(self, config, sample_frame):
        """Test that depth map is normalized to 0-1."""
        estimator = DepthEstimator(device='cpu')
        depth_map = estimator.estimate_depth(sample_frame)

        if depth_map is not None:
            assert depth_map.min() >= 0.0
            assert depth_map.max() <= 1.0

    def test_get_depth_at_bbox(self, config, sample_frame):
        """Test depth extraction at bounding box."""
        estimator = DepthEstimator(device='cpu')
        depth_map = estimator.estimate_depth(sample_frame)

        bbox = [100, 100, 200, 200]
        depth = estimator.get_depth_at_bbox(depth_map, bbox)

        assert isinstance(depth, float)
        assert depth >= 0.0


class TestDetectionPipeline:
    """Tests for DetectionPipeline class."""

    def test_initialization(self, config):
        """Test pipeline initialization."""
        pipeline = DetectionPipeline(config)
        assert pipeline is not None
        assert pipeline.detector is not None
        assert pipeline.depth_estimator is not None

    def test_process_frame(self, config, sample_frame):
        """Test frame processing."""
        pipeline = DetectionPipeline(config)
        detections, depth_map = pipeline.process_frame(sample_frame)

        assert isinstance(detections, list)
        assert depth_map is None or isinstance(depth_map, np.ndarray)

    def test_detections_have_depth(self, config, sample_frame):
        """Test that detections include depth information."""
        pipeline = DetectionPipeline(config)
        detections, _ = pipeline.process_frame(sample_frame)

        for det in detections:
            assert 'depth' in det
            assert 'is_danger' in det

    def test_navigation_summary(self, config, sample_frame):
        """Test navigation summary generation."""
        pipeline = DetectionPipeline(config)
        detections, _ = pipeline.process_frame(sample_frame)

        summary = pipeline.get_navigation_summary(detections)

        assert 'total_objects' in summary
        assert 'danger_objects' in summary
        assert 'caution_objects' in summary
        assert 'safe_objects' in summary
        assert 'path_clear' in summary
        assert isinstance(summary['total_objects'], int)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
