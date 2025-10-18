"""
Unit tests for utility functions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
import cv2
from utils import (
    ConfigManager, FrameProcessor, DataLogger,
    PerformanceMonitor, check_device
)


class TestConfigManager:
    """Tests for ConfigManager class."""

    def test_default_config(self):
        """Test that default config is returned when file not found."""
        config = ConfigManager('nonexistent.yaml')
        assert config.config is not None
        assert 'camera' in config.config

    def test_get_with_dot_notation(self):
        """Test config retrieval with dot notation."""
        config = ConfigManager.__new__(ConfigManager)
        config.config = {
            'camera': {
                'width': 640,
                'height': 480
            }
        }

        assert config.get('camera.width') == 640
        assert config.get('camera.height') == 480
        assert config.get('camera.nonexistent', 1024) == 1024


class TestFrameProcessor:
    """Tests for FrameProcessor utilities."""

    @pytest.fixture
    def sample_frame(self):
        """Create sample frame."""
        return np.zeros((480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def sample_detections(self):
        """Create sample detections."""
        return [
            {
                'label': 'person',
                'bbox': [100, 100, 200, 200],
                'confidence': 0.9,
                'depth': 2.5
            }
        ]

    def test_resize_frame(self, sample_frame):
        """Test frame resizing."""
        resized = FrameProcessor.resize_frame(sample_frame, 320, 240)
        assert resized.shape == (240, 320, 3)

    def test_annotate_detections(self, sample_frame, sample_detections):
        """Test detection annotation."""
        annotated = FrameProcessor.annotate_detections(sample_frame, sample_detections)

        assert annotated.shape == sample_frame.shape
        # Check that frame was modified (annotation added)
        assert not np.array_equal(annotated, sample_frame)

    def test_encode_frame_base64(self, sample_frame):
        """Test frame encoding to base64."""
        encoded = FrameProcessor.encode_frame_base64(sample_frame)

        assert isinstance(encoded, str)
        assert len(encoded) > 0


class TestDataLogger:
    """Tests for DataLogger class."""

    def test_initialization(self, tmp_path):
        """Test logger initialization."""
        logger = DataLogger(log_dir=str(tmp_path))
        assert logger is not None
        assert os.path.exists(logger.session_file)

    def test_log_detection(self, tmp_path):
        """Test detection logging."""
        logger = DataLogger(log_dir=str(tmp_path))

        detections = [
            {'label': 'person', 'depth': 2.5, 'confidence': 0.9}
        ]

        logger.log_detection(1, detections)

        # Check file exists and has content
        assert os.path.exists(logger.session_file)
        with open(logger.session_file, 'r') as f:
            content = f.read()
            assert len(content) > 0
            assert 'person' in content

    def test_load_session_data(self, tmp_path):
        """Test loading session data."""
        logger = DataLogger(log_dir=str(tmp_path))

        detections = [
            {'label': 'car', 'depth': 1.5, 'confidence': 0.85}
        ]

        logger.log_detection(1, detections)
        data = logger.load_session_data()

        assert len(data) == 1
        assert data[0]['frame_id'] == 1


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = PerformanceMonitor()
        assert monitor is not None

    def test_timer(self):
        """Test timer functionality."""
        import time
        monitor = PerformanceMonitor()

        monitor.start_timer('test')
        time.sleep(0.01)
        elapsed = monitor.stop_timer('test')

        assert elapsed >= 10.0  # At least 10ms

    def test_log_frame_time(self):
        """Test frame time logging."""
        monitor = PerformanceMonitor()

        monitor.log_frame_time(16.7)
        monitor.log_frame_time(20.0)

        assert len(monitor.frame_times) == 2

    def test_get_avg_fps(self):
        """Test average FPS calculation."""
        monitor = PerformanceMonitor()

        # Simulate 60 FPS (16.67ms per frame)
        for _ in range(10):
            monitor.log_frame_time(16.67)

        fps = monitor.get_avg_fps()
        assert 55 < fps < 65  # Around 60 FPS

    def test_get_stats(self):
        """Test statistics retrieval."""
        monitor = PerformanceMonitor()

        monitor.log_frame_time(10.0)
        monitor.log_frame_time(20.0)
        monitor.log_frame_time(15.0)

        stats = monitor.get_stats()

        assert 'avg_frame_time_ms' in stats
        assert 'min_frame_time_ms' in stats
        assert 'max_frame_time_ms' in stats
        assert 'fps' in stats

        assert stats['avg_frame_time_ms'] == 15.0
        assert stats['min_frame_time_ms'] == 10.0
        assert stats['max_frame_time_ms'] == 20.0


class TestDeviceCheck:
    """Tests for device checking."""

    def test_check_device(self):
        """Test device detection."""
        device = check_device()
        assert device in ['mps', 'cuda', 'cpu']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
