"""
Tests for OrbyGlasses 2025 Dark-Themed Depth Visualizer
"""

import pytest
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from visualization.depth_visualizer_2025 import (
    DarkThemeDepthVisualizer,
    HapticAudioConverter
)


class TestDarkThemeDepthVisualizer:
    """Tests for DarkThemeDepthVisualizer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = DarkThemeDepthVisualizer()
        self.depth_map = np.random.rand(480, 640) * 10.0  # Random depth 0-10m

    def test_initialization(self):
        """Test visualizer initialization."""
        assert self.visualizer is not None
        assert self.visualizer.danger_threshold == 0.5
        assert self.visualizer.caution_threshold == 1.5
        assert self.visualizer.safe_threshold == 3.5

    def test_create_obsidian_colormap(self):
        """Test obsidian colormap creation."""
        depth_norm = self.depth_map / 10.0
        colored = self.visualizer.create_obsidian_colormap(depth_norm)

        assert colored.shape == (480, 640, 3)
        assert colored.dtype == np.uint8
        assert np.min(colored) >= 0
        assert np.max(colored) <= 255

    def test_enhance_edges(self):
        """Test edge enhancement."""
        depth_norm = self.depth_map / 10.0
        colored = self.visualizer.create_obsidian_colormap(depth_norm)
        enhanced = self.visualizer.enhance_edges(colored)

        assert enhanced.shape == colored.shape
        assert enhanced.dtype == np.uint8

    def test_visualize(self):
        """Test full visualization pipeline."""
        result = self.visualizer.visualize(self.depth_map, normalize=True)

        assert result is not None
        assert result.dtype == np.uint8
        # Check that image height increased due to legend
        assert result.shape[0] > self.depth_map.shape[0]

    def test_visualize_without_legend(self):
        """Test visualization without legend."""
        self.visualizer.show_legend = False
        result = self.visualizer.visualize(self.depth_map, normalize=True)

        assert result.shape[0] == self.depth_map.shape[0]


class TestHapticAudioConverter:
    """Tests for HapticAudioConverter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = HapticAudioConverter()
        self.depth_map = np.random.rand(480, 640)  # Normalized 0-1

    def test_initialization(self):
        """Test converter initialization."""
        assert self.converter is not None
        assert self.converter.num_motors == 10
        assert self.converter.min_frequency == 50
        assert self.converter.max_frequency == 300

    def test_depth_to_haptic_pattern(self):
        """Test depth to haptic pattern conversion."""
        pattern = self.converter.depth_to_haptic_pattern(self.depth_map)

        assert pattern.shape == (10, 2)  # 10 motors, [frequency, intensity]
        assert np.all(pattern[:, 0] >= self.converter.min_frequency)
        assert np.all(pattern[:, 0] <= self.converter.max_frequency)
        assert np.all(pattern[:, 1] >= 0)
        assert np.all(pattern[:, 1] <= 255)

    def test_depth_to_audio_sonification(self):
        """Test depth to audio sonification conversion."""
        audio = self.converter.depth_to_audio_sonification(self.depth_map)

        expected_length = int(self.converter.tone_duration * self.converter.sample_rate)
        assert audio.shape == (expected_length, 2)  # Stereo
        assert np.max(np.abs(audio)) <= 1.0  # Normalized

    def test_generate_danger_alert_with_danger(self):
        """Test danger alert generation when danger present."""
        # Create depth map with danger zone
        danger_depth = np.ones((480, 640)) * 0.5
        danger_depth[200:300, 300:400] = 0.1  # Danger zone

        alert = self.converter.generate_danger_alert(danger_depth, threshold=0.2)

        assert alert is not None
        assert 'danger_level' in alert
        assert 'haptic_pattern' in alert
        assert 'audio_alert' in alert
        assert 'message' in alert
        assert alert['danger_level'] > 0

    def test_generate_danger_alert_without_danger(self):
        """Test danger alert generation when no danger present."""
        safe_depth = np.ones((480, 640)) * 0.5  # All safe

        alert = self.converter.generate_danger_alert(safe_depth, threshold=0.2)

        assert alert is None


class TestIntegration:
    """Integration tests for depth visualizer + haptic/audio converter."""

    def test_full_pipeline(self):
        """Test complete pipeline from depth map to visualization + haptic/audio."""
        # Create sample depth map
        depth_map = np.random.rand(480, 640) * 10.0

        # Visualize
        visualizer = DarkThemeDepthVisualizer()
        visual = visualizer.visualize(depth_map, normalize=True)

        # Convert to haptic/audio
        converter = HapticAudioConverter()
        depth_norm = depth_map / 10.0
        haptic_pattern = converter.depth_to_haptic_pattern(depth_norm)
        audio_signal = converter.depth_to_audio_sonification(depth_norm)

        # Verify all outputs
        assert visual is not None
        assert haptic_pattern is not None
        assert audio_signal is not None
        assert haptic_pattern.shape[0] == 10
        assert audio_signal.shape[1] == 2  # Stereo


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
