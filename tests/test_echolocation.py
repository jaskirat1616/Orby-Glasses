"""
Unit tests for echolocation module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import numpy as np
from echolocation import EcholocationEngine, AudioCueGenerator
from utils import ConfigManager


@pytest.fixture
def config():
    """Create test configuration."""
    config_dict = {
        'echolocation': {
            'room_dimensions': [10, 10, 3],
            'sample_rate': 16000,
            'duration': 0.1
        },
        'audio': {
            'echolocation_enabled': True
        }
    }
    config = ConfigManager.__new__(ConfigManager)
    config.config = config_dict
    return config


@pytest.fixture
def sample_detections():
    """Create sample detections."""
    return [
        {
            'label': 'person',
            'bbox': [100, 100, 200, 200],
            'center': [150, 150],
            'depth': 2.5,
            'confidence': 0.9,
            'is_danger': False
        },
        {
            'label': 'car',
            'bbox': [300, 200, 450, 350],
            'center': [375, 275],
            'depth': 1.2,
            'confidence': 0.85,
            'is_danger': True
        }
    ]


class TestEcholocationEngine:
    """Tests for EcholocationEngine class."""

    def test_initialization(self, config):
        """Test echolocation engine initialization."""
        engine = EcholocationEngine(config)
        assert engine is not None
        assert engine.sample_rate == 16000
        assert len(engine.room_dims) == 3

    def test_generate_beep(self, config):
        """Test beep generation."""
        engine = EcholocationEngine(config)
        beep = engine.generate_beep(frequency=500, duration=0.1)

        assert isinstance(beep, np.ndarray)
        assert len(beep) == int(engine.sample_rate * 0.1)

    def test_distance_to_frequency(self, config):
        """Test distance to frequency mapping."""
        engine = EcholocationEngine(config)

        # Closer objects should have higher frequency
        freq_close = engine.distance_to_frequency(1.0)
        freq_far = engine.distance_to_frequency(8.0)

        assert freq_close > freq_far
        assert 300 <= freq_far <= 1000
        assert 300 <= freq_close <= 1000

    def test_distance_to_volume(self, config):
        """Test distance to volume mapping."""
        engine = EcholocationEngine(config)

        # Closer objects should be louder
        vol_close = engine.distance_to_volume(1.0)
        vol_far = engine.distance_to_volume(8.0)

        assert vol_close > vol_far
        assert 0.0 <= vol_far <= 1.0
        assert 0.0 <= vol_close <= 1.0

    def test_position_from_detection(self, config, sample_detections):
        """Test 3D position calculation from detection."""
        engine = EcholocationEngine(config)
        frame_shape = (480, 640)

        position = engine.position_from_detection(sample_detections[0], frame_shape)

        assert isinstance(position, np.ndarray)
        assert len(position) == 3
        # Check bounds
        assert 0 <= position[0] <= engine.room_dims[0]
        assert 0 <= position[1] <= engine.room_dims[1]
        assert 0 <= position[2] <= engine.room_dims[2]

    def test_create_simple_stereo_beep(self, config, sample_detections):
        """Test simple stereo beep creation."""
        engine = EcholocationEngine(config)
        frame_shape = (480, 640)

        stereo = engine.create_simple_stereo_beep(sample_detections[0], frame_shape)

        assert isinstance(stereo, np.ndarray)
        assert stereo.shape[0] == 2  # Stereo (left and right)
        assert stereo.shape[1] > 0

    def test_create_danger_alert(self, config, sample_detections):
        """Test danger alert creation."""
        engine = EcholocationEngine(config)
        frame_shape = (480, 640)

        danger_objects = [d for d in sample_detections if d.get('is_danger', False)]
        alert = engine.create_danger_alert(danger_objects, frame_shape)

        assert isinstance(alert, np.ndarray)
        assert alert.shape[0] == 2  # Stereo

    def test_generate_navigation_audio(self, config, sample_detections):
        """Test navigation audio generation."""
        engine = EcholocationEngine(config)
        frame_shape = (480, 640)

        audio = engine.generate_navigation_audio(sample_detections, frame_shape)

        assert isinstance(audio, np.ndarray)
        assert audio.shape[0] == 2  # Stereo
        assert audio.shape[1] > 0


class TestAudioCueGenerator:
    """Tests for AudioCueGenerator class."""

    def test_initialization(self, config):
        """Test audio cue generator initialization."""
        generator = AudioCueGenerator(config)
        assert generator is not None
        assert generator.echolocation is not None

    def test_generate_cues(self, config, sample_detections):
        """Test cue generation."""
        generator = AudioCueGenerator(config)
        frame_shape = (480, 640)

        audio, message = generator.generate_cues(sample_detections, frame_shape)

        assert isinstance(audio, np.ndarray)
        assert isinstance(message, str)
        assert len(message) > 0

    def test_danger_message(self, config, sample_detections):
        """Test that danger objects generate appropriate messages."""
        generator = AudioCueGenerator(config)
        frame_shape = (480, 640)

        danger_dets = [d for d in sample_detections if d.get('is_danger', False)]
        audio, message = generator.generate_cues(danger_dets, frame_shape)

        assert 'warning' in message.lower() or 'caution' in message.lower()

    def test_empty_detections(self, config):
        """Test with no detections."""
        generator = AudioCueGenerator(config)
        frame_shape = (480, 640)

        audio, message = generator.generate_cues([], frame_shape)

        assert isinstance(audio, np.ndarray)
        assert isinstance(message, str)
        assert 'clear' in message.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
