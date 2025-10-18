"""
OrbyGlasses - Echolocation Simulation
Generates spatial audio cues based on object positions and depths.
"""

import numpy as np
import pyroomacoustics as pra
from scipy import signal
import logging
from typing import List, Dict, Tuple, Optional


class EcholocationEngine:
    """
    Simulates bio-mimetic echolocation using spatial audio.
    Creates binaural audio cues based on object positions and distances.
    """

    def __init__(self, config):
        """
        Initialize echolocation engine.

        Args:
            config: ConfigManager instance
        """
        self.config = config

        # Room dimensions [x, y, z] in meters
        self.room_dims = config.get('echolocation.room_dimensions', [10, 10, 3])
        self.sample_rate = config.get('echolocation.sample_rate', 16000)
        self.beep_duration = config.get('echolocation.duration', 0.1)

        # Create virtual room for spatial audio
        self._initialize_room()

        # Listener position (centered in room, at ear height)
        self.listener_pos = np.array([
            self.room_dims[0] / 2,
            self.room_dims[1] / 2,
            1.6  # Average ear height
        ])

        logging.info(f"Echolocation engine initialized: Room {self.room_dims}m")

    def _initialize_room(self):
        """Create virtual room for acoustic simulation."""
        try:
            # Create room with absorption
            self.room = pra.ShoeBox(
                self.room_dims,
                fs=self.sample_rate,
                materials=pra.Material(0.2),  # Moderate absorption
                max_order=3  # Reflection order
            )
        except Exception as e:
            logging.error(f"Failed to create room: {e}")
            self.room = None

    def generate_beep(self, frequency: float, duration: float = None) -> np.ndarray:
        """
        Generate a beep sound.

        Args:
            frequency: Frequency in Hz
            duration: Duration in seconds

        Returns:
            Audio signal
        """
        if duration is None:
            duration = self.beep_duration

        t = np.linspace(0, duration, int(self.sample_rate * duration))

        # Generate tone with envelope
        beep = np.sin(2 * np.pi * frequency * t)

        # Apply envelope (fade in/out)
        envelope = signal.windows.hann(len(beep))
        beep = beep * envelope

        return beep

    def distance_to_frequency(self, distance: float) -> float:
        """
        Map distance to frequency (closer = higher frequency).

        Args:
            distance: Distance in meters

        Returns:
            Frequency in Hz
        """
        # Map 0-10m to 1000-300 Hz (closer is higher pitch)
        min_freq = 300
        max_freq = 1000
        max_dist = 10.0

        # Clamp distance
        distance = np.clip(distance, 0, max_dist)

        # Inverse mapping
        freq = max_freq - (distance / max_dist) * (max_freq - min_freq)

        return freq

    def distance_to_volume(self, distance: float) -> float:
        """
        Map distance to volume (closer = louder).

        Args:
            distance: Distance in meters

        Returns:
            Volume multiplier (0-1)
        """
        max_dist = 10.0
        distance = np.clip(distance, 0.1, max_dist)

        # Inverse square law approximation
        volume = 1.0 / (distance ** 0.5)
        volume = np.clip(volume, 0.1, 1.0)

        return volume

    def position_from_detection(self, detection: Dict, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Convert detection to 3D position in room.

        Args:
            detection: Detection dict with 'center' and 'depth'
            frame_shape: (height, width) of frame

        Returns:
            3D position [x, y, z] in room coordinates
        """
        center = detection.get('center', [frame_shape[1] / 2, frame_shape[0] / 2])
        depth = detection.get('depth', 5.0)

        # Normalize center to -1 to 1
        center_x_norm = (center[0] / frame_shape[1]) * 2 - 1  # -1 (left) to 1 (right)
        center_y_norm = (center[1] / frame_shape[0]) * 2 - 1  # -1 (top) to 1 (bottom)

        # Map to room coordinates
        # x: left-right relative to listener
        # y: forward distance (depth)
        # z: up-down relative to listener

        x = self.listener_pos[0] + center_x_norm * 2  # ±2m from center
        y = self.listener_pos[1] + depth  # Forward distance
        z = self.listener_pos[2] + center_y_norm * 0.5  # ±0.5m vertical

        # Clamp to room bounds
        x = np.clip(x, 0.1, self.room_dims[0] - 0.1)
        y = np.clip(y, 0.1, self.room_dims[1] - 0.1)
        z = np.clip(z, 0.1, self.room_dims[2] - 0.1)

        return np.array([x, y, z])

    def create_spatial_audio(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create binaural spatial audio from detections.

        Args:
            detections: List of detections with depth
            frame_shape: (height, width) of frame

        Returns:
            Stereo audio signal [2, samples]
        """
        if self.room is None or len(detections) == 0:
            # Return silence
            samples = int(self.sample_rate * self.beep_duration)
            return np.zeros((2, samples))

        try:
            # Clear previous sources
            self.room.sources = []

            # Add listener (microphone array for binaural)
            if not hasattr(self.room, 'mic_array') or self.room.mic_array is None:
                # Create binaural microphone (two mics separated by head width)
                mic_locs = np.array([
                    [self.listener_pos[0] - 0.1, self.listener_pos[0] + 0.1],  # Left/right
                    [self.listener_pos[1], self.listener_pos[1]],
                    [self.listener_pos[2], self.listener_pos[2]]
                ])
                self.room.add_microphone_array(mic_locs)

            # Add sound sources for each detection
            for det in detections[:5]:  # Limit to 5 objects to avoid clutter
                depth = det.get('depth', 5.0)
                position = self.position_from_detection(det, frame_shape)

                # Generate beep based on distance
                frequency = self.distance_to_frequency(depth)
                volume = self.distance_to_volume(depth)
                beep = self.generate_beep(frequency) * volume

                # Add source to room
                self.room.add_source(position, signal=beep)

            # Simulate acoustics
            self.room.simulate()

            # Get binaural signal
            audio = self.room.mic_array.signals

            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-8)

            return audio

        except Exception as e:
            logging.error(f"Spatial audio generation error: {e}")
            samples = int(self.sample_rate * self.beep_duration)
            return np.zeros((2, samples))

    def create_simple_stereo_beep(self, detection: Dict, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create simple stereo beep for a single detection (fallback method).

        Args:
            detection: Detection dict
            frame_shape: (height, width)

        Returns:
            Stereo audio [2, samples]
        """
        center = detection.get('center', [frame_shape[1] / 2, frame_shape[0] / 2])
        depth = detection.get('depth', 5.0)

        # Pan based on horizontal position (-1 = left, 1 = right)
        pan = (center[0] / frame_shape[1]) * 2 - 1

        # Generate beep
        frequency = self.distance_to_frequency(depth)
        volume = self.distance_to_volume(depth)
        beep = self.generate_beep(frequency) * volume

        # Apply panning
        left_gain = (1 - pan) / 2
        right_gain = (1 + pan) / 2

        left_channel = beep * left_gain
        right_channel = beep * right_gain

        stereo = np.vstack([left_channel, right_channel])

        return stereo

    def create_danger_alert(self, danger_objects: List[Dict], frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create urgent alert for danger objects.

        Args:
            danger_objects: List of detections marked as danger
            frame_shape: (height, width)

        Returns:
            Stereo alert audio
        """
        if len(danger_objects) == 0:
            samples = int(self.sample_rate * 0.2)
            return np.zeros((2, samples))

        # Create rapid beeps for alert
        beeps = []
        for _ in range(3):  # Triple beep
            beep = self.generate_beep(1200, duration=0.05)  # High pitch
            silence = np.zeros(int(self.sample_rate * 0.05))
            beeps.extend([beep, silence])

        alert = np.concatenate(beeps)

        # Make stereo
        stereo = np.vstack([alert, alert])

        return stereo

    def generate_navigation_audio(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Main method to generate navigation audio cues.

        Args:
            detections: List of detections
            frame_shape: (height, width)

        Returns:
            Stereo audio signal
        """
        if len(detections) == 0:
            return np.zeros((2, int(self.sample_rate * 0.1)))

        # Check for danger objects
        danger_objects = [d for d in detections if d.get('is_danger', False)]

        if len(danger_objects) > 0:
            # Prioritize danger alert
            return self.create_danger_alert(danger_objects, frame_shape)
        else:
            # Normal spatial audio
            try:
                return self.create_spatial_audio(detections, frame_shape)
            except Exception as e:
                logging.warning(f"Spatial audio failed, using simple beep: {e}")
                # Fallback to simple beep for closest object
                if len(detections) > 0:
                    return self.create_simple_stereo_beep(detections[0], frame_shape)
                return np.zeros((2, int(self.sample_rate * 0.1)))


class AudioCueGenerator:
    """
    High-level audio cue generator combining echolocation with voice cues.
    """

    def __init__(self, config):
        """Initialize audio cue generator."""
        self.config = config
        self.echolocation = EcholocationEngine(config)
        self.enabled = config.get('audio.echolocation_enabled', True)

    def generate_cues(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> Tuple[np.ndarray, str]:
        """
        Generate audio cues and voice message.

        Args:
            detections: List of detections
            frame_shape: Frame dimensions

        Returns:
            Tuple of (audio_signal, voice_message)
        """
        if not self.enabled or len(detections) == 0:
            return np.zeros((2, 1000)), "Path clear"

        # Generate spatial audio
        audio = self.echolocation.generate_navigation_audio(detections, frame_shape)

        # Generate voice message
        danger_objects = [d for d in detections if d.get('is_danger', False)]
        caution_objects = [d for d in detections if d.get('depth', 10) < 3.0 and not d.get('is_danger', False)]

        if len(danger_objects) > 0:
            closest = min(danger_objects, key=lambda x: x.get('depth', 10))
            message = f"Warning! {closest['label']} {closest['depth']:.1f} meters ahead"
        elif len(caution_objects) > 0:
            closest = min(caution_objects, key=lambda x: x.get('depth', 10))
            message = f"Caution. {closest['label']} at {closest['depth']:.1f} meters"
        else:
            message = f"{len(detections)} objects detected. Path clear"

        return audio, message
