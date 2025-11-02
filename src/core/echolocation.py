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

        # Adaptive audio beaconing frequencies
        self.safe_path_frequency = config.get('audio.beacon_safe_frequency', 440)  # 440Hz for safe path
        self.obstacle_frequency = config.get('audio.beacon_obstacle_frequency', 880)  # 880Hz for obstacles
        self.warning_pulse_rate = config.get('audio.warning_pulse_rate', 0.2)  # seconds for warning pulse

        # Create virtual room for spatial audio
        self._initialize_room()

        # Listener position (centered in room, at ear height)
        self.listener_pos = np.array([
            self.room_dims[0] / 2,
            self.room_dims[1] / 2,
            1.6  # Average ear height
        ])

        logging.info(f"Echolocation engine initialized: Room {self.room_dims}m")
        logging.info(f"Adaptive audio beaconing: Safe path {self.safe_path_frequency}Hz, Obstacle {self.obstacle_frequency}Hz")

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
            # Use simple stereo method instead of complex room simulation
            # This is more reliable and still provides spatial cues
            return self._create_mixed_stereo(detections, frame_shape)

        except Exception as e:
            logging.error(f"Spatial audio generation error: {e}")
            samples = int(self.sample_rate * self.beep_duration)
            return np.zeros((2, samples))

    def _create_mixed_stereo(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create mixed stereo audio from multiple detections (simpler, more reliable).

        Args:
            detections: List of detections
            frame_shape: (height, width)

        Returns:
            Stereo audio [2, samples]
        """
        # Start with silence
        samples = int(self.sample_rate * self.beep_duration)
        left_channel = np.zeros(samples)
        right_channel = np.zeros(samples)

        # Mix in beeps for each detection (up to 5)
        for det in detections[:5]:
            try:
                center = det.get('center', [frame_shape[1] / 2, frame_shape[0] / 2])
                depth = det.get('depth', 5.0)

                # Pan based on horizontal position (-1 = left, 1 = right)
                pan = (center[0] / frame_shape[1]) * 2 - 1

                # Generate beep
                frequency = self.distance_to_frequency(depth)
                volume = self.distance_to_volume(depth) * 0.3  # Reduce volume to avoid clipping

                beep = self.generate_beep(frequency) * volume

                # Apply panning
                left_gain = (1 - pan) / 2
                right_gain = (1 + pan) / 2

                # Mix into channels
                left_channel += beep * left_gain
                right_channel += beep * right_gain

            except Exception as e:
                logging.debug(f"Skipping detection due to error: {e}")
                continue

        # Normalize to prevent clipping
        max_val = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
        if max_val > 0:
            left_channel = left_channel / max_val
            right_channel = right_channel / max_val

        stereo = np.vstack([left_channel, right_channel])
        return stereo

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

    def generate_safe_path_beacon(self, frame_shape: Tuple[int, int], direction: str = 'center') -> np.ndarray:
        """
        Generate a safe path beacon (440Hz chime) in a specific direction.
        
        Args:
            frame_shape: (height, width) of frame
            direction: Direction for the beacon ('left', 'center', 'right', or 'custom')
            
        Returns:
            Stereo audio signal for safe path
        """
        # Generate chime at 440Hz
        chime = self.generate_beep(self.safe_path_frequency, duration=0.15)
        volume = 0.3  # Moderate volume for safe path indicator
        
        # Apply direction-based panning
        if direction == 'left':
            pan = -0.7  # Left channel emphasis
        elif direction == 'right':
            pan = 0.7   # Right channel emphasis
        elif direction == 'center':
            pan = 0.0   # Balanced
        else:
            pan = 0.0   # Default to center
        
        # Calculate gains based on pan
        left_gain = (1 - pan) / 2
        right_gain = (1 + pan) / 2
        
        # Apply volume and panning
        left_channel = chime * volume * left_gain
        right_channel = chime * volume * right_gain
        
        stereo = np.vstack([left_channel, right_channel])
        return stereo

    def generate_obstacle_warning(self, detection: Dict, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Generate an obstacle warning tone (880Hz) with direction-specific panning.
        
        Args:
            detection: Detection dictionary with 'center' and 'depth'
            frame_shape: (height, width) of frame
            
        Returns:
            Stereo audio signal for obstacle warning
        """
        center = detection.get('center', [frame_shape[1] / 2, frame_shape[0] / 2])
        
        # Calculate horizontal position for panning (-1 to 1)
        pan = (center[0] / frame_shape[1]) * 2 - 1
        
        # Generate warning tone at 880Hz
        warning = self.generate_beep(self.obstacle_frequency, duration=0.08)
        volume = self.distance_to_volume(detection.get('depth', 5.0)) * 0.5  # Adjust volume based on distance
        
        # Calculate gains based on pan
        left_gain = (1 - pan) / 2
        right_gain = (1 + pan) / 2
        
        # Apply volume and panning
        left_channel = warning * volume * left_gain
        right_channel = warning * volume * right_gain
        
        stereo = np.vstack([left_channel, right_channel])
        return stereo

    def generate_adaptive_beaconing_audio(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Generate adaptive beaconing audio: safe path chimes and obstacle warnings.
        
        Args:
            detections: List of detections with depth information
            frame_shape: (height, width) of frame
            
        Returns:
            Stereo audio signal for adaptive beaconing
        """
        # Start with silence - longer duration for more noticeable beacons
        samples = int(self.sample_rate * 0.3)  # Increased duration to 0.3 seconds for better audibility
        left_channel = np.zeros(samples)
        right_channel = np.zeros(samples)
        
        # Separate safe and danger zones
        danger_objects = [d for d in detections if d.get('is_danger', False)]
        
        if danger_objects:
            # Generate more prominent warning tones for obstacles (880Hz)
            for det in danger_objects[:3]:  # Limit to 3 danger objects to avoid audio clutter
                warning_audio = self.generate_obstacle_warning(det, frame_shape)
                
                # Boost volume for warnings
                warning_audio *= 1.5  # 50% volume boost for warnings
                
                # Mix into channels
                if warning_audio.shape[1] <= len(left_channel):
                    left_channel[:warning_audio.shape[1]] += warning_audio[0, :]
                    right_channel[:warning_audio.shape[1]] += warning_audio[1, :]
                else:
                    # If warning is longer, truncate it
                    left_channel += warning_audio[0, :len(left_channel)]
                    right_channel += warning_audio[1, :len(right_channel)]
        else:
            # If no immediate danger, generate safe path beacons
            # Create a sequence of safe path chimes to be more noticeable
            safe_chime = self.generate_safe_path_beacon(frame_shape, direction='center')
            
            # Add multiple safe path chimes with slight timing difference
            if safe_chime.shape[1] <= len(left_channel):
                # Add 2-3 chimes in sequence for more noticeable safe path indication
                chime_duration = safe_chime.shape[1]
                for i in range(min(2, len(left_channel) // (chime_duration * 2))):  # Up to 2 chimes
                    start_idx = i * chime_duration * 2
                    end_idx = start_idx + chime_duration
                    if end_idx <= len(left_channel):
                        left_channel[start_idx:end_idx] += safe_chime[0, :min(chime_duration, safe_chime.shape[1])] * 0.6
                        right_channel[start_idx:end_idx] += safe_chime[1, :min(chime_duration, safe_chime.shape[1])] * 0.6
            else:
                left_channel += safe_chime[0, :len(left_channel)] * 0.6
                right_channel += safe_chime[1, :len(right_channel)] * 0.6
        
        # Normalize to prevent clipping but maintain audibility
        max_val = max(np.max(np.abs(left_channel)), np.max(np.abs(right_channel)))
        if max_val > 0 and max_val < 0.1:  # Boost if too quiet
            left_channel = left_channel * (0.3 / max_val) if max_val > 0 else left_channel * 0.3
            right_channel = right_channel * (0.3 / max_val) if max_val > 0 else right_channel * 0.3
        elif max_val > 1.0:
            left_channel = left_channel / max_val
            right_channel = right_channel / max_val
        elif max_val > 0:
            # Boost slightly if in a reasonable range
            left_channel = left_channel * 1.5  # Boost by 50% to make more noticeable
            right_channel = right_channel * 1.5
        
        stereo = np.vstack([left_channel, right_channel])
        return stereo


class AudioCueGenerator:
    """
    High-level audio cue generator combining echolocation with voice cues.
    """

    def __init__(self, config):
        """Initialize audio cue generator."""
        self.config = config
        self.echolocation = EcholocationEngine(config)
        self.enabled = config.get('audio.echolocation_enabled', True)
        self.adaptive_beaconing_enabled = config.get('audio.adaptive_beaconing_enabled', True)

    def generate_cues(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> Tuple[np.ndarray, str]:
        """
        Generate breakthrough spatial audio cues for blind navigation.
        Creates 3D spatial audio that maps to real-world positions.
        """
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

        # Generate audio based on adaptive beaconing setting
        if self.adaptive_beaconing_enabled:
            # Use adaptive beaconing: 440Hz for safe path, 880Hz for obstacles
            audio = self.echolocation.generate_adaptive_beaconing_audio(detections, frame_shape)
        else:
            # Use traditional spatial audio
            audio = self.echolocation.generate_navigation_audio(detections, frame_shape)

        # Generate voice message
        danger_objects = [d for d in detections if d.get('is_danger', False)]
        caution_objects = [d for d in detections if d.get('depth') is not None and d.get('depth') < 3.0 and not d.get('is_danger', False)]

        if len(danger_objects) > 0:
            closest = min(danger_objects, key=lambda x: x.get('depth', 10))
            message = f"Warning! {closest['label']} {closest['depth']:.1f} meters ahead"
        elif len(caution_objects) > 0:
            closest = min(caution_objects, key=lambda x: x.get('depth', 10))
            message = f"Caution. {closest['label']} at {closest['depth']:.1f} meters"
        else:
            message = f"{len(detections)} objects detected. Path clear"

        return audio, message
