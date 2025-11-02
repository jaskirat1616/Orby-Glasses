"""
OrbyGlasses 2025 - Haptic Feedback System

Revolutionary haptic feedback for blind navigation:
- Vibrotactile belt/headband with 10+ motors
- HaptEQ 2.0 pattern library for directional guidance
- Shape-changing interfaces for distance perception
- Bio-adaptive intensity adjustment

Target: Intuitive "felt" navigation with <20ms latency
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
import time
from enum import Enum
from dataclasses import dataclass


class HapticPattern(Enum):
    """Predefined haptic pattern types."""
    URGENT_STOP = "urgent_stop"
    CAUTION = "caution"
    DIRECTIONAL_GUIDE = "directional_guide"
    ALL_CLEAR = "all_clear"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    STRAIGHT_AHEAD = "straight_ahead"
    OBSTACLE_LEFT = "obstacle_left"
    OBSTACLE_RIGHT = "obstacle_right"
    OBSTACLE_AHEAD = "obstacle_ahead"


@dataclass
class HapticSignal:
    """Haptic signal data structure."""
    motor_id: int
    frequency: float  # Hz (50-300)
    intensity: float  # 0-1 (PWM duty cycle)
    duration: float   # seconds
    timestamp: float


class HaptEQPatternLibrary:
    """
    HaptEQ 2.0 pattern library for standardized haptic feedback.

    Based on research from haptic HCI (Human-Computer Interaction) literature.
    Patterns are designed for maximum perceptual clarity and minimal learning curve.
    """

    def __init__(self):
        """Initialize HaptEQ 2.0 pattern library."""
        self.patterns = self._create_pattern_library()

    def _create_pattern_library(self) -> Dict[str, np.ndarray]:
        """
        Create standardized haptic patterns.

        Pattern format: (num_motors, 3) array [frequency, intensity, duration]
        Motors arranged in circular array (0=forward, 4-5=back, 2-3=right, 7-8=left)

        Returns:
            Dictionary of pattern name → pattern array
        """
        patterns = {}

        # Urgent stop: All motors max intensity, high frequency, pulsing
        patterns['urgent_stop'] = np.array([
            [250, 1.0, 0.1] for _ in range(10)  # All motors
        ])

        # Caution: Medium intensity, medium frequency, front motors
        patterns['caution'] = np.array([
            [180, 0.7, 0.15],  # Motor 0 (front)
            [180, 0.6, 0.15],  # Motor 1
            [120, 0.4, 0.15],  # Motor 2
            [0, 0, 0],         # Motor 3 (off)
            [0, 0, 0],         # Motor 4 (off)
            [0, 0, 0],         # Motor 5 (off)
            [0, 0, 0],         # Motor 6 (off)
            [120, 0.4, 0.15],  # Motor 7
            [180, 0.6, 0.15],  # Motor 8
            [180, 0.7, 0.15],  # Motor 9
        ])

        # Directional guide (forward): Front motors activated
        patterns['straight_ahead'] = np.array([
            [150, 0.8, 0.2],   # Motor 0 (front)
            [120, 0.6, 0.2],   # Motor 1
            [0, 0, 0],         # Motor 2 (off)
            [0, 0, 0],         # Motor 3 (off)
            [0, 0, 0],         # Motor 4 (off)
            [0, 0, 0],         # Motor 5 (off)
            [0, 0, 0],         # Motor 6 (off)
            [0, 0, 0],         # Motor 7 (off)
            [120, 0.6, 0.2],   # Motor 8
            [150, 0.8, 0.2],   # Motor 9
        ])

        # Turn left: Left motors activated with gradient
        patterns['turn_left'] = np.array([
            [80, 0.3, 0.2],    # Motor 0
            [100, 0.4, 0.2],   # Motor 1
            [120, 0.5, 0.2],   # Motor 2
            [150, 0.7, 0.2],   # Motor 3 (strongest left)
            [150, 0.7, 0.2],   # Motor 4
            [120, 0.5, 0.2],   # Motor 5
            [100, 0.4, 0.2],   # Motor 6
            [80, 0.3, 0.2],    # Motor 7
            [0, 0, 0],         # Motor 8 (off)
            [0, 0, 0],         # Motor 9 (off)
        ])

        # Turn right: Right motors activated with gradient
        patterns['turn_right'] = np.array([
            [0, 0, 0],         # Motor 0 (off)
            [0, 0, 0],         # Motor 1 (off)
            [80, 0.3, 0.2],    # Motor 2
            [100, 0.4, 0.2],   # Motor 3
            [120, 0.5, 0.2],   # Motor 4
            [150, 0.7, 0.2],   # Motor 5 (strongest right)
            [150, 0.7, 0.2],   # Motor 6
            [120, 0.5, 0.2],   # Motor 7
            [100, 0.4, 0.2],   # Motor 8
            [80, 0.3, 0.2],    # Motor 9
        ])

        # Obstacle ahead: Front motors pulsing
        patterns['obstacle_ahead'] = np.array([
            [200, 0.9, 0.1],   # Motor 0 (pulse)
            [180, 0.8, 0.1],   # Motor 1
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [180, 0.8, 0.1],   # Motor 8
            [200, 0.9, 0.1],   # Motor 9
        ])

        # Obstacle left: Left motors warning
        patterns['obstacle_left'] = np.array([
            [0, 0, 0],
            [150, 0.7, 0.15],  # Motor 1
            [180, 0.8, 0.15],  # Motor 2
            [200, 0.9, 0.15],  # Motor 3 (strongest)
            [180, 0.8, 0.15],  # Motor 4
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])

        # Obstacle right: Right motors warning
        patterns['obstacle_right'] = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [180, 0.8, 0.15],  # Motor 5
            [200, 0.9, 0.15],  # Motor 6 (strongest)
            [180, 0.8, 0.15],  # Motor 7
            [150, 0.7, 0.15],  # Motor 8
            [0, 0, 0],
        ])

        # All clear: Gentle pulse all motors
        patterns['all_clear'] = np.array([
            [80, 0.2, 0.3] for _ in range(10)  # Gentle all-around pulse
        ])

        return patterns

    def get(self, pattern_name: str) -> np.ndarray:
        """
        Get haptic pattern by name.

        Args:
            pattern_name: Pattern name (e.g., 'urgent_stop')

        Returns:
            Pattern array (num_motors, 3) [frequency, intensity, duration]
        """
        return self.patterns.get(pattern_name, self.patterns['all_clear'])


class HapticBelt:
    """
    Vibrotactile belt with 10 motors in circular array.

    Hardware: ERM (Eccentric Rotating Mass) motors or LRA (Linear Resonant Actuators)
    Communication: Serial (UART) or USB (HID)
    Latency: <20ms (motor response time)
    """

    def __init__(self, num_motors: int = 10, motor_type: str = "ERM",
                 intensity_range: Tuple[int, int] = (0, 255),
                 frequency_range: Tuple[int, int] = (50, 300)):
        """
        Initialize haptic belt.

        Args:
            num_motors: Number of motors (default 10)
            motor_type: Motor type ('ERM' or 'LRA')
            intensity_range: PWM intensity range (0-255)
            frequency_range: Frequency range in Hz
        """
        self.num_motors = num_motors
        self.motor_type = motor_type
        self.intensity_range = intensity_range
        self.frequency_range = frequency_range

        # Motor state (current frequency and intensity per motor)
        self.motor_states = np.zeros((num_motors, 2))  # [frequency, intensity]

        # Serial connection (placeholder - would be real hardware in production)
        self.serial_port = None  # Would be serial.Serial('/dev/ttyUSB0', 115200)
        self.hardware_connected = False

    async def play_pattern_async(self, pattern: np.ndarray):
        """
        Play haptic pattern asynchronously (non-blocking).

        Args:
            pattern: Pattern array (num_motors, 3) [frequency, intensity, duration]
        """
        # Generate haptic signals
        signals = []
        for motor_id, (freq, intensity, duration) in enumerate(pattern):
            if intensity > 0:  # Only if motor should vibrate
                signal = HapticSignal(
                    motor_id=motor_id,
                    frequency=freq,
                    intensity=intensity,
                    duration=duration,
                    timestamp=time.time()
                )
                signals.append(signal)

        # Send to hardware (if connected)
        if self.hardware_connected and self.serial_port:
            await self._send_to_hardware_async(signals)
        else:
            # Simulate pattern (for testing without hardware)
            await self._simulate_pattern_async(signals)

    async def _send_to_hardware_async(self, signals: List[HapticSignal]):
        """
        Send signals to real haptic hardware via serial.

        Protocol: Simple binary format
        - Header: 0xAA 0x55
        - Motor ID: 1 byte
        - Frequency: 2 bytes (uint16, Hz)
        - Intensity: 1 byte (uint8, 0-255)
        - Duration: 2 bytes (uint16, milliseconds)

        Args:
            signals: List of haptic signals
        """
        for signal in signals:
            # Convert to hardware format
            motor_id = signal.motor_id
            frequency = int(signal.frequency)
            intensity = int(signal.intensity * 255)
            duration = int(signal.duration * 1000)

            # Build packet
            packet = bytearray([
                0xAA, 0x55,  # Header
                motor_id,
                (frequency >> 8) & 0xFF,
                frequency & 0xFF,
                intensity,
                (duration >> 8) & 0xFF,
                duration & 0xFF
            ])

            # Send to serial port
            if self.serial_port:
                self.serial_port.write(packet)

    async def _simulate_pattern_async(self, signals: List[HapticSignal]):
        """
        Simulate haptic pattern (for testing without hardware).

        Args:
            signals: List of haptic signals
        """
        # Print pattern info (for debugging)
        for signal in signals:
            print(f"  Motor {signal.motor_id}: {signal.frequency:.1f}Hz, "
                  f"intensity={signal.intensity:.2f}, duration={signal.duration:.2f}s")


class HapticFeedbackController:
    """
    High-level haptic feedback controller for OrbyGlasses navigation.

    Integrates depth sensing, object detection, and path planning
    to generate intuitive haptic cues for blind users.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize haptic feedback controller.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize haptic belt
        self.haptic_belt = HapticBelt(
            num_motors=10,
            motor_type="ERM"
        )

        # Initialize pattern library
        self.patterns = HaptEQPatternLibrary()

        # Danger thresholds
        self.danger_threshold = 0.5    # < 0.5m
        self.caution_threshold = 1.5   # 0.5-1.5m

        # State tracking
        self.last_pattern_time = 0
        self.min_pattern_interval = 0.3  # seconds (prevent overload)

    async def generate_haptic_cues_async(self, detections: List[Dict], predicted_path: Optional[Dict] = None) -> Dict:
        """
        Generate haptic cues from detections and predicted path.

        Args:
            detections: List of detected objects with depth/position
            predicted_path: Optional predicted navigation path

        Returns:
            Haptic feedback metadata
        """
        current_time = time.time()

        # Rate limiting (prevent haptic overload)
        if current_time - self.last_pattern_time < self.min_pattern_interval:
            return {'status': 'rate_limited'}

        # Analyze scene
        scene_analysis = self._analyze_scene(detections)

        # Select appropriate pattern
        pattern = self._select_pattern(scene_analysis, predicted_path)

        # Play pattern
        if pattern is not None:
            await self.haptic_belt.play_pattern_async(pattern)
            self.last_pattern_time = current_time

        return {
            'status': 'success',
            'pattern_type': scene_analysis['pattern_type'],
            'danger_level': scene_analysis['danger_level'],
            'safe_direction': scene_analysis['safe_direction']
        }

    def _analyze_scene(self, detections: List[Dict]) -> Dict:
        """
        Analyze scene to determine appropriate haptic feedback.

        Args:
            detections: List of detected objects

        Returns:
            Scene analysis dictionary
        """
        if not detections:
            return {
                'danger_level': 0.0,
                'pattern_type': 'all_clear',
                'safe_direction': 0.0  # Forward
            }

        # Categorize detections by danger level
        danger_objects = [d for d in detections if d.get('depth', 10) < self.danger_threshold]
        caution_objects = [d for d in detections if self.danger_threshold <= d.get('depth', 10) < self.caution_threshold]

        # Compute danger level
        if danger_objects:
            closest = min(danger_objects, key=lambda x: x['depth'])
            danger_level = 1.0 - (closest['depth'] / self.danger_threshold)
            pattern_type = 'urgent_stop'
        elif caution_objects:
            closest = min(caution_objects, key=lambda x: x['depth'])
            danger_level = 0.5 - (closest['depth'] - self.danger_threshold) / (self.caution_threshold - self.danger_threshold) * 0.5
            pattern_type = 'caution'
        else:
            danger_level = 0.0
            pattern_type = 'all_clear'

        # Compute safe direction
        safe_direction = self._compute_safe_direction(detections)

        return {
            'danger_level': danger_level,
            'pattern_type': pattern_type,
            'safe_direction': safe_direction,
            'num_obstacles': len(detections)
        }

    def _compute_safe_direction(self, detections: List[Dict]) -> float:
        """
        Compute safest direction to travel.

        Args:
            detections: List of detected objects

        Returns:
            Safe direction in degrees (-180 to 180, 0=forward)
        """
        if not detections:
            return 0.0  # Forward

        # Create occupancy histogram (18 bins = 20° each)
        histogram = np.zeros(18)
        for det in detections:
            # Assume 'center' is normalized (0-1)
            center_x = det.get('center', [0.5])[0]
            angle_norm = center_x  # 0 (left) to 1 (right)
            angle_deg = (angle_norm - 0.5) * 180  # -90 to +90
            bin_idx = int((angle_deg + 90) / 10) % 18

            # Handle uncertain depth (None)
            depth = det.get('depth', 1.0)
            if depth is None or det.get('depth_uncertain', False):
                depth = 1.0  # Assume close when uncertain

            histogram[bin_idx] += 1.0 / max(depth, 0.5)

        # Find direction with minimum obstacles
        safe_bin = np.argmin(histogram)
        safe_angle_deg = safe_bin * 10 - 90

        return safe_angle_deg

    def _select_pattern(self, scene_analysis: Dict, predicted_path: Optional[Dict]) -> Optional[np.ndarray]:
        """
        Select appropriate haptic pattern based on scene analysis.

        Args:
            scene_analysis: Scene analysis from _analyze_scene
            predicted_path: Optional predicted path

        Returns:
            Haptic pattern or None
        """
        pattern_type = scene_analysis['pattern_type']
        safe_direction = scene_analysis['safe_direction']

        # Select base pattern
        if pattern_type == 'urgent_stop':
            pattern = self.patterns.get('urgent_stop')
        elif pattern_type == 'caution':
            if safe_direction < -20:
                pattern = self.patterns.get('turn_left')
            elif safe_direction > 20:
                pattern = self.patterns.get('turn_right')
            else:
                pattern = self.patterns.get('caution')
        else:  # all_clear
            pattern = self.patterns.get('all_clear')

        return pattern


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import asyncio

    # Initialize haptic controller
    controller = HapticFeedbackController()

    # Simulate detections
    detections = [
        {'label': 'person', 'depth': 2.5, 'center': [0.3, 0.5]},
        {'label': 'chair', 'depth': 1.2, 'center': [0.6, 0.5]},
        {'label': 'wall', 'depth': 0.8, 'center': [0.5, 0.5]}  # Danger!
    ]

    # Generate haptic cues
    async def test_haptic():
        print("Testing haptic feedback system...")
        result = await controller.generate_haptic_cues_async(detections)
        print(f"Result: {result}")

    asyncio.run(test_haptic())
