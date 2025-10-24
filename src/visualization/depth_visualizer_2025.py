"""
OrbyGlasses 2025 - Dark-Themed Depth Visualization with Haptic/Audio Conversion

Revolutionary depth visualization system optimized for blind navigation:
- Hyper-contrast dark palette (black-to-obsidian gradients)
- Semantic heatmaps for spatial intuition
- Haptic/audio signal conversion for "felt" spatial awareness
- Perceptual uniform colormaps for accessibility

Target: Ultra-clear depth perception with multimodal feedback
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class DarkThemeDepthVisualizer:
    """
    2025 SOTA depth visualization with dark themes and multimodal conversion.

    Features:
    - Obsidian color scheme (black → dark blue → dark green → dark red)
    - Perceptually uniform gradients for accessibility
    - Edge-enhanced sharpening for maximum clarity
    - Histogram equalization for contrast boost
    - Semantic segmentation overlay (danger zones highlighted)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize dark-themed depth visualizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Color scheme: Obsidian (dark palette)
        self.danger_color = (0, 0, 180)      # Dark red (BGR)
        self.caution_color = (0, 120, 140)   # Dark orange (BGR)
        self.safe_color = (60, 100, 0)       # Dark green (BGR)
        self.far_color = (140, 60, 0)        # Dark blue (BGR)
        self.void_color = (20, 20, 20)       # Near-black (BGR)

        # Distance thresholds (meters)
        self.danger_threshold = 0.5    # < 0.5m: immediate danger
        self.caution_threshold = 1.5   # 0.5-1.5m: caution zone
        self.safe_threshold = 3.5      # 1.5-3.5m: safe zone
        # > 3.5m: far zone

        # Visualization parameters
        self.enable_edge_enhancement = True
        self.apply_histogram_eq = True
        self.overlay_semantic = True
        self.show_legend = True

    def create_obsidian_colormap(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Create obsidian (dark) colormap with perceptual uniformity.

        Color progression:
        - 0.0-0.2 (0-0.5m): Dark red → Bright red (DANGER)
        - 0.2-0.4 (0.5-1.5m): Bright red → Dark orange (CAUTION)
        - 0.4-0.7 (1.5-3.5m): Dark orange → Dark green (SAFE)
        - 0.7-1.0 (>3.5m): Dark green → Dark blue → Near-black (FAR)

        Args:
            depth_map: Normalized depth map (0-1, where 0=close, 1=far)

        Returns:
            Colored depth map (BGR, uint8)
        """
        h, w = depth_map.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        # Clip depth to [0, 1]
        depth_norm = np.clip(depth_map, 0, 1)

        # Apply histogram equalization for better contrast
        if self.apply_histogram_eq:
            depth_uint8 = (depth_norm * 255).astype(np.uint8)
            depth_eq = cv2.equalizeHist(depth_uint8).astype(np.float32) / 255.0
        else:
            depth_eq = depth_norm

        # Zone 1: Immediate danger (0-0.2) → Dark red to bright red
        mask1 = depth_eq < 0.2
        if np.any(mask1):
            t = depth_eq[mask1] / 0.2
            colored[mask1] = np.stack([
                np.zeros_like(t),                          # B = 0
                np.zeros_like(t),                          # G = 0
                (120 + t * 135).astype(np.uint8)          # R = 120 → 255
            ], axis=-1)

        # Zone 2: Caution (0.2-0.4) → Bright red to dark orange
        mask2 = (depth_eq >= 0.2) & (depth_eq < 0.4)
        if np.any(mask2):
            t = (depth_eq[mask2] - 0.2) / 0.2
            colored[mask2] = np.stack([
                np.zeros_like(t),                          # B = 0
                (t * 100).astype(np.uint8),               # G = 0 → 100
                (255 - t * 115).astype(np.uint8)          # R = 255 → 140
            ], axis=-1)

        # Zone 3: Safe (0.4-0.7) → Dark orange to dark green
        mask3 = (depth_eq >= 0.4) & (depth_eq < 0.7)
        if np.any(mask3):
            t = (depth_eq[mask3] - 0.4) / 0.3
            colored[mask3] = np.stack([
                np.zeros_like(t),                          # B = 0
                (100 + t * 60).astype(np.uint8),          # G = 100 → 160
                (140 - t * 140).astype(np.uint8)          # R = 140 → 0
            ], axis=-1)

        # Zone 4: Far (0.7-1.0) → Dark green to dark blue to near-black
        mask4 = depth_eq >= 0.7
        if np.any(mask4):
            t = (depth_eq[mask4] - 0.7) / 0.3
            colored[mask4] = np.stack([
                (t * 140).astype(np.uint8),               # B = 0 → 140
                (160 * (1 - t)).astype(np.uint8),         # G = 160 → 0
                np.zeros_like(t)                           # R = 0
            ], axis=-1)

        return colored

    def enhance_edges_func(self, colored_depth: np.ndarray) -> np.ndarray:
        """
        Apply edge enhancement for maximum clarity.

        Args:
            colored_depth: Colored depth map (BGR)

        Returns:
            Edge-enhanced depth map
        """
        # Sharpening kernel (high-pass filter)
        kernel = np.array([
            [-1, -1, -1],
            [-1,  9, -1],
            [-1, -1, -1]
        ], dtype=np.float32)

        # Apply sharpening
        sharpened = cv2.filter2D(colored_depth, -1, kernel)

        # Blend with original (80% sharpened, 20% original)
        enhanced = cv2.addWeighted(sharpened, 0.8, colored_depth, 0.2, 0)

        return enhanced

    def overlay_semantic_zones(self, colored_depth: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Overlay semantic zone boundaries for clarity.

        Args:
            colored_depth: Colored depth map
            depth_map: Normalized depth map (0-1)

        Returns:
            Depth map with zone overlays
        """
        overlay = colored_depth.copy()

        # Convert depth thresholds to normalized values
        max_depth = 10.0  # Assume max depth of 10m
        danger_norm = self.danger_threshold / max_depth
        caution_norm = self.caution_threshold / max_depth
        safe_norm = self.safe_threshold / max_depth

        # Create zone masks
        danger_mask = (depth_map < danger_norm).astype(np.uint8) * 255
        caution_mask = ((depth_map >= danger_norm) & (depth_map < caution_norm)).astype(np.uint8) * 255
        safe_mask = ((depth_map >= caution_norm) & (depth_map < safe_norm)).astype(np.uint8) * 255

        # Find contours for zone boundaries
        danger_contours, _ = cv2.findContours(danger_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        caution_contours, _ = cv2.findContours(caution_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        safe_contours, _ = cv2.findContours(safe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours (zone boundaries)
        cv2.drawContours(overlay, danger_contours, -1, (0, 0, 255), 2)    # Bright red
        cv2.drawContours(overlay, caution_contours, -1, (0, 165, 255), 1)  # Orange
        cv2.drawContours(overlay, safe_contours, -1, (0, 255, 0), 1)       # Green

        return overlay

    def add_legend(self, image: np.ndarray) -> np.ndarray:
        """
        Add color legend to depth visualization.

        Args:
            image: Depth visualization

        Returns:
            Image with legend
        """
        h, w = image.shape[:2]

        # Legend dimensions
        legend_height = 100
        legend_width = w
        legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8) + 30  # Dark background

        # Draw color gradient
        gradient_height = 30
        gradient_y = 20
        for x in range(legend_width):
            t = x / legend_width
            if t < 0.2:  # Danger zone
                color = (0, 0, int(120 + (t / 0.2) * 135))
            elif t < 0.4:  # Caution zone
                t_norm = (t - 0.2) / 0.2
                color = (0, int(t_norm * 100), int(255 - t_norm * 115))
            elif t < 0.7:  # Safe zone
                t_norm = (t - 0.4) / 0.3
                color = (0, int(100 + t_norm * 60), int(140 - t_norm * 140))
            else:  # Far zone
                t_norm = (t - 0.7) / 0.3
                color = (int(t_norm * 140), int(160 * (1 - t_norm)), 0)

            cv2.line(legend, (x, gradient_y), (x, gradient_y + gradient_height), color, 1)

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(legend, "DANGER", (10, gradient_y + gradient_height + 20), font, 0.5, (0, 0, 255), 1)
        cv2.putText(legend, "CAUTION", (int(w * 0.25), gradient_y + gradient_height + 20), font, 0.5, (0, 165, 255), 1)
        cv2.putText(legend, "SAFE", (int(w * 0.5), gradient_y + gradient_height + 20), font, 0.5, (0, 255, 0), 1)
        cv2.putText(legend, "FAR", (int(w * 0.75), gradient_y + gradient_height + 20), font, 0.5, (100, 100, 255), 1)

        # Concatenate legend with image
        result = np.vstack([image, legend])

        return result

    def visualize(self, depth_map: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Create dark-themed depth visualization with all enhancements.

        Args:
            depth_map: Raw depth map (metric depth or disparity)
            normalize: Whether to normalize depth map to [0, 1]

        Returns:
            Enhanced dark-themed depth visualization (BGR)
        """
        # Normalize depth if requested
        if normalize:
            depth_min = np.min(depth_map[depth_map > 0])
            depth_max = np.max(depth_map)
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
            # Invert (0=close, 1=far)
            depth_norm = 1.0 - depth_norm
        else:
            depth_norm = depth_map

        # Create obsidian colormap
        colored = self.create_obsidian_colormap(depth_norm)

        # Enhance edges
        if self.enable_edge_enhancement:
            colored = self.enhance_edges_func(colored)

        # Overlay semantic zones
        if self.overlay_semantic:
            colored = self.overlay_semantic_zones(colored, depth_norm)

        # Add legend
        if self.show_legend:
            colored = self.add_legend(colored)

        return colored


class HapticAudioConverter:
    """
    Convert depth maps to haptic/audio signals for "felt" spatial awareness.

    Features:
    - Vibrotactile patterns (frequency + intensity modulation)
    - Audio sonification (pitch gradients for distance)
    - Directional cues (left/center/right zones)
    - Danger alerts (urgent pulses for obstacles <0.5m)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize haptic/audio converter.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Haptic parameters
        self.num_motors = 10  # Vibrotactile belt with 10 motors
        self.min_frequency = 50   # Hz (low rumble)
        self.max_frequency = 300  # Hz (high buzz)
        self.min_intensity = 0    # 0-255 (PWM)
        self.max_intensity = 255

        # Audio parameters
        self.sample_rate = 16000  # Hz
        self.min_pitch = 200      # Hz (low tone)
        self.max_pitch = 2000     # Hz (high tone)
        self.tone_duration = 0.1  # seconds

        # Spatial zones (divide image into left/center/right)
        self.num_zones = 3  # Left, Center, Right

    def depth_to_haptic_pattern(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Convert depth map to haptic vibration pattern.

        Pattern encoding:
        - Frequency: Inversely proportional to distance (close=high freq)
        - Intensity: Inversely proportional to distance (close=high intensity)
        - Spatial: Map to motor positions (left/center/right)

        Args:
            depth_map: Normalized depth map (0=close, 1=far)

        Returns:
            Haptic pattern (num_motors x 2) - [frequency, intensity] per motor
        """
        h, w = depth_map.shape

        # Divide image into zones (angular sectors)
        zone_width = w // self.num_motors
        haptic_pattern = np.zeros((self.num_motors, 2))  # [frequency, intensity]

        for i in range(self.num_motors):
            # Extract zone
            zone_start = i * zone_width
            zone_end = (i + 1) * zone_width if i < self.num_motors - 1 else w
            zone = depth_map[:, zone_start:zone_end]

            # Compute minimum depth in zone (closest obstacle)
            min_depth = np.min(zone[zone > 0]) if np.any(zone > 0) else 1.0

            # Convert to frequency (inverse relationship)
            frequency = self.min_frequency + (1.0 - min_depth) * (self.max_frequency - self.min_frequency)

            # Convert to intensity (inverse relationship)
            intensity = self.min_intensity + (1.0 - min_depth) * (self.max_intensity - self.min_intensity)

            # Danger boost (if obstacle < 0.5m, max frequency/intensity)
            if min_depth < 0.2:  # Normalized (0.5m / 10m = 0.05, but use 0.2 for safety)
                frequency = self.max_frequency
                intensity = self.max_intensity

            haptic_pattern[i] = [frequency, intensity]

        return haptic_pattern

    def depth_to_audio_sonification(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Convert depth map to audio sonification (pitch gradient).

        Sonification encoding:
        - Pitch: Inversely proportional to distance (close=high pitch)
        - Volume: Inversely proportional to distance (close=loud)
        - Stereo: Left/right panning based on obstacle position

        Args:
            depth_map: Normalized depth map (0=close, 1=far)

        Returns:
            Audio signal (stereo, sample_rate)
        """
        h, w = depth_map.shape

        # Divide image into 3 zones (left, center, right)
        zone_width = w // self.num_zones
        audio_signal = np.zeros((int(self.tone_duration * self.sample_rate), 2))  # Stereo

        for i in range(self.num_zones):
            # Extract zone
            zone_start = i * zone_width
            zone_end = (i + 1) * zone_width if i < self.num_zones - 1 else w
            zone = depth_map[:, zone_start:zone_end]

            # Compute minimum depth in zone
            min_depth = np.min(zone[zone > 0]) if np.any(zone > 0) else 1.0

            # Convert to pitch (inverse relationship)
            pitch = self.min_pitch + (1.0 - min_depth) * (self.max_pitch - self.min_pitch)

            # Generate sine wave tone
            t = np.linspace(0, self.tone_duration, int(self.tone_duration * self.sample_rate))
            tone = np.sin(2 * np.pi * pitch * t)

            # Apply volume envelope (ADSR)
            envelope = np.exp(-3 * t)  # Exponential decay
            tone = tone * envelope * (1.0 - min_depth)  # Scale by proximity

            # Stereo panning
            if i == 0:  # Left zone
                audio_signal[:, 0] += tone * 0.8
                audio_signal[:, 1] += tone * 0.2
            elif i == 1:  # Center zone
                audio_signal[:, 0] += tone * 0.5
                audio_signal[:, 1] += tone * 0.5
            else:  # Right zone
                audio_signal[:, 0] += tone * 0.2
                audio_signal[:, 1] += tone * 0.8

        # Normalize audio
        audio_signal = audio_signal / (np.max(np.abs(audio_signal)) + 1e-8)

        return audio_signal

    def generate_danger_alert(self, depth_map: np.ndarray, threshold: float = 0.2) -> Optional[Dict]:
        """
        Generate urgent haptic/audio alert for immediate danger.

        Args:
            depth_map: Normalized depth map (0=close, 1=far)
            threshold: Danger threshold (normalized)

        Returns:
            Alert dictionary with haptic/audio signals, or None if no danger
        """
        # Check for danger zone obstacles
        danger_mask = depth_map < threshold

        if not np.any(danger_mask):
            return None

        # Find closest obstacle
        min_depth = np.min(depth_map[danger_mask])

        # Generate urgent haptic pattern (all motors max intensity)
        haptic_pattern = np.ones((self.num_motors, 2))
        haptic_pattern[:, 0] = self.max_frequency
        haptic_pattern[:, 1] = self.max_intensity

        # Generate urgent audio alert (high-pitched beep)
        t = np.linspace(0, 0.2, int(0.2 * self.sample_rate))
        beep_freq = 1000  # Hz
        beep = np.sin(2 * np.pi * beep_freq * t)
        audio_alert = np.column_stack([beep, beep])  # Stereo

        return {
            'danger_level': 1.0 - min_depth,
            'haptic_pattern': haptic_pattern,
            'audio_alert': audio_alert,
            'message': f"⚠️ DANGER: Obstacle at {min_depth * 10:.1f}m"
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Create sample depth map
    depth_sample = np.random.rand(480, 640) * 10.0  # Random depth 0-10m

    # Initialize visualizer
    visualizer = DarkThemeDepthVisualizer()

    # Visualize depth
    depth_colored = visualizer.visualize(depth_sample, normalize=True)

    # Display
    cv2.imshow("Dark-Themed Depth Visualization", depth_colored)
    cv2.waitKey(0)

    # Initialize haptic/audio converter
    converter = HapticAudioConverter()

    # Convert to haptic pattern
    haptic_pattern = converter.depth_to_haptic_pattern(depth_sample / 10.0)
    print(f"Haptic pattern (10 motors): {haptic_pattern.shape}")
    print(f"Motor 0 (leftmost): frequency={haptic_pattern[0, 0]:.1f} Hz, intensity={haptic_pattern[0, 1]:.0f}/255")

    # Convert to audio sonification
    audio_signal = converter.depth_to_audio_sonification(depth_sample / 10.0)
    print(f"Audio signal: {audio_signal.shape} (stereo)")

    # Check for danger
    danger_alert = converter.generate_danger_alert(depth_sample / 10.0, threshold=0.2)
    if danger_alert:
        print(f"⚠️ DANGER ALERT: {danger_alert['message']}")

    cv2.destroyAllWindows()
