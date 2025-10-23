"""
Robot-Style Navigation UI
Clean, fast, informative - like a real robot's vision system.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional


class RobotUI:
    """Clean robot-style navigation interface."""

    def __init__(self, width: int = 640, height: int = 480):
        """Initialize robot UI."""
        self.width = width
        self.height = height

        # Colors (BGR)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (0, 255, 255)
        self.RED = (0, 0, 255)
        self.BLUE = (255, 100, 0)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (100, 100, 100)

    def draw_clean_overlay(self, frame: np.ndarray, detections: List[Dict],
                          fps: float, safe_direction: str) -> np.ndarray:
        """
        Draw clean navigation overlay with large readable text.

        Args:
            frame: Input frame
            detections: Object detections
            fps: Current FPS
            safe_direction: Safe direction to move

        Returns:
            Frame with overlay
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # Calculate scaling factor for larger display
        scale = max(w / 640, 1.0)

        # Draw center crosshair for aiming
        center_x, center_y = w // 2, h // 2
        crosshair_size = int(30 * scale)
        cv2.line(overlay, (center_x - crosshair_size, center_y),
                (center_x + crosshair_size, center_y), self.GREEN, 2)
        cv2.line(overlay, (center_x, center_y - crosshair_size),
                (center_x, center_y + crosshair_size), self.GREEN, 2)
        cv2.circle(overlay, (center_x, center_y), crosshair_size, self.GREEN, 2)
        cv2.circle(overlay, (center_x, center_y), crosshair_size + 10, self.GREEN, 1)

        # Draw zone indicators (left, center, right)
        left_bound = w // 3
        right_bound = 2 * w // 3
        cv2.line(overlay, (left_bound, 0), (left_bound, h), self.GRAY, 2)
        cv2.line(overlay, (right_bound, 0), (right_bound, h), self.GRAY, 2)

        # Draw detections with enhanced boxes
        danger_count = 0
        caution_count = 0

        for det in detections:
            bbox = det.get('bbox', [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            label = det.get('label', 'object')
            depth = det.get('depth', 0.0)

            # Color by distance/risk
            if depth < 1.0:
                color = self.RED
                danger_count += 1
                thickness = 4
            elif depth < 2.5:
                color = self.YELLOW
                caution_count += 1
                thickness = 3
            else:
                color = self.GREEN
                thickness = 2

            # Draw detection box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            # Corner brackets for clarity
            corner_len = int(20 * scale)
            cv2.line(overlay, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
            cv2.line(overlay, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)
            cv2.line(overlay, (x2, y1), (x2 - corner_len, y1), color, thickness + 1)
            cv2.line(overlay, (x2, y1), (x2, y1 + corner_len), color, thickness + 1)

            # Large, clear label with background
            text = f"{label.upper()} {depth:.1f}m"
            font_scale = 0.9 * scale
            font_thickness = int(2 * scale)
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Background for text
            text_y = max(y1 - 10, text_h + 10)
            cv2.rectangle(overlay,
                         (x1, text_y - text_h - 8),
                         (x1 + text_w + 10, text_y + 5),
                         color, -1)

            # Text
            cv2.putText(overlay, text, (x1 + 5, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)

        # Top status bar with large text
        status_h = int(80 * scale)
        # Semi-transparent dark background
        overlay_bg = overlay.copy()
        cv2.rectangle(overlay_bg, (0, 0), (w, status_h), self.BLACK, -1)
        cv2.addWeighted(overlay_bg, 0.7, overlay, 0.3, 0, overlay)

        # Large FPS indicator
        fps_color = self.GREEN if fps > 15 else self.YELLOW if fps > 10 else self.RED
        fps_scale = 1.2 * scale
        fps_thick = int(3 * scale)
        cv2.putText(overlay, f"FPS: {fps:.0f}", (int(15 * scale), int(45 * scale)),
                   cv2.FONT_HERSHEY_SIMPLEX, fps_scale, fps_color, fps_thick)

        # Object counts with larger text
        count_scale = 0.9 * scale
        count_thick = int(2 * scale)
        danger_x = int(200 * scale)
        cv2.putText(overlay, f"DANGER: {danger_count}", (danger_x, int(45 * scale)),
                   cv2.FONT_HERSHEY_SIMPLEX, count_scale,
                   self.RED if danger_count > 0 else self.WHITE, count_thick)

        caution_x = int(450 * scale)
        cv2.putText(overlay, f"CAUTION: {caution_count}", (caution_x, int(45 * scale)),
                   cv2.FONT_HERSHEY_SIMPLEX, count_scale,
                   self.YELLOW if caution_count > 0 else self.WHITE, count_thick)

        # Direction arrow
        self._draw_direction_arrow(overlay, safe_direction, center_x, status_h + int(40 * scale))

        # Status text
        if danger_count > 0:
            status = "⚠ STOP"
            status_color = self.RED
        elif caution_count > 0:
            status = "⚠ CAUTION"
            status_color = self.YELLOW
        else:
            status = "✓ CLEAR"
            status_color = self.GREEN

        status_scale = 1.2 * scale
        status_thick = int(3 * scale)
        # Get text size to position from right
        (status_w, status_h_text), _ = cv2.getTextSize(
            status, cv2.FONT_HERSHEY_SIMPLEX, status_scale, status_thick
        )
        cv2.putText(overlay, status, (w - status_w - int(20 * scale), int(50 * scale)),
                   cv2.FONT_HERSHEY_SIMPLEX, status_scale, status_color, status_thick)

        # Corner frame markers
        frame_size = int(40 * scale)
        frame_thick = 3
        # Top-left
        cv2.line(overlay, (0, 0), (frame_size, 0), self.BLUE, frame_thick)
        cv2.line(overlay, (0, 0), (0, frame_size), self.BLUE, frame_thick)
        # Top-right
        cv2.line(overlay, (w, 0), (w - frame_size, 0), self.BLUE, frame_thick)
        cv2.line(overlay, (w, 0), (w, frame_size), self.BLUE, frame_thick)
        # Bottom-left
        cv2.line(overlay, (0, h), (frame_size, h), self.BLUE, frame_thick)
        cv2.line(overlay, (0, h), (0, h - frame_size), self.BLUE, frame_thick)
        # Bottom-right
        cv2.line(overlay, (w, h), (w - frame_size, h), self.BLUE, frame_thick)
        cv2.line(overlay, (w, h), (w, h - frame_size), self.BLUE, frame_thick)

        return overlay

    def _draw_direction_arrow(self, frame: np.ndarray, direction: str,
                             center_x: int, y: int):
        """Draw directional arrow."""
        h, w = frame.shape[:2]
        scale = max(w / 640, 1.0)
        arrow_length = int(60 * scale)
        arrow_thick = int(5 * scale)

        if direction == 'forward':
            # Large up arrow
            cv2.arrowedLine(frame, (center_x, y + arrow_length), (center_x, y),
                          self.GREEN, arrow_thick, tipLength=0.4)
            # Text below
            cv2.putText(frame, "FORWARD", (center_x - 50, y + arrow_length + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7 * scale, self.GREEN, 2)
        elif direction == 'left':
            # Large left arrow
            cv2.arrowedLine(frame, (center_x + arrow_length, y), (center_x - arrow_length, y),
                          self.BLUE, arrow_thick, tipLength=0.4)
            cv2.putText(frame, "← GO LEFT", (center_x - 80, y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, self.BLUE, 2)
        elif direction == 'right':
            # Large right arrow
            cv2.arrowedLine(frame, (center_x - arrow_length, y), (center_x + arrow_length, y),
                          self.BLUE, arrow_thick, tipLength=0.4)
            cv2.putText(frame, "GO RIGHT →", (center_x - 80, y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8 * scale, self.BLUE, 2)
        elif direction == 'stop':
            # Large stop indicator
            radius = int(30 * scale)
            cv2.circle(frame, (center_x, y), radius, self.RED, arrow_thick)
            cross_size = int(22 * scale)
            cv2.line(frame, (center_x - cross_size, y - cross_size),
                    (center_x + cross_size, y + cross_size), self.RED, arrow_thick)
            cv2.line(frame, (center_x - cross_size, y + cross_size),
                    (center_x + cross_size, y - cross_size), self.RED, arrow_thick)
            cv2.putText(frame, "STOP!", (center_x - 40, y + radius + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, self.RED, 3)

    def create_mini_map(self, slam_result: Optional[Dict], size: int = 300) -> np.ndarray:
        """
        Create clean mini-map like robots use.

        Args:
            slam_result: SLAM tracking result
            size: Map size in pixels

        Returns:
            Mini-map image
        """
        # Black background
        mini_map = np.zeros((size, size, 3), dtype=np.uint8)

        # Draw grid
        grid_step = size // 10
        for i in range(0, size, grid_step):
            cv2.line(mini_map, (i, 0), (i, size), self.GRAY, 1)
            cv2.line(mini_map, (0, i), (size, i), self.GRAY, 1)

        if slam_result:
            # Draw robot position (center of map)
            center = (size // 2, size // 2)
            cv2.circle(mini_map, center, 8, self.GREEN, -1)

            # Draw heading indicator
            # TODO: Add orientation from SLAM
            cv2.arrowedLine(mini_map, center, (center[0], center[1] - 20),
                          self.GREEN, 2, tipLength=0.4)

        # Map title
        cv2.putText(mini_map, "MAP", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.WHITE, 1)

        return mini_map

    def create_depth_view(self, depth_map: np.ndarray, size: int = 300) -> np.ndarray:
        """
        Create clean depth visualization.

        Args:
            depth_map: Depth map
            size: View size

        Returns:
            Depth visualization
        """
        if depth_map is None:
            return np.zeros((size, size, 3), dtype=np.uint8)

        # Resize depth map
        depth_resized = cv2.resize(depth_map, (size, size))

        # Convert to color (use TURBO colormap - better than JET)
        depth_colored = cv2.applyColorMap(
            (depth_resized * 255).astype(np.uint8),
            cv2.COLORMAP_TURBO
        )

        # Add title
        cv2.putText(depth_colored, "DEPTH", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.WHITE, 2)

        return depth_colored
