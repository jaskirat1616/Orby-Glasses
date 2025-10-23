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
        Draw clean robot-style overlay.

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

        # Draw center crosshair (robot's focus point)
        center_x, center_y = w // 2, h // 2
        cv2.line(overlay, (center_x - 20, center_y), (center_x + 20, center_y),
                self.GREEN, 1)
        cv2.line(overlay, (center_x, center_y - 20), (center_x, center_y + 20),
                self.GREEN, 1)
        cv2.circle(overlay, (center_x, center_y), 30, self.GREEN, 1)

        # Draw zone indicators (left, center, right)
        left_bound = w // 3
        right_bound = 2 * w // 3
        cv2.line(overlay, (left_bound, 0), (left_bound, h), self.GRAY, 1)
        cv2.line(overlay, (right_bound, 0), (right_bound, h), self.GRAY, 1)

        # Draw detections with clean boxes
        danger_count = 0
        caution_count = 0

        for det in detections:
            bbox = det.get('bbox', [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            label = det.get('label', 'object')
            depth = det.get('depth', 0.0)

            # Color by risk
            if depth < 1.0:
                color = self.RED
                danger_count += 1
            elif depth < 2.5:
                color = self.YELLOW
                caution_count += 1
            else:
                color = self.GREEN

            # Clean box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            # Label
            text = f"{label} {depth:.1f}m"
            cv2.putText(overlay, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Top status bar
        status_h = 60
        cv2.rectangle(overlay, (0, 0), (w, status_h), self.BLACK, -1)

        # FPS indicator
        fps_color = self.GREEN if fps > 15 else self.YELLOW if fps > 10 else self.RED
        cv2.putText(overlay, f"FPS: {fps:.0f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)

        # Object counts
        cv2.putText(overlay, f"DANGER: {danger_count}", (150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.RED if danger_count > 0 else self.WHITE, 2)
        cv2.putText(overlay, f"CAUTION: {caution_count}", (300, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.YELLOW if caution_count > 0 else self.WHITE, 2)

        # Safe direction arrow
        self._draw_direction_arrow(overlay, safe_direction, center_x, status_h + 30)

        # Status text
        if danger_count > 0:
            status = "STOP"
            status_color = self.RED
        elif caution_count > 0:
            status = "CAUTION"
            status_color = self.YELLOW
        else:
            status = "CLEAR"
            status_color = self.GREEN

        cv2.putText(overlay, status, (w - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        return overlay

    def _draw_direction_arrow(self, frame: np.ndarray, direction: str,
                             center_x: int, y: int):
        """Draw directional arrow indicator."""
        arrow_length = 40

        if direction == 'forward':
            # Up arrow
            cv2.arrowedLine(frame, (center_x, y + arrow_length), (center_x, y),
                          self.GREEN, 3, tipLength=0.3)
        elif direction == 'left':
            # Left arrow
            cv2.arrowedLine(frame, (center_x + arrow_length, y), (center_x - arrow_length, y),
                          self.BLUE, 3, tipLength=0.3)
        elif direction == 'right':
            # Right arrow
            cv2.arrowedLine(frame, (center_x - arrow_length, y), (center_x + arrow_length, y),
                          self.BLUE, 3, tipLength=0.3)
        elif direction == 'stop':
            # Stop indicator
            cv2.circle(frame, (center_x, y), 20, self.RED, 3)
            cv2.line(frame, (center_x - 15, y - 15), (center_x + 15, y + 15), self.RED, 3)
            cv2.line(frame, (center_x - 15, y + 15), (center_x + 15, y - 15), self.RED, 3)

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
