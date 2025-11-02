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
                          fps: float, safe_direction: str, depth_map: np.ndarray = None) -> np.ndarray:
        """
        Draw clean overlay for camera window with enhanced visualizations.

        Args:
            frame: Input frame
            detections: Object detections
            fps: Current FPS
            safe_direction: Safe direction to move
            depth_map: Optional depth map for overlay

        Returns:
            Frame with overlay
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]

        # CLEAN: No depth overlay on camera window (confusing for users)
        # Depth is shown in separate window with better visualization

        # Draw simple boxes on objects
        danger_count = 0
        caution_count = 0

        for det in detections:
            bbox = det.get('bbox', [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            label = det.get('label', 'object')
            depth = det.get('depth', 0.0)

            # Handle None depth
            if depth is None or det.get('depth_uncertain', False):
                depth = 1.0  # Assume close when uncertain
                color = self.RED
                danger_count += 1
                thickness = 3
            # Color by distance
            elif depth < 1.0:
                color = self.RED
                danger_count += 1
                thickness = 3
            elif depth < 2.5:
                color = self.YELLOW
                caution_count += 1
                thickness = 2
            else:
                color = self.GREEN
                thickness = 2

            # Simple box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

            # Label with distance (increased text size) - with black background for readability
            text = f"{label} {depth:.1f}m"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)
            cv2.rectangle(overlay, (x1, y1 - text_h - 10), (x1 + text_w + 5, y1 - 5), self.BLACK, -1)
            cv2.putText(overlay, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)  # Bigger text and thicker

        # Draw safe direction arrow
        if safe_direction and danger_count > 0:
            self._draw_safe_direction_arrow(overlay, safe_direction, (w, h))

        # Top status bar - simple
        cv2.rectangle(overlay, (0, 0), (w, 35), self.BLACK, -1)

        # FPS removed - cleaner UI

        # Status (increased text size)
        if danger_count > 0:
            status = "DANGER"
            status_color = self.RED
        elif caution_count > 0:
            status = "CAUTION"
            status_color = self.YELLOW
        else:
            status = "CLEAR"
            status_color = self.GREEN

        cv2.putText(overlay, status, (w - 160, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)

        return overlay

    def _create_depth_zones_overlay(self, depth_map: np.ndarray, frame_size: tuple) -> np.ndarray:
        """
        Create colored overlay showing danger/caution/safe zones.

        Args:
            depth_map: Normalized depth map (0-1)
            frame_size: (width, height) of output frame

        Returns:
            RGB overlay image
        """
        w, h = frame_size

        # Resize depth map if needed
        if depth_map.shape[:2] != (h, w):
            depth_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LANCZOS4)
        else:
            depth_resized = depth_map.copy()

        # Create color overlay based on depth zones
        overlay = np.zeros((h, w, 3), dtype=np.uint8)

        # Danger zone: red (< 0.3 normalized ~ < 1.5m)
        danger_mask = depth_resized < 0.3
        overlay[danger_mask] = [0, 0, 100]  # Dark red

        # Caution zone: yellow (0.3-0.5 ~ 1.5-3.5m)
        caution_mask = (depth_resized >= 0.3) & (depth_resized < 0.5)
        overlay[caution_mask] = [0, 100, 100]  # Dark yellow

        # Safe zone: green (> 0.5 ~ > 3.5m)
        safe_mask = depth_resized >= 0.5
        overlay[safe_mask] = [0, 100, 0]  # Dark green

        return overlay

    def _draw_safe_direction_arrow(self, frame: np.ndarray, direction: str, frame_size: tuple):
        """
        Draw arrow indicating safe direction to move.

        Args:
            frame: Frame to draw on
            direction: 'left', 'right', or 'forward'
            frame_size: (width, height)
        """
        w, h = frame_size
        center_x, center_y = w // 2, h - 100

        if direction == 'left':
            # Arrow pointing left
            pt1 = (center_x - 50, center_y)
            pt2 = (center_x - 100, center_y)
            cv2.arrowedLine(frame, pt1, pt2, self.GREEN, 5, tipLength=0.3)
            cv2.putText(frame, "GO LEFT", (center_x - 150, center_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.GREEN, 3)
        elif direction == 'right':
            # Arrow pointing right
            pt1 = (center_x + 50, center_y)
            pt2 = (center_x + 100, center_y)
            cv2.arrowedLine(frame, pt1, pt2, self.GREEN, 5, tipLength=0.3)
            cv2.putText(frame, "GO RIGHT", (center_x + 20, center_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.GREEN, 3)


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
