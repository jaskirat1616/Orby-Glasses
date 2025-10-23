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
        Draw clean overlay for camera window.

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

            # Color by distance
            if depth < 1.0:
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

            # Label with distance
            text = f"{label} {depth:.1f}m"
            cv2.putText(overlay, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Top status bar - simple
        cv2.rectangle(overlay, (0, 0), (w, 35), self.BLACK, -1)

        # FPS
        fps_color = self.GREEN if fps > 15 else self.YELLOW if fps > 10 else self.RED
        cv2.putText(overlay, f"FPS: {fps:.0f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)

        # Status
        if danger_count > 0:
            status = "DANGER"
            status_color = self.RED
        elif caution_count > 0:
            status = "CAUTION"
            status_color = self.YELLOW
        else:
            status = "CLEAR"
            status_color = self.GREEN

        cv2.putText(overlay, status, (w - 120, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return overlay


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
