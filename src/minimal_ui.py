"""
Minimal UI for Blind Navigation
Clean, simple, only essential information.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional


class MinimalUI:
    """Ultra-minimal UI focused on blind user needs."""

    def __init__(self):
        """Initialize minimal UI."""
        # Colors (BGR)
        self.RED = (0, 0, 255)
        self.YELLOW = (0, 255, 255)
        self.GREEN = (0, 255, 0)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

    def draw_overlay(self, frame: np.ndarray, detections: List[Dict],
                     fps: float, guidance_text: str) -> np.ndarray:
        """
        Draw minimal overlay - only what matters.

        Args:
            frame: Input frame
            detections: Object detections
            fps: Current FPS
            guidance_text: LLM-generated guidance

        Returns:
            Frame with minimal overlay
        """
        h, w = frame.shape[:2]

        # Just draw boxes on objects
        for det in detections:
            bbox = det.get('bbox', [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            depth = det.get('depth', 0.0)
            label = det.get('label', '')

            # Color by distance
            if depth < 1.0:
                color = self.RED
            elif depth < 2.5:
                color = self.YELLOW
            else:
                color = self.GREEN

            # Simple box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label
            text = f"{label} {depth:.1f}m"
            cv2.putText(frame, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Top bar - FPS only
        cv2.rectangle(frame, (0, 0), (w, 35), self.BLACK, -1)
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.WHITE, 2)

        # Bottom bar - Guidance text from LLM
        if guidance_text:
            # Multi-line support for longer guidance
            lines = guidance_text.split('.')
            bar_height = 60 + (len(lines) * 25)
            cv2.rectangle(frame, (0, h - bar_height), (w, h), self.BLACK, -1)

            y_offset = h - bar_height + 25
            for line in lines[:3]:  # Max 3 lines
                if line.strip():
                    cv2.putText(frame, line.strip(), (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.WHITE, 2)
                    y_offset += 25

        return frame

    def create_slam_map(self, slam_result: Optional[Dict], map_points: np.ndarray,
                       size: int = 400) -> np.ndarray:
        """
        Create simple SLAM map.

        Args:
            slam_result: SLAM tracking result
            map_points: Map point cloud
            size: Map size in pixels

        Returns:
            Map image
        """
        # Black background
        map_img = np.zeros((size, size, 3), dtype=np.uint8)

        # Grid
        grid_step = size // 10
        for i in range(0, size, grid_step):
            cv2.line(map_img, (i, 0), (i, size), (30, 30, 30), 1)
            cv2.line(map_img, (0, i), (size, i), (30, 30, 30), 1)

        center = size // 2
        scale = 20  # pixels per meter

        # Draw map points
        if map_points is not None:
            try:
                # Handle different map_points formats
                if isinstance(map_points, np.ndarray):
                    points_to_draw = map_points[:500] if len(map_points) > 500 else map_points
                    for point in points_to_draw:
                        if len(point) >= 3:
                            x = int(center + point[0] * scale)
                            y = int(center - point[2] * scale)  # Flip Y for display
                            if 0 <= x < size and 0 <= y < size:
                                cv2.circle(map_img, (x, y), 2, (100, 100, 100), -1)
                elif isinstance(map_points, dict):
                    # Handle dict format
                    points_list = list(map_points.values())[:500]
                    for point in points_list:
                        if hasattr(point, '__len__') and len(point) >= 3:
                            x = int(center + point[0] * scale)
                            y = int(center - point[2] * scale)
                            if 0 <= x < size and 0 <= y < size:
                                cv2.circle(map_img, (x, y), 2, (100, 100, 100), -1)
            except Exception:
                pass  # Skip if map points format is unexpected

        # Draw robot position
        if slam_result:
            pos = slam_result.get('position', [0, 0, 0])
            robot_x = int(center + pos[0] * scale)
            robot_y = int(center - pos[2] * scale)

            # Robot marker
            cv2.circle(map_img, (robot_x, robot_y), 8, (0, 255, 0), -1)
            cv2.circle(map_img, (robot_x, robot_y), 12, (0, 255, 0), 2)

            # Direction indicator
            cv2.arrowedLine(map_img, (robot_x, robot_y),
                          (robot_x, robot_y - 20), (0, 255, 0), 2)

        # Title
        cv2.putText(map_img, "MAP", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.WHITE, 2)

        # Scale
        cv2.line(map_img, (10, size - 30), (10 + scale, size - 30), self.WHITE, 2)
        cv2.putText(map_img, "1m", (15, size - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.WHITE, 1)

        return map_img
