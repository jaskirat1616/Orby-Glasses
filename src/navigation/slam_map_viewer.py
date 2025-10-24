"""
SLAM Map Viewer - Always-visible 2D map like robots use
Shows top-down view of explored environment
"""
import cv2
import numpy as np
from typing import Dict, List, Tuple


class SLAMMapViewer:
    """Creates a persistent 2D top-down map view of SLAM exploration."""

    def __init__(self, map_size: int = 800, meters_per_pixel: float = 0.05):
        """
        Initialize map viewer.

        Args:
            map_size: Size of map image in pixels (square)
            meters_per_pixel: Resolution of map (meters per pixel)
        """
        self.map_size = map_size
        self.meters_per_pixel = meters_per_pixel
        self.center = map_size // 2

        # Create persistent map
        self.map_image = np.zeros((map_size, map_size, 3), dtype=np.uint8)
        self.map_image[:] = (30, 30, 30)  # Dark gray background

        # Track what we've drawn
        self.trajectory_points = []
        self.landmarks = []

        # Colors
        self.color_trajectory = (0, 255, 0)  # Green for path
        self.color_current = (0, 0, 255)  # Red for current position
        self.color_landmarks = (255, 255, 0)  # Cyan for map points
        self.color_grid = (50, 50, 50)  # Grid lines

    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """
        Convert world coordinates (meters) to pixel coordinates.

        Args:
            x, y: Position in meters (world frame)

        Returns:
            (px, py): Pixel coordinates
        """
        # Convert meters to pixels
        px = int(self.center + (x / self.meters_per_pixel))
        py = int(self.center - (y / self.meters_per_pixel))  # Flip Y for image coords

        return px, py

    def draw_grid(self, grid_spacing: float = 1.0):
        """
        Draw grid lines on map.

        Args:
            grid_spacing: Spacing between grid lines in meters
        """
        # Calculate grid spacing in pixels
        pixel_spacing = int(grid_spacing / self.meters_per_pixel)

        # Draw vertical lines
        for i in range(0, self.map_size, pixel_spacing):
            cv2.line(self.map_image, (i, 0), (i, self.map_size),
                    self.color_grid, 1)

        # Draw horizontal lines
        for i in range(0, self.map_size, pixel_spacing):
            cv2.line(self.map_image, (0, i), (self.map_size, i),
                    self.color_grid, 1)

        # Draw center crosshair (origin)
        cv2.line(self.map_image,
                (self.center - 10, self.center),
                (self.center + 10, self.center),
                (100, 100, 100), 2)
        cv2.line(self.map_image,
                (self.center, self.center - 10),
                (self.center, self.center + 10),
                (100, 100, 100), 2)

    def update(self, slam_result: Dict, map_points: Dict = None):
        """
        Update map with new SLAM data.

        Args:
            slam_result: SLAM tracking result
            map_points: Dictionary of map points (optional)
        """
        if slam_result is None:
            return

        # Get current position
        position = slam_result.get('position', [0, 0, 0])
        x, y, z = position

        # Convert to pixel coords
        px, py = self.world_to_pixel(x, y)

        # Check if in bounds
        if 0 <= px < self.map_size and 0 <= py < self.map_size:
            # Add to trajectory
            self.trajectory_points.append((px, py))

            # Draw trajectory (connect last few points)
            if len(self.trajectory_points) > 1:
                # Draw lines between recent points
                for i in range(max(0, len(self.trajectory_points) - 50),
                             len(self.trajectory_points) - 1):
                    pt1 = self.trajectory_points[i]
                    pt2 = self.trajectory_points[i + 1]
                    cv2.line(self.map_image, pt1, pt2, self.color_trajectory, 2)

            # Draw current position (larger circle)
            cv2.circle(self.map_image, (px, py), 8, self.color_current, -1)
            cv2.circle(self.map_image, (px, py), 10, (255, 255, 255), 2)

        # Draw map points (landmarks) if available
        if map_points:
            for point_id, point in map_points.items():
                if hasattr(point, 'position'):
                    px_l, py_l = self.world_to_pixel(point.position[0], point.position[1])
                    if 0 <= px_l < self.map_size and 0 <= py_l < self.map_size:
                        # Small dots for landmarks
                        cv2.circle(self.map_image, (px_l, py_l), 2,
                                 self.color_landmarks, -1)

    def get_map_image(self) -> np.ndarray:
        """
        Get current map image with overlays.

        Returns:
            Map image with info overlays
        """
        # Create copy for overlay
        display = self.map_image.copy()

        # Add info text
        info_text = [
            f"Trajectory: {len(self.trajectory_points)} points",
            f"Scale: {self.meters_per_pixel*100:.1f} cm/pixel",
            "Grid: 1m spacing"
        ]

        y_offset = 30
        for text in info_text:
            cv2.putText(display, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

        # Add compass (N/S/E/W)
        compass_x, compass_y = self.map_size - 60, 60
        cv2.putText(display, "N", (compass_x, compass_y - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.arrowedLine(display, (compass_x, compass_y),
                       (compass_x, compass_y - 25), (255, 255, 255), 2)

        return display

    def reset(self):
        """Reset map to blank state."""
        self.map_image = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.map_image[:] = (30, 30, 30)
        self.trajectory_points = []
        self.landmarks = []
        self.draw_grid()

    def save_map(self, filename: str):
        """Save map to file."""
        cv2.imwrite(filename, self.map_image)
