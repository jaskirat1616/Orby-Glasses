"""
Advanced Navigation Panel - Robotics-Style Multi-View Display
Combines multiple navigation visualizations without impacting main loop performance.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import time


class AdvancedNavigationPanel:
    """
    Advanced robotics-style navigation panel with multiple views:
    - Birds-eye overhead view with obstacles
    - Real-time trajectory path
    - Compass and heading indicator
    - Waypoint/goal markers
    - Distance and bearing to goal
    - Safe zone visualization
    """

    def __init__(self, panel_width: int = 400, panel_height: int = 600):
        """
        Initialize advanced navigation panel.

        Args:
            panel_width: Width of panel in pixels
            panel_height: Height of panel in pixels
        """
        self.width = panel_width
        self.height = panel_height

        # Sub-panel sizes
        self.overhead_size = 300
        self.compass_size = 100

        # Colors (BGR)
        self.BG_COLOR = (20, 20, 20)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (0, 255, 255)
        self.RED = (0, 0, 255)
        self.BLUE = (255, 100, 0)
        self.WHITE = (255, 255, 255)
        self.GRAY = (80, 80, 80)
        self.CYAN = (255, 255, 0)
        self.PURPLE = (255, 0, 255)

        # State tracking
        self.trajectory = []  # List of (x, y) positions
        self.max_trajectory_points = 200
        self.current_position = (0, 0, 0)  # (x, y, z)
        self.current_heading = 0.0  # radians
        self.goal_position = None
        self.waypoints = []
        self.obstacles = []

        # Visualization settings
        self.meters_per_pixel = 0.05  # Map resolution
        self.overhead_range = 5.0  # meters visible in overhead view

        # Performance optimization - only update at 10Hz
        self.last_update_time = 0
        self.update_interval = 0.1  # 100ms = 10Hz

    def update(self, slam_result: Optional[Dict], detections: List[Dict],
               goal: Optional[Tuple[float, float]] = None):
        """
        Update navigation panel data (non-blocking, fast).

        Args:
            slam_result: SLAM tracking result with position/orientation
            detections: Object detections with depth
            goal: Optional goal position (x, y)
        """
        current_time = time.time()

        # Performance: Only update at 10Hz to avoid impacting main loop
        if current_time - self.last_update_time < self.update_interval:
            return

        self.last_update_time = current_time

        # Update position from SLAM
        if slam_result and slam_result.get('position'):
            pos = slam_result['position']
            self.current_position = (pos[0], pos[1], pos[2])

            # Add to trajectory
            self.trajectory.append((pos[0], pos[1]))
            if len(self.trajectory) > self.max_trajectory_points:
                self.trajectory.pop(0)

            # Extract heading from pose matrix if available (FIXED for accuracy)
            if 'pose' in slam_result:
                pose = slam_result['pose']
                # Properly extract yaw from rotation matrix
                # For a 4x4 transformation matrix, rotation is in top-left 3x3
                # Yaw (rotation around Z-axis) is: atan2(R[1,0], R[0,0])
                self.current_heading = np.arctan2(pose[1, 0], pose[0, 0])
            elif len(self.trajectory) > 1:
                # Fallback: Calculate heading from trajectory movement
                dx = self.trajectory[-1][0] - self.trajectory[-2][0]
                dy = self.trajectory[-1][1] - self.trajectory[-2][1]
                if abs(dx) > 0.01 or abs(dy) > 0.01:  # Only update if moved
                    self.current_heading = np.arctan2(dy, dx)

        # Update obstacles from detections
        self.obstacles = []
        for det in detections:
            depth = det.get('depth', 0.0)
            if depth > 0.1 and depth < 5.0:  # Valid range
                # Estimate position based on depth and bbox center
                bbox = det.get('bbox', [0, 0, 0, 0])
                center_x = (bbox[0] + bbox[2]) / 2
                frame_center = 160  # Assuming 320 width

                # Rough angle estimation
                angle_offset = (center_x - frame_center) / frame_center * 0.5  # ~30 deg FOV
                obs_angle = self.current_heading + angle_offset

                # Project obstacle position
                obs_x = self.current_position[0] + depth * np.cos(obs_angle)
                obs_y = self.current_position[1] + depth * np.sin(obs_angle)

                self.obstacles.append({
                    'position': (obs_x, obs_y),
                    'depth': depth,
                    'label': det.get('label', 'object')
                })

        # Update goal
        if goal is not None:
            self.goal_position = goal

    def render(self) -> np.ndarray:
        """
        Render complete navigation panel (optimized for speed).

        Returns:
            Navigation panel image
        """
        # Create panel background
        panel = np.full((self.height, self.width, 3), self.BG_COLOR, dtype=np.uint8)

        # Layout: Overhead view (top), Compass (middle), Info (bottom)
        # FIXED: Add more spacing to prevent overlap
        y_offset = 10

        # 1. Overhead birds-eye view
        overhead = self._render_overhead_view()
        panel[y_offset:y_offset+self.overhead_size,
              (self.width-self.overhead_size)//2:(self.width+self.overhead_size)//2] = overhead
        y_offset += self.overhead_size + 15  # Reduced spacing

        # 2. Compass and heading (smaller for more space)
        compass = self._render_compass()
        compass_x = (self.width - self.compass_size) // 2
        panel[y_offset:y_offset+self.compass_size,
              compass_x:compass_x+self.compass_size] = compass
        y_offset += self.compass_size + 15  # Reduced spacing

        # 3. Navigation info panel (more space for text)
        self._render_info_panel(panel, y_offset)

        # Title
        cv2.putText(panel, "NAV", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.WHITE, 2)

        return panel

    def _render_overhead_view(self) -> np.ndarray:
        """
        Render birds-eye overhead view with obstacles and trajectory.

        Returns:
            Overhead view image
        """
        size = self.overhead_size
        view = np.full((size, size, 3), (30, 30, 30), dtype=np.uint8)
        center = size // 2

        # Draw grid (1 meter spacing)
        grid_spacing = int(1.0 / self.meters_per_pixel)
        for i in range(0, size, grid_spacing):
            cv2.line(view, (i, 0), (i, size), self.GRAY, 1)
            cv2.line(view, (0, i), (size, i), self.GRAY, 1)

        # Draw range circles
        for radius_m in [1.0, 2.0, 3.0]:
            radius_px = int(radius_m / self.meters_per_pixel)
            cv2.circle(view, (center, center), radius_px, (60, 60, 60), 1)

        # Helper function to convert world to view coords
        def world_to_view(wx, wy):
            # Relative to current position
            dx = wx - self.current_position[0]
            dy = wy - self.current_position[1]

            # Rotate by heading (so forward is up)
            cos_h = np.cos(-self.current_heading)
            sin_h = np.sin(-self.current_heading)
            rx = dx * cos_h - dy * sin_h
            ry = dx * sin_h + dy * cos_h

            # To pixels
            px = int(center + rx / self.meters_per_pixel)
            py = int(center - ry / self.meters_per_pixel)  # Flip Y
            return px, py

        # Draw trajectory path
        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory) - 1):
                p1 = world_to_view(self.trajectory[i][0], self.trajectory[i][1])
                p2 = world_to_view(self.trajectory[i+1][0], self.trajectory[i+1][1])

                # Check bounds
                if (0 <= p1[0] < size and 0 <= p1[1] < size and
                    0 <= p2[0] < size and 0 <= p2[1] < size):
                    # Gradient color based on recency
                    alpha = int(255 * (i / len(self.trajectory)))
                    color = (0, alpha, 0)
                    cv2.line(view, p1, p2, color, 2)

        # Draw obstacles
        for obs in self.obstacles:
            px, py = world_to_view(obs['position'][0], obs['position'][1])
            if 0 <= px < size and 0 <= py < size:
                # Color by distance
                depth = obs['depth']
                if depth < 1.0:
                    color = self.RED
                    radius = 8
                elif depth < 2.0:
                    color = self.YELLOW
                    radius = 6
                else:
                    color = self.BLUE
                    radius = 4

                cv2.circle(view, (px, py), radius, color, -1)
                cv2.circle(view, (px, py), radius+2, self.WHITE, 1)

        # Draw goal marker
        if self.goal_position:
            gx, gy = world_to_view(self.goal_position[0], self.goal_position[1])
            if 0 <= gx < size and 0 <= gy < size:
                # Star marker for goal
                cv2.drawMarker(view, (gx, gy), self.PURPLE, cv2.MARKER_STAR,
                             15, 3)
                cv2.putText(view, "GOAL", (gx-20, gy-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.PURPLE, 1)

        # Draw robot (current position) at center
        # Triangle pointing in heading direction
        triangle_pts = np.array([
            [center, center - 15],      # Front point (up)
            [center - 8, center + 10],   # Back left
            [center + 8, center + 10]    # Back right
        ], np.int32)
        cv2.fillPoly(view, [triangle_pts], self.GREEN)
        cv2.polylines(view, [triangle_pts], True, self.WHITE, 2)

        # Draw FOV cone
        fov_angle = 0.5  # radians (~30 degrees each side)
        fov_length = int(3.0 / self.meters_per_pixel)  # 3 meter cone

        left_pt = (int(center + fov_length * np.sin(fov_angle)),
                   int(center - fov_length * np.cos(fov_angle)))
        right_pt = (int(center - fov_length * np.sin(fov_angle)),
                    int(center - fov_length * np.cos(fov_angle)))

        pts = np.array([[center, center], left_pt, right_pt], np.int32)
        overlay = view.copy()
        cv2.fillPoly(overlay, [pts], (0, 100, 0))
        cv2.addWeighted(view, 0.7, overlay, 0.3, 0, view)

        # Labels
        cv2.putText(view, "OVERHEAD VIEW", (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.WHITE, 1)

        return view

    def _render_compass(self) -> np.ndarray:
        """
        Render compass with heading indicator.

        Returns:
            Compass image
        """
        size = self.compass_size
        compass = np.full((size, size, 3), (30, 30, 30), dtype=np.uint8)
        center = (size // 2, size // 2)
        radius = size // 2 - 10

        # Draw compass circle
        cv2.circle(compass, center, radius, self.GRAY, 2)

        # Draw cardinal directions
        directions = [
            ("N", 0, self.WHITE),
            ("E", np.pi/2, self.GRAY),
            ("S", np.pi, self.GRAY),
            ("W", -np.pi/2, self.GRAY)
        ]

        for label, angle, color in directions:
            x = int(center[0] + (radius - 15) * np.sin(angle))
            y = int(center[1] - (radius - 15) * np.cos(angle))
            cv2.putText(compass, label, (x-5, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw heading arrow
        arrow_end_x = int(center[0] + (radius - 5) * np.sin(self.current_heading))
        arrow_end_y = int(center[1] - (radius - 5) * np.cos(self.current_heading))
        cv2.arrowedLine(compass, center, (arrow_end_x, arrow_end_y),
                       self.GREEN, 3, tipLength=0.3)

        # Heading in degrees
        heading_deg = int(np.degrees(self.current_heading) % 360)
        cv2.putText(compass, f"{heading_deg}", (center[0]-15, center[1]+5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.WHITE, 2)

        return compass

    def _render_info_panel(self, panel: np.ndarray, y_start: int):
        """
        Render navigation info text panel with proper spacing (FIXED).

        Args:
            panel: Panel image to draw on
            y_start: Y position to start drawing
        """
        info_x = 10
        y = y_start
        max_y = self.height - 15  # Leave more margin at bottom
        line_spacing = 16  # FIXED: Consistent spacing between lines

        # Helper to check bounds and handle spacing properly
        def safe_text(text, pos_y, color, size=0.35, bold=1):
            if pos_y + line_spacing < max_y:
                cv2.putText(panel, text, (info_x, pos_y),
                           cv2.FONT_HERSHEY_SIMPLEX, size, color, bold)
                return pos_y + line_spacing
            return pos_y

        # Position (compact format)
        x, y_pos, z = self.current_position
        y = safe_text(f"Pos: {x:.1f}, {y_pos:.1f}m", y, self.CYAN, 0.38, 1)
        y += 3

        # Heading
        heading_deg = int(np.degrees(self.current_heading) % 360)
        y = safe_text(f"Heading: {heading_deg}", y, self.CYAN, 0.38, 1)
        y += 3

        # Distance traveled
        distance = sum(np.sqrt((self.trajectory[i+1][0] - self.trajectory[i][0])**2 +
                              (self.trajectory[i+1][1] - self.trajectory[i][1])**2)
                      for i in range(len(self.trajectory)-1)) if len(self.trajectory) > 1 else 0
        y = safe_text(f"Dist: {distance:.1f}m", y, self.CYAN, 0.38, 1)
        y += 5

        # Obstacles detected
        danger_obs = sum(1 for obs in self.obstacles if obs['depth'] < 1.0)
        caution_obs = sum(1 for obs in self.obstacles if 1.0 <= obs['depth'] < 2.0)

        y = safe_text(f"Obstacles:", y, self.CYAN, 0.38, 1)
        y = safe_text(f"  Danger: {danger_obs}", y, self.RED, 0.35, 1)
        y = safe_text(f"  Caution: {caution_obs}", y, self.YELLOW, 0.35, 1)
        y += 5

        # Goal info
        if self.goal_position and y + (line_spacing * 3) < max_y:
            # Calculate distance and bearing to goal
            dx = self.goal_position[0] - self.current_position[0]
            dy = self.goal_position[1] - self.current_position[1]
            dist_to_goal = np.sqrt(dx**2 + dy**2)
            bearing_to_goal = np.arctan2(dy, dx)
            bearing_deg = int(np.degrees(bearing_to_goal) % 360)

            y = safe_text(f"Goal:", y, self.PURPLE, 0.38, 1)
            y = safe_text(f"  {dist_to_goal:.1f}m", y, self.WHITE, 0.35, 1)
            y = safe_text(f"  {bearing_deg}", y, self.WHITE, 0.35, 1)

    def set_goal(self, x: float, y: float):
        """Set navigation goal position."""
        self.goal_position = (x, y)

    def clear_goal(self):
        """Clear navigation goal."""
        self.goal_position = None

    def add_waypoint(self, x: float, y: float):
        """Add waypoint to path."""
        self.waypoints.append((x, y))

    def clear_waypoints(self):
        """Clear all waypoints."""
        self.waypoints = []

    def render_side_by_side(self, slam_map_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render navigation panel side-by-side with SLAM map for unified display.

        Args:
            slam_map_image: Optional SLAM map image to display alongside

        Returns:
            Combined image with SLAM map (left) and navigation info (right)
        """
        # Get nav panel
        nav_panel = self.render()

        if slam_map_image is None:
            # No SLAM map, just return nav panel
            return nav_panel

        # Resize SLAM map to match nav panel height
        slam_h, slam_w = slam_map_image.shape[:2]
        target_height = self.height

        # Maintain aspect ratio
        aspect = slam_w / slam_h
        target_width = int(target_height * aspect)

        slam_resized = cv2.resize(slam_map_image, (target_width, target_height),
                                 interpolation=cv2.INTER_LINEAR)

        # Create combined image (SLAM left, Nav panel right)
        combined_width = target_width + self.width
        combined = np.zeros((target_height, combined_width, 3), dtype=np.uint8)

        # Place SLAM map on left
        combined[:, :target_width] = slam_resized

        # Place nav panel on right
        combined[:, target_width:] = nav_panel

        # Draw separator line
        cv2.line(combined, (target_width, 0), (target_width, target_height),
                (80, 80, 80), 2)

        return combined

    def reset(self):
        """Reset all navigation data."""
        self.trajectory = []
        self.current_position = (0, 0, 0)
        self.current_heading = 0.0
        self.goal_position = None
        self.waypoints = []
        self.obstacles = []
