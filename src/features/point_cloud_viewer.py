"""
Real-time 3D Point Cloud Viewer for OrbyGlasses
Creates dense, colorful 3D visualization like RGB-D SLAM systems
"""

import numpy as np
import cv2
import logging
from typing import Optional, Tuple, Dict
import time
from collections import deque


class PointCloudViewer:
    """
    Real-time 3D point cloud visualization with color.
    Creates dense point clouds from RGB-D data and SLAM poses.
    """

    def __init__(self, config):
        """
        Initialize point cloud viewer.

        Args:
            config: ConfigManager instance
        """
        self.config = config
        self.enabled = config.get('point_cloud_viewer.enabled', True)

        # Camera intrinsics
        self.fx = config.get('mapping3d.fx', 500)
        self.fy = config.get('mapping3d.fy', 500)
        self.cx = config.get('camera.width', 320) / 2
        self.cy = config.get('camera.height', 320) / 2

        # Point cloud storage
        self.points = []  # List of 3D points [x, y, z]
        self.colors = []  # List of RGB colors [r, g, b]
        self.max_points = config.get('point_cloud_viewer.max_points', 100000)

        # Subsampling for performance
        self.subsample = config.get('point_cloud_viewer.subsample', 4)

        # Depth range
        self.min_depth = config.get('point_cloud_viewer.min_depth', 0.1)
        self.max_depth = config.get('point_cloud_viewer.max_depth', 5.0)

        # Visualization state
        self.view_angle_x = 30.0
        self.view_angle_y = 0.0
        self.view_angle_z = 45.0
        self.view_scale = 50.0
        self.view_offset_x = 0.0
        self.view_offset_y = 0.0

        # Mouse control state
        self.mouse_is_pressed = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # Update control
        self.last_update = 0
        self.update_interval = config.get('point_cloud_viewer.update_interval', 0.1)

        # Statistics
        self.total_points_added = 0
        self.frame_count = 0

        logging.info("Point Cloud Viewer initialized")
        logging.info(f"  Max points: {self.max_points:,}")
        logging.info(f"  Subsample: every {self.subsample} pixels")

    def handle_mouse_events(self, event, x, y, flags, param):
        """Handle mouse events for interactive control."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_is_pressed = True
            self.last_mouse_x = x
            self.last_mouse_y = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_is_pressed = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mouse_is_pressed:
                dx = x - self.last_mouse_x
                dy = y - self.last_mouse_y

                if flags & cv2.EVENT_FLAG_SHIFTKEY:
                    # Pan
                    self.view_offset_x += dx * 0.1
                    self.view_offset_y += dy * 0.1
                else:
                    # Rotate
                    self.view_angle_z += dx * 0.5
                    self.view_angle_x += dy * 0.5

                self.last_mouse_x = x
                self.last_mouse_y = y
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.handle_mouse_wheel(1)
            else:
                self.handle_mouse_wheel(-1)

    def add_frame(self, rgb_frame: np.ndarray, depth_map: np.ndarray,
                  camera_pose: Optional[np.ndarray] = None):
        """
        Add a new RGB-D frame to the point cloud.

        Args:
            rgb_frame: RGB image (H x W x 3)
            depth_map: Depth map (H x W), normalized 0-1
            camera_pose: 4x4 camera pose matrix (optional)
        """
        if not self.enabled:
            return

        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return

        if camera_pose is None:
            camera_pose = np.eye(4)

        h, w = depth_map.shape

        # Subsample points for performance
        new_points = []
        new_colors = []

        for v in range(0, h, self.subsample):
            for u in range(0, w, self.subsample):
                # Get depth (convert from normalized to meters)
                depth_norm = depth_map[v, u]
                depth = depth_norm * self.max_depth

                # Skip invalid depths
                if depth < self.min_depth or depth > self.max_depth:
                    continue

                # Get color (BGR to RGB)
                color = rgb_frame[v, u]
                rgb = [int(color[2]), int(color[1]), int(color[0])]

                # Back-project to 3D (camera coordinates)
                x_cam = (u - self.cx) * depth / self.fx
                y_cam = (v - self.cy) * depth / self.fy
                z_cam = depth

                # Transform to world coordinates
                point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
                point_world = camera_pose @ point_cam

                new_points.append(point_world[:3].tolist())
                new_colors.append(rgb)

        # Add to point cloud
        if new_points:
            self.points.extend(new_points)
            self.colors.extend(new_colors)
            self.total_points_added += len(new_points)

            # Limit total points (keep most recent)
            if len(self.points) > self.max_points:
                excess = len(self.points) - self.max_points
                self.points = self.points[excess:]
                self.colors = self.colors[excess:]

        self.last_update = current_time
        self.frame_count += 1

        if self.frame_count % 10 == 0:
            logging.debug(f"Point cloud: {len(self.points):,} points")

    def visualize(self, camera_position: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create visualization of the point cloud.

        Args:
            camera_position: Current camera position [x, y, z]

        Returns:
            Visualization image
        """
        img_size = 800
        canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

        if not self.points:
            # No points yet
            cv2.putText(canvas, "Building 3D Point Cloud...",
                       (img_size//2 - 180, img_size//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            cv2.putText(canvas, "Move camera slowly with good lighting",
                       (img_size//2 - 250, img_size//2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
            return canvas

        # Convert to numpy arrays
        points_array = np.array(self.points)
        colors_array = np.array(self.colors)

        # Project 3D points to 2D
        def project_point(point):
            x, y, z = point

            # Apply rotations
            angle_x = np.radians(self.view_angle_x)
            angle_y = np.radians(self.view_angle_y)
            angle_z = np.radians(self.view_angle_z)

            # Rotate around Z (yaw)
            x_rot = x * np.cos(angle_z) - y * np.sin(angle_z)
            y_rot = x * np.sin(angle_z) + y * np.cos(angle_z)
            z_rot = z

            # Rotate around Y (pitch)
            x_rot2 = x_rot * np.cos(angle_y) + z_rot * np.sin(angle_y)
            z_rot2 = -x_rot * np.sin(angle_y) + z_rot * np.cos(angle_y)

            # Rotate around X (roll/tilt)
            y_final = y_rot * np.cos(angle_x) - z_rot2 * np.sin(angle_x)
            z_final = y_rot * np.sin(angle_x) + z_rot2 * np.cos(angle_x)

            # Project to screen
            screen_x = int((x_rot2 + self.view_offset_x) * self.view_scale + img_size / 2)
            screen_y = int((-y_final + self.view_offset_y) * self.view_scale + img_size / 2)

            return screen_x, screen_y, z_final

        # Project all points and sort by depth
        projected = []
        for i, point in enumerate(points_array):
            sx, sy, depth = project_point(point)
            if 0 <= sx < img_size and 0 <= sy < img_size:
                projected.append((sx, sy, depth, colors_array[i]))

        # Sort by depth (back to front)
        projected.sort(key=lambda x: x[2])

        # Draw points
        point_size = max(1, int(self.view_scale / 30))
        for sx, sy, depth, color in projected:
            # Convert RGB to BGR for OpenCV
            bgr_color = (int(color[2]), int(color[1]), int(color[0]))
            cv2.circle(canvas, (sx, sy), point_size, bgr_color, -1)

        # Draw camera position if provided
        if camera_position is not None:
            sx, sy, _ = project_point(camera_position)
            if 0 <= sx < img_size and 0 <= sy < img_size:
                cv2.circle(canvas, (sx, sy), 8, (0, 165, 255), -1)  # Orange
                cv2.circle(canvas, (sx, sy), 10, (0, 0, 0), 2)
                cv2.putText(canvas, "CAMERA", (sx + 12, sy - 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Add info overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 10), (300, 140), (240, 240, 240), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (10, 10), (300, 140), (100, 100, 100), 2)

        cv2.putText(canvas, "3D Point Cloud", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(canvas, f"Points: {len(self.points):,}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 200), 1)
        cv2.putText(canvas, f"Total Added: {self.total_points_added:,}", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(canvas, f"Zoom: {self.view_scale:.1f}x", (20, 115),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        return canvas

    def update_view_controls(self, key: int) -> bool:
        """Update view based on keyboard input."""
        updated = False

        # Zoom
        if key == ord('+') or key == ord('='):
            self.view_scale *= 1.2
            updated = True
        elif key == ord('-') or key == ord('_'):
            self.view_scale *= 0.8
            updated = True

        # Pan
        elif key == 82:  # Up
            self.view_offset_y += 0.5
            updated = True
        elif key == 84:  # Down
            self.view_offset_y -= 0.5
            updated = True
        elif key == 81:  # Left
            self.view_offset_x -= 0.5
            updated = True
        elif key == 83:  # Right
            self.view_offset_x += 0.5
            updated = True

        # Rotate Z
        elif key == ord('q') or key == ord('Q'):
            self.view_angle_z -= 5.0
            updated = True
        elif key == ord('e') or key == ord('E'):
            self.view_angle_z += 5.0
            updated = True

        # Tilt X
        elif key == ord('w') or key == ord('W'):
            self.view_angle_x += 5.0
            updated = True
        elif key == ord('s') or key == ord('S'):
            self.view_angle_x -= 5.0
            updated = True

        # Rotate Y
        elif key == ord('a') or key == ord('A'):
            self.view_angle_y -= 5.0
            updated = True
        elif key == ord('d') or key == ord('D'):
            self.view_angle_y += 5.0
            updated = True

        # Reset
        elif key == ord('r') or key == ord('R'):
            self.view_angle_x = 30.0
            self.view_angle_y = 0.0
            self.view_angle_z = 45.0
            self.view_scale = 50.0
            self.view_offset_x = 0.0
            self.view_offset_y = 0.0
            updated = True

        # Clear
        elif key == ord('c') or key == ord('C'):
            self.clear()
            updated = True

        return updated

    def handle_mouse_wheel(self, delta: int):
        """Handle mouse wheel for zooming."""
        if delta > 0:
            self.view_scale *= 1.1
        else:
            self.view_scale *= 0.9
        self.view_scale = np.clip(self.view_scale, 10.0, 200.0)

    def clear(self):
        """Clear all points from the cloud."""
        self.points = []
        self.colors = []
        logging.info("Point cloud cleared")

    def get_stats(self) -> Dict:
        """Get statistics about the point cloud."""
        return {
            'num_points': len(self.points),
            'total_added': self.total_points_added,
            'frames_processed': self.frame_count,
            'max_capacity': self.max_points,
            'usage_percent': len(self.points) / self.max_points * 100 if self.max_points > 0 else 0
        }
