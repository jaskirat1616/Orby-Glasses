"""
3D Real-time Mapping Module
Visualizes detected objects and depth information as a live 3D point cloud.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
import time

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False


class Mapper3D:
    """
    Real-time 3D mapping using Open3D for visualization.
    Converts depth maps and detections into a live point cloud.

    Note: Uses non-blocking visualization to avoid threading issues on macOS.
    """

    def __init__(self, config):
        """
        Initialize 3D mapper.

        Args:
            config: Configuration manager
        """
        self.config = config

        # Get configuration
        self.enabled = config.get('mapping3d.enabled', False)  # Disabled by default

        if not OPEN3D_AVAILABLE:
            print("⚠️  Open3D not available, 3D mapping disabled")
            self.enabled = False

        self.max_depth = config.get('mapping3d.max_depth', 10.0)
        self.voxel_size = config.get('mapping3d.voxel_size', 0.1)
        self.update_interval = config.get('mapping3d.update_interval', 0.3)
        self.skip_bbox = config.get('mapping3d.skip_bbox', True)

        # Camera intrinsics (approximate for typical webcam)
        self.fx = config.get('mapping3d.fx', 500)  # Focal length x
        self.fy = config.get('mapping3d.fy', 500)  # Focal length y
        self.cx = config.get('camera.width', 640) / 2  # Principal point x
        self.cy = config.get('camera.height', 480) / 2  # Principal point y

        # Visualization
        self.vis = None
        self.pcd = None
        self.coordinate_frame = None
        self.running = False

        # Point cloud data
        self.points = []
        self.colors = []

        # Detection bounding boxes
        self.bbox_line_sets = []

        # Last update time
        self.last_update = 0

        # Geometry update flags
        self.geometry_added = False

        if self.enabled:
            try:
                self.initialize_visualizer()
                print("✓ 3D mapping initialized successfully")
            except Exception as e:
                print(f"⚠️  Failed to initialize 3D mapping: {e}")
                self.enabled = False

    def initialize_visualizer(self):
        """Initialize Open3D visualizer (non-blocking mode)."""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name='OrbyGlasses - 3D Map', width=800, height=600)

        # Create initial point cloud
        self.pcd = o3d.geometry.PointCloud()

        # Add coordinate frame
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0]
        )
        self.vis.add_geometry(self.coordinate_frame)

        # Configure render options
        render_option = self.vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.array([0.1, 0.1, 0.1])

        # Set camera viewpoint
        view_control = self.vis.get_view_control()
        view_control.set_front([0, 0, -1])
        view_control.set_lookat([0, 0, 2])
        view_control.set_up([0, -1, 0])
        view_control.set_zoom(0.5)

        self.running = True

    def depth_to_point_cloud(
        self,
        depth_map: np.ndarray,
        rgb_frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert depth map to 3D point cloud.

        Args:
            depth_map: Normalized depth map (0-1)
            rgb_frame: RGB image

        Returns:
            Tuple of (points, colors) as numpy arrays
        """
        h, w = depth_map.shape

        # Create meshgrid of pixel coordinates
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Convert normalized depth to actual depth in meters
        depth = depth_map * self.max_depth

        # Filter out invalid depths
        valid_mask = (depth > 0) & (depth < self.max_depth)

        # Backproject to 3D
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        # Z = depth
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth

        # Stack coordinates
        points = np.stack([x, y, z], axis=-1)

        # Get colors from RGB image (normalize to 0-1)
        colors = rgb_frame.astype(np.float32) / 255.0

        # Apply valid mask and flatten
        points = points[valid_mask]
        colors = colors[valid_mask]

        # Subsample for performance (every Nth point)
        step = self.config.get('mapping3d.subsample_step', 4)
        points = points[::step]
        colors = colors[::step]

        return points, colors

    def create_detection_bbox(
        self,
        detection: Dict,
        depth_map: np.ndarray
    ) -> Optional['o3d.geometry.LineSet']:
        """
        Create 3D bounding box for a detection.

        Args:
            detection: Detection dictionary with bbox and label
            depth_map: Depth map

        Returns:
            Open3D LineSet representing the bounding box
        """
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox

        # Get average depth in bbox region
        h, w = depth_map.shape
        x1_clip = max(0, min(int(x1), w - 1))
        x2_clip = max(0, min(int(x2), w - 1))
        y1_clip = max(0, min(int(y1), h - 1))
        y2_clip = max(0, min(int(y2), h - 1))

        if x2_clip <= x1_clip or y2_clip <= y1_clip:
            return None

        depth_region = depth_map[y1_clip:y2_clip, x1_clip:x2_clip]
        avg_depth = np.median(depth_region) * self.max_depth

        if avg_depth <= 0 or avg_depth > self.max_depth:
            return None

        # Calculate 3D coordinates of bbox corners
        # Top-left, top-right, bottom-right, bottom-left
        corners_2d = [
            (x1, y1), (x2, y1), (x2, y2), (x1, y2)
        ]

        # Estimate bbox depth (assume 0.5m depth for objects)
        depth_extent = 0.5

        # Create 3D corners (front and back face)
        corners_3d = []
        for (u, v) in corners_2d:
            x_front = (u - self.cx) * avg_depth / self.fx
            y_front = (v - self.cy) * avg_depth / self.fy
            z_front = avg_depth

            x_back = (u - self.cx) * (avg_depth + depth_extent) / self.fx
            y_back = (v - self.cy) * (avg_depth + depth_extent) / self.fy
            z_back = avg_depth + depth_extent

            corners_3d.append([x_front, y_front, z_front])
            corners_3d.append([x_back, y_back, z_back])

        corners_3d = np.array(corners_3d)

        # Define edges of the bounding box
        lines = [
            [0, 2], [2, 4], [4, 6], [6, 0],  # Front face
            [1, 3], [3, 5], [5, 7], [7, 1],  # Back face
            [0, 1], [2, 3], [4, 5], [6, 7],  # Connecting edges
        ]

        # Create LineSet
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners_3d)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Color based on danger level
        if detection.get('is_danger', False):
            color = [1, 0, 0]  # Red for danger
        elif detection.get('depth', 10) < 2.5:
            color = [1, 0.65, 0]  # Orange for caution
        else:
            color = [0, 1, 0]  # Green for safe

        colors = [color for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set

    def update(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray,
        detections: List[Dict]
    ):
        """
        Update the 3D map with new frame data (non-blocking).

        Args:
            frame: RGB frame
            depth_map: Normalized depth map (0-1)
            detections: List of detection dictionaries
        """
        if not self.enabled or not self.running:
            return

        try:
            current_time = time.time()
            if current_time - self.last_update < self.update_interval:
                return

            self.last_update = current_time

            # Convert depth to point cloud
            points, colors = self.depth_to_point_cloud(depth_map, frame)

            # Downsample using voxel grid
            if len(points) > 0:
                # Additional subsampling before voxel grid (for speed)
                if len(points) > 5000:
                    step = len(points) // 5000
                    points = points[::step]
                    colors = colors[::step]

                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(points)
                temp_pcd.colors = o3d.utility.Vector3dVector(colors)

                # Voxel downsampling
                downsampled = temp_pcd.voxel_down_sample(voxel_size=self.voxel_size)

                self.points = np.asarray(downsampled.points)
                self.colors = np.asarray(downsampled.colors)

                # Update point cloud geometry
                self.pcd.points = o3d.utility.Vector3dVector(self.points)
                self.pcd.colors = o3d.utility.Vector3dVector(self.colors)

                # Add or update point cloud geometry
                if not self.geometry_added:
                    self.vis.add_geometry(self.pcd)
                    self.geometry_added = True
                else:
                    self.vis.update_geometry(self.pcd)

                # Only add bounding boxes if enabled (they slow things down significantly)
                if not self.skip_bbox:
                    # Remove old bounding boxes
                    for bbox in self.bbox_line_sets:
                        self.vis.remove_geometry(bbox, reset_bounding_box=False)
                    self.bbox_line_sets = []

                    # Create and add new bounding boxes for detections
                    for detection in detections[:3]:  # Limit to top 3 detections
                        bbox_geom = self.create_detection_bbox(detection, depth_map)
                        if bbox_geom is not None:
                            self.vis.add_geometry(bbox_geom, reset_bounding_box=False)
                            self.bbox_line_sets.append(bbox_geom)

                # Non-blocking update (fast)
                self.vis.poll_events()
                self.vis.update_renderer()

        except Exception as e:
            # Silently handle errors to avoid crashing main loop
            pass

    def stop(self):
        """Stop the visualizer."""
        self.running = False
        if self.vis is not None:
            try:
                self.vis.destroy_window()
            except:
                pass

    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
