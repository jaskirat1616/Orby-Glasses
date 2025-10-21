"""
3D Occupancy Grid Mapping for OrbyGlasses
Real-time volumetric environment representation using depth and SLAM data.
"""

import numpy as np
import cv2
import logging
from typing import Dict, Tuple, Optional, List
from collections import defaultdict
import time


class OccupancyGrid3D:
    """
    3D Occupancy Grid Map for volumetric environment representation.

    Uses a sparse voxel grid to efficiently represent occupied, free, and unknown space.
    Integrates with SLAM for accurate pose estimation and depth maps for obstacle detection.
    """

    def __init__(self, config):
        """
        Initialize 3D Occupancy Grid.

        Args:
            config: ConfigManager instance
        """
        self.config = config
        self.enabled = config.get('occupancy_grid_3d.enabled', True)

        # Grid parameters
        self.grid_size = config.get('occupancy_grid_3d.grid_size', [20.0, 20.0, 3.0])  # [x, y, z] in meters
        self.resolution = config.get('occupancy_grid_3d.resolution', 0.1)  # meters per voxel
        self.origin = np.array([-self.grid_size[0]/2, -self.grid_size[1]/2, 0.0])  # Grid origin in world coords

        # Calculate grid dimensions in voxels
        self.grid_dims = [
            int(self.grid_size[0] / self.resolution),
            int(self.grid_size[1] / self.resolution),
            int(self.grid_size[2] / self.resolution)
        ]

        # Sparse occupancy grid storage (only store occupied and observed voxels)
        # Key: (x, y, z) voxel index, Value: occupancy probability (0.0 = free, 1.0 = occupied)
        self.grid = defaultdict(lambda: 0.5)  # Default: unknown (0.5)

        # Log-odds representation for Bayesian updates
        self.log_odds_occupied = config.get('occupancy_grid_3d.log_odds_occupied', 0.7)
        self.log_odds_free = config.get('occupancy_grid_3d.log_odds_free', -0.4)
        self.log_odds_min = config.get('occupancy_grid_3d.log_odds_min', -5.0)
        self.log_odds_max = config.get('occupancy_grid_3d.log_odds_max', 5.0)

        # Camera intrinsics (from SLAM/config)
        self.fx = config.get('mapping3d.fx', 500)
        self.fy = config.get('mapping3d.fy', 500)
        self.cx = config.get('camera.width', 320) / 2
        self.cy = config.get('camera.height', 320) / 2

        # Ray-casting parameters
        self.max_range = config.get('occupancy_grid_3d.max_range', 5.0)  # Maximum sensor range
        self.min_range = config.get('occupancy_grid_3d.min_range', 0.1)  # Minimum sensor range

        # Visualization settings
        self.visualize = config.get('occupancy_grid_3d.visualize', True)
        self.update_interval = config.get('occupancy_grid_3d.update_interval', 0.5)  # seconds
        self.last_update_time = 0

        # Statistics
        self.total_updates = 0
        self.occupied_voxels = set()
        self.free_voxels = set()

        logging.info(f"3D Occupancy Grid initialized:")
        logging.info(f"  Grid size: {self.grid_size} meters")
        logging.info(f"  Resolution: {self.resolution} m/voxel")
        logging.info(f"  Grid dimensions: {self.grid_dims} voxels")
        logging.info(f"  Total capacity: {self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2]:,} voxels")

    def world_to_voxel(self, point_3d: np.ndarray) -> Tuple[int, int, int]:
        """
        Convert 3D world coordinates to voxel indices.

        Args:
            point_3d: [x, y, z] in world coordinates (meters)

        Returns:
            Tuple of (ix, iy, iz) voxel indices
        """
        relative = point_3d - self.origin
        ix = int(relative[0] / self.resolution)
        iy = int(relative[1] / self.resolution)
        iz = int(relative[2] / self.resolution)
        return (ix, iy, iz)

    def voxel_to_world(self, voxel_idx: Tuple[int, int, int]) -> np.ndarray:
        """
        Convert voxel indices to 3D world coordinates (voxel center).

        Args:
            voxel_idx: (ix, iy, iz) voxel indices

        Returns:
            [x, y, z] world coordinates (meters)
        """
        ix, iy, iz = voxel_idx
        x = self.origin[0] + (ix + 0.5) * self.resolution
        y = self.origin[1] + (iy + 0.5) * self.resolution
        z = self.origin[2] + (iz + 0.5) * self.resolution
        return np.array([x, y, z])

    def is_valid_voxel(self, voxel_idx: Tuple[int, int, int]) -> bool:
        """Check if voxel index is within grid bounds."""
        ix, iy, iz = voxel_idx
        return (0 <= ix < self.grid_dims[0] and
                0 <= iy < self.grid_dims[1] and
                0 <= iz < self.grid_dims[2])

    def update_from_depth(self, depth_map: np.ndarray, camera_pose: np.ndarray):
        """
        Update occupancy grid from depth map and camera pose.

        Uses ray-casting: mark voxels along ray as free, endpoint as occupied.

        Args:
            depth_map: Depth map (H x W) normalized 0-1
            camera_pose: 4x4 camera pose matrix (world coordinates)
        """
        if not self.enabled:
            return

        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return  # Skip update to maintain performance

        h, w = depth_map.shape
        camera_position = camera_pose[:3, 3]

        # Subsample depth map for performance (every Nth pixel)
        subsample = self.config.get('occupancy_grid_3d.subsample_step', 8)

        rays_cast = 0
        voxels_updated = 0

        for v in range(0, h, subsample):
            for u in range(0, w, subsample):
                depth_normalized = depth_map[v, u]

                # Convert normalized depth to metric depth
                depth = depth_normalized * self.max_range

                # Skip invalid depths
                if depth < self.min_range or depth > self.max_range:
                    continue

                # Back-project pixel to 3D point in camera frame
                x_cam = (u - self.cx) * depth / self.fx
                y_cam = (v - self.cy) * depth / self.fy
                z_cam = depth

                # Transform to world frame
                point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
                point_world = (camera_pose @ point_cam)[:3]

                # Ray-cast from camera to point
                updated = self._ray_cast_update(camera_position, point_world)
                voxels_updated += updated
                rays_cast += 1

        self.total_updates += 1
        self.last_update_time = current_time

        logging.debug(f"Occupancy grid updated: {rays_cast} rays, {voxels_updated} voxels modified")

    def _ray_cast_update(self, start: np.ndarray, end: np.ndarray) -> int:
        """
        Cast ray from start to end point, updating voxels along the way.

        Args:
            start: Ray start point [x, y, z] (camera position)
            end: Ray end point [x, y, z] (observed obstacle)

        Returns:
            Number of voxels updated
        """
        # Bresenham's line algorithm in 3D (voxel traversal)
        start_voxel = self.world_to_voxel(start)
        end_voxel = self.world_to_voxel(end)

        # Get all voxels along ray
        ray_voxels = self._bresenham_3d(start_voxel, end_voxel)

        updates = 0

        # Mark all voxels except last as free
        for voxel in ray_voxels[:-1]:
            if self.is_valid_voxel(voxel):
                self._update_voxel(voxel, occupied=False)
                updates += 1

        # Mark last voxel as occupied (obstacle endpoint)
        if ray_voxels and self.is_valid_voxel(ray_voxels[-1]):
            self._update_voxel(ray_voxels[-1], occupied=True)
            updates += 1

        return updates

    def _update_voxel(self, voxel_idx: Tuple[int, int, int], occupied: bool):
        """
        Update voxel occupancy using log-odds Bayesian update.

        Args:
            voxel_idx: Voxel index (ix, iy, iz)
            occupied: True if obstacle detected, False if free space
        """
        # Get current log-odds value (initialize to 0 = unknown)
        current_log_odds = self.grid.get(voxel_idx, 0.0)

        # Bayesian update
        if occupied:
            new_log_odds = current_log_odds + self.log_odds_occupied
            self.occupied_voxels.add(voxel_idx)
            self.free_voxels.discard(voxel_idx)
        else:
            new_log_odds = current_log_odds + self.log_odds_free
            self.free_voxels.add(voxel_idx)
            self.occupied_voxels.discard(voxel_idx)

        # Clamp to prevent overflow
        new_log_odds = np.clip(new_log_odds, self.log_odds_min, self.log_odds_max)

        # Store updated value
        self.grid[voxel_idx] = new_log_odds

    def _bresenham_3d(self, start: Tuple[int, int, int], end: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        """
        3D Bresenham's line algorithm for voxel traversal.

        Args:
            start: Start voxel (ix, iy, iz)
            end: End voxel (ix, iy, iz)

        Returns:
            List of voxel indices along the ray
        """
        x0, y0, z0 = start
        x1, y1, z1 = end

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dz = abs(z1 - z0)

        xs = 1 if x1 > x0 else -1
        ys = 1 if y1 > y0 else -1
        zs = 1 if z1 > z0 else -1

        voxels = []

        # Driving axis
        if dx >= dy and dx >= dz:
            p1 = 2 * dy - dx
            p2 = 2 * dz - dx
            while x0 != x1:
                voxels.append((x0, y0, z0))
                x0 += xs
                if p1 >= 0:
                    y0 += ys
                    p1 -= 2 * dx
                if p2 >= 0:
                    z0 += zs
                    p2 -= 2 * dx
                p1 += 2 * dy
                p2 += 2 * dz
        elif dy >= dx and dy >= dz:
            p1 = 2 * dx - dy
            p2 = 2 * dz - dy
            while y0 != y1:
                voxels.append((x0, y0, z0))
                y0 += ys
                if p1 >= 0:
                    x0 += xs
                    p1 -= 2 * dy
                if p2 >= 0:
                    z0 += zs
                    p2 -= 2 * dy
                p1 += 2 * dx
                p2 += 2 * dz
        else:
            p1 = 2 * dy - dz
            p2 = 2 * dx - dz
            while z0 != z1:
                voxels.append((x0, y0, z0))
                z0 += zs
                if p1 >= 0:
                    y0 += ys
                    p1 -= 2 * dz
                if p2 >= 0:
                    x0 += xs
                    p2 -= 2 * dz
                p1 += 2 * dy
                p2 += 2 * dx

        voxels.append((x0, y0, z0))  # Add final point
        return voxels

    def is_occupied(self, point_3d: np.ndarray, threshold: float = 0.0) -> bool:
        """
        Check if a 3D point is occupied.

        Args:
            point_3d: [x, y, z] world coordinates
            threshold: Log-odds threshold for occupancy (default: 0 = 50% probability)

        Returns:
            True if occupied, False otherwise
        """
        voxel = self.world_to_voxel(point_3d)
        if not self.is_valid_voxel(voxel):
            return False

        log_odds = self.grid.get(voxel, 0.0)
        return log_odds > threshold

    def get_occupancy_probability(self, point_3d: np.ndarray) -> float:
        """
        Get occupancy probability at a 3D point.

        Args:
            point_3d: [x, y, z] world coordinates

        Returns:
            Probability of occupancy (0.0 = free, 1.0 = occupied, 0.5 = unknown)
        """
        voxel = self.world_to_voxel(point_3d)
        if not self.is_valid_voxel(voxel):
            return 0.5  # Unknown

        log_odds = self.grid.get(voxel, 0.0)
        # Convert log-odds to probability: p = 1 / (1 + exp(-log_odds))
        probability = 1.0 / (1.0 + np.exp(-log_odds))
        return probability

    def get_occupied_voxels(self, threshold: float = 0.0) -> np.ndarray:
        """
        Get all occupied voxel positions in world coordinates.

        Args:
            threshold: Log-odds threshold for occupancy

        Returns:
            Nx3 array of occupied voxel centers in world coordinates
        """
        occupied_positions = []

        for voxel_idx, log_odds in self.grid.items():
            if log_odds > threshold:
                world_pos = self.voxel_to_world(voxel_idx)
                occupied_positions.append(world_pos)

        if not occupied_positions:
            return np.zeros((0, 3))

        return np.array(occupied_positions)

    def get_2d_slice(self, z_height: float = 0.5, threshold: float = 0.0) -> np.ndarray:
        """
        Get 2D occupancy grid slice at specified height (for visualization/planning).

        Args:
            z_height: Height in meters to extract slice
            threshold: Log-odds threshold for occupancy

        Returns:
            2D occupancy grid (H x W) where 1 = occupied, 0 = free, 0.5 = unknown
        """
        iz = int((z_height - self.origin[2]) / self.resolution)

        if not (0 <= iz < self.grid_dims[2]):
            return np.ones((self.grid_dims[1], self.grid_dims[0])) * 0.5

        slice_grid = np.ones((self.grid_dims[1], self.grid_dims[0])) * 0.5

        for (ix, iy, voxel_iz), log_odds in self.grid.items():
            if voxel_iz == iz:
                # Convert log-odds to probability for visualization
                prob = 1.0 / (1.0 + np.exp(-log_odds))
                slice_grid[iy, ix] = prob

        return slice_grid

    def visualize_3d(self) -> Optional[np.ndarray]:
        """
        Create 3D visualization of occupancy grid.

        Returns:
            Visualization image (if using matplotlib), None otherwise
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            # Get occupied voxels
            occupied = self.get_occupied_voxels(threshold=0.5)

            if len(occupied) == 0:
                logging.warning("No occupied voxels to visualize")
                return None

            # Create figure
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot occupied voxels
            ax.scatter(occupied[:, 0], occupied[:, 1], occupied[:, 2],
                      c='red', marker='s', s=20, alpha=0.6, label='Occupied')

            # Set labels and limits
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'3D Occupancy Grid ({len(occupied)} occupied voxels)')
            ax.legend()

            # Set equal aspect ratio
            max_range = np.array([occupied[:, 0].max()-occupied[:, 0].min(),
                                 occupied[:, 1].max()-occupied[:, 1].min(),
                                 occupied[:, 2].max()-occupied[:, 2].min()]).max() / 2.0

            mid_x = (occupied[:, 0].max()+occupied[:, 0].min()) * 0.5
            mid_y = (occupied[:, 1].max()+occupied[:, 1].min()) * 0.5
            mid_z = (occupied[:, 2].max()+occupied[:, 2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            # Convert to image
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            plt.close(fig)
            return img

        except ImportError:
            logging.warning("Matplotlib not available for 3D visualization")
            return None

    def visualize_2d_slice(self, z_height: float = 1.5) -> np.ndarray:
        """
        Visualize 2D occupancy slice at head height.

        Args:
            z_height: Height to extract slice (default: 1.5m = head height)

        Returns:
            Colored visualization image
        """
        slice_grid = self.get_2d_slice(z_height)

        # Convert probabilities to colors (grayscale)
        vis_img = (slice_grid * 255).astype(np.uint8)

        # Apply colormap: blue = free, red = occupied, gray = unknown
        vis_colored = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)

        # Resize for better visibility
        vis_colored = cv2.resize(vis_colored, (400, 400), interpolation=cv2.INTER_NEAREST)

        # Add text overlay
        cv2.putText(vis_colored, f"Occupancy Grid @ z={z_height:.1f}m",
                   (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(vis_colored, f"Updates: {self.total_updates}",
                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis_colored, f"Occupied: {len(self.occupied_voxels)}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis_colored

    def clear(self):
        """Clear the occupancy grid."""
        self.grid.clear()
        self.occupied_voxels.clear()
        self.free_voxels.clear()
        self.total_updates = 0
        logging.info("Occupancy grid cleared")

    def get_stats(self) -> Dict:
        """Get occupancy grid statistics."""
        return {
            'total_voxels_stored': len(self.grid),
            'occupied_voxels': len(self.occupied_voxels),
            'free_voxels': len(self.free_voxels),
            'total_updates': self.total_updates,
            'grid_size_meters': self.grid_size,
            'resolution': self.resolution,
            'memory_usage_mb': len(self.grid) * 16 / (1024 * 1024)  # Rough estimate
        }
