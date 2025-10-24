"""
3D Occupancy Grid Mapping for OrbyGlasses
Real-time volumetric environment representation using depth and SLAM data.
Accuracy with ray-casting and sensor fusion.
"""

import numpy as np
import cv2
import logging
from typing import Dict, Tuple, Optional, List
from collections import defaultdict, deque
import time


class OccupancyGrid3D:
    """
    Simplified 3D Occupancy Grid Map for precise volumetric environment representation.

    Uses a sparse voxel grid to efficiently represent occupied, free, and unknown space.
    Integrates with SLAM for pose estimation and depth maps for obstacle detection.
    Includes temporal consistency and sensor fusion for better accuracy.
    """

    def __init__(self, config):
        """
        Initialize 3D Occupancy Grid with accuracy.

        Args:
            config: ConfigManager instance
        """
        self.config = config
        self.enabled = config.get('occupancy_grid_3d.enabled', True)

        # Grid parameters for high accuracy
        self.grid_size = config.get('occupancy_grid_3d.grid_size', [20.0, 20.0, 3.0])  # [x, y, z] in meters
        self.resolution = config.get('occupancy_grid_3d.resolution', 0.1)  # meters per voxel (high res for accuracy)
        self.origin = np.array([-self.grid_size[0]/2, -self.grid_size[1]/2, 0.0])  # Grid origin in world coords

        # Calculate grid dimensions in voxels
        self.grid_dims = [
            int(self.grid_size[0] / self.resolution),
            int(self.grid_size[1] / self.resolution),
            int(self.grid_size[2] / self.resolution)
        ]

        # Sparse occupancy grid storage (only store occupied and observed voxels)
        # Key: (x, y, z) voxel index, Value: log-odds value
        self.grid = defaultdict(lambda: 0.0)  # Default: unknown (0.0 log-odds = 0.5 probability)

        # High-accuracy log-odds parameters for Bayesian updates
        self.log_odds_occupied = config.get('occupancy_grid_3d.log_odds_occupied', 0.8)  # Stronger evidence for occupied
        self.log_odds_free = config.get('occupancy_grid_3d.log_odds_free', -0.6)  # Stronger evidence for free
        self.log_odds_min = config.get('occupancy_grid_3d.log_odds_min', -6.0)  # More extreme bounds
        self.log_odds_max = config.get('occupancy_grid_3d.log_odds_max', 6.0)  # More extreme bounds

        # Sensor-specific parameters for accuracy
        self.depth_uncertainty = config.get('occupancy_grid_3d.depth_uncertainty', 0.05)  # 5cm uncertainty
        self.camera_fov_horizontal = config.get('occupancy_grid_3d.camera_fov_horizontal', 60.0)  # degrees
        self.camera_fov_vertical = config.get('occupancy_grid_3d.camera_fov_vertical', 45.0)  # degrees

        # Camera intrinsics (from SLAM/config)
        self.fx = config.get('mapping3d.fx', 500)
        self.fy = config.get('mapping3d.fy', 500)
        self.cx = config.get('camera.width', 320) / 2
        self.cy = config.get('camera.height', 320) / 2

        # Ray-casting parameters for better accuracy
        self.max_range = config.get('occupancy_grid_3d.max_range', 5.0)  # Maximum sensor range
        self.min_range = config.get('occupancy_grid_3d.min_range', 0.1)  # Minimum sensor range
        self.range_variance = config.get('occupancy_grid_3d.range_variance', 0.02)  # For probabilistic modeling

        # Accuracy and filtering enhancements
        self.temporal_consistency = config.get('occupancy_grid_3d.temporal_consistency', True)
        self.min_observations = config.get('occupancy_grid_3d.min_observations', 2)  # Min observations to confirm
        self.confirmation_threshold = config.get('occupancy_grid_3d.confirmation_threshold', 1.0)
        self.decay_rate = config.get('occupancy_grid_3d.decay_rate', 0.001)  # Slow decay for consistency

        # Multi-resolution mapping for different distance accuracy
        self.near_resolution = config.get('occupancy_grid_3d.near_resolution', 0.05)  # Higher res near camera
        self.near_distance = config.get('occupancy_grid_3d.near_distance', 1.0)  # Distance threshold for high res
        self.far_resolution = config.get('occupancy_grid_3d.far_resolution', 0.2)  # Lower res further away

        # Visualization settings
        self.visualize = config.get('occupancy_grid_3d.visualize', True)
        self.update_interval = config.get('occupancy_grid_3d.update_interval', 0.1)  # Faster updates for accuracy
        self.last_update_time = 0

        # Temporal filtering and history
        self.temporal_filtering = config.get('occupancy_grid_3d.temporal_filtering', True)
        self.observation_history = defaultdict(list)  # Track observations over time
        self.max_history_length = config.get('occupancy_grid_3d.max_history_length', 10)
        self.history_decay = config.get('occupancy_grid_3d.history_decay', 0.95)

        # Statistics and performance
        self.total_updates = 0
        self.occupied_voxels = set()
        self.free_voxels = set()
        self.frame_count = 0
        self.total_voxels_updated = 0
        self.last_positions = deque(maxlen=10)  # Track camera positions for temporal consistency

        # Ray casting optimization
        self.use_3d_bresenham = True  # Use 3D Bresenham algorithm
        self.max_voxels_per_ray = config.get('occupancy_grid_3d.max_voxels_per_ray', 200)  # Limit per ray

        # Mouse control state
        self.mouse_is_pressed = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        logging.info(f"Simplified 3D Occupancy Grid initialized:")
        logging.info(f"  Grid size: {self.grid_size} meters")
        logging.info(f"  Resolution: {self.resolution} m/voxel")
        logging.info(f"  Grid dimensions: {self.grid_dims} voxels")
        logging.info(f"  Total capacity: {self.grid_dims[0] * self.grid_dims[1] * self.grid_dims[2]:,} voxels")
        logging.info(f"  Temporal consistency: {self.temporal_consistency}")
        logging.info(f"  Minimum observations: {self.min_observations}")
        logging.info(f"  Update interval: {self.update_interval}s")

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

    def world_to_voxel(self, point_3d: np.ndarray, camera_position: Optional[np.ndarray] = None) -> Tuple[int, int, int]:
        """
        Convert 3D world coordinates to voxel indices with multi-resolution support.

        Args:
            point_3d: [x, y, z] in world coordinates (meters)
            camera_position: Optional camera position for variable resolution

        Returns:
            Tuple of (ix, iy, iz) voxel indices
        """
        relative = point_3d - self.origin
        
        # Determine resolution based on distance from camera if provided
        if camera_position is not None:
            distance = np.linalg.norm(point_3d - camera_position)
            if distance <= self.near_distance:
                resolution = self.near_resolution
            else:
                resolution = self.far_resolution
        else:
            resolution = self.resolution  # Default resolution
            
        ix = int(relative[0] / resolution)
        iy = int(relative[1] / resolution)
        iz = int(relative[2] / resolution)
        return (ix, iy, iz)

    def voxel_to_world(self, voxel_idx: Tuple[int, int, int], resolution: Optional[float] = None) -> np.ndarray:
        """
        Convert voxel indices to 3D world coordinates (voxel center).

        Args:
            voxel_idx: (ix, iy, iz) voxel indices
            resolution: Optional resolution to use for conversion

        Returns:
            [x, y, z] world coordinates (meters)
        """
        ix, iy, iz = voxel_idx
        res = resolution if resolution is not None else self.resolution
        x = self.origin[0] + (ix + 0.5) * res
        y = self.origin[1] + (iy + 0.5) * res
        z = self.origin[2] + (iz + 0.5) * res
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

        Uses ray-casting with uncertainty modeling and temporal filtering.

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

        # Add current position to history for temporal consistency
        self.last_positions.append(camera_position.copy())

        # Process depth map
        subsample = self.config.get('occupancy_grid_3d.subsample_step', 2)  # More dense sampling for accuracy

        rays_cast = 0
        voxels_updated = 0

        # Process depth map with uncertainty modeling
        for v in range(0, h, subsample):
            for u in range(0, w, subsample):
                depth_normalized = depth_map[v, u]

                # Convert normalized depth to metric depth
                depth = depth_normalized * self.max_range

                # Skip invalid depths
                if depth < self.min_range or depth > self.max_range:
                    continue

                # Calculate 3D point in camera frame
                x_cam = (u - self.cx) * depth / self.fx
                y_cam = (v - self.cy) * depth / self.fy
                z_cam = depth

                # Transform to world frame
                point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
                point_world = (camera_pose @ point_cam)[:3]

                # Calculate uncertainty based on depth and pixel location
                depth_uncertainty = self._calculate_depth_uncertainty(depth, u, v, w, h)
                
                # Perform ray casting with uncertainty
                ray_voxels = self._ray_cast_with_uncertainty(
                    camera_position, point_world, depth_uncertainty
                )

                # Update voxels along the ray
                for i, voxel in enumerate(ray_voxels):
                    if self.is_valid_voxel(voxel):
                        # Mark as free (except the endpoint)
                        is_endpoint = (i == len(ray_voxels) - 1)
                        self._update_voxel_with_temporal_filtering(
                            voxel, occupied=is_endpoint, uncertainty=depth_uncertainty
                        )
                        voxels_updated += 1
                        rays_cast += 1

        # Apply temporal consistency decay
        if self.temporal_consistency and self.decay_rate > 0:
            self._apply_temporal_decay()

        self.total_updates += 1
        self.frame_count += 1
        self.last_update_time = current_time

        if rays_cast > 0 and self.frame_count % 10 == 0:
            logging.info(f"Grid: {rays_cast} rays, {len(self.occupied_voxels)} occupied, {len(self.free_voxels)} free, updates: {self.total_updates}")

    def _calculate_depth_uncertainty(self, depth: float, u: int, v: int, w: int, h: int) -> float:
        """
        Calculate depth uncertainty based on depth value and pixel location.

        Args:
            depth: Measured depth
            u, v: Pixel coordinates
            w, h: Image dimensions

        Returns:
            Uncertainty value
        """
        # Base uncertainty from sensor
        base_uncertainty = self.depth_uncertainty
        
        # Increase uncertainty with depth (sensor noise increases)
        depth_factor = depth / self.max_range
        depth_uncertainty = base_uncertainty * (1.0 + 0.5 * depth_factor)
        
        # Increase uncertainty at image edges (less precise)
        center_u, center_v = w / 2, h / 2
        pixel_distance = np.sqrt((u - center_u)**2 + (v - center_v)**2)
        max_distance = np.sqrt(center_u**2 + center_v**2)
        edge_factor = pixel_distance / max_distance
        edge_uncertainty = base_uncertainty * 0.3 * edge_factor
        
        return depth_uncertainty + depth_uncertainty + edge_uncertainty

    def _ray_cast_with_uncertainty(self, start: np.ndarray, end: np.ndarray, uncertainty: float) -> List[Tuple[int, int, int]]:
        """
        Perform ray casting with uncertainty modeling.

        Args:
            start: Ray start point [x, y, z] (camera position)
            end: Ray end point [x, y, z] (observed obstacle)
            uncertainty: Depth uncertainty value

        Returns:
            List of voxel indices along the ray
        """
        # Convert to voxel coordinates
        start_voxel = self.world_to_voxel(start)
        end_voxel = self.world_to_voxel(end)

        # Get all voxels along ray using 3D Bresenham
        ray_voxels = self._bresenham_3d(start_voxel, end_voxel)

        # Apply uncertainty by including neighboring voxels
        if uncertainty > self.resolution * 1.5:  # Only if uncertainty is significant
            expanded_voxels = []
            for voxel in ray_voxels:
                # Add the main voxel
                expanded_voxels.append(voxel)
                
                # Add neighboring voxels within uncertainty range
                uncertainty_voxels = int(uncertainty / self.resolution)
                for dx in range(-uncertainty_voxels, uncertainty_voxels + 1):
                    for dy in range(-uncertainty_voxels, uncertainty_voxels + 1):
                        for dz in range(-uncertainty_voxels, uncertainty_voxels + 1):
                            neighbor = (voxel[0] + dx, voxel[1] + dy, voxel[2] + dz)
                            if neighbor not in expanded_voxels and self.is_valid_voxel(neighbor):
                                expanded_voxels.append(neighbor)
            
            return expanded_voxels
        else:
            return ray_voxels

    def _update_voxel_with_temporal_filtering(self, voxel_idx: Tuple[int, int, int], occupied: bool, uncertainty: float = 0.0):
        """
        Update voxel with temporal filtering and uncertainty modeling.

        Args:
            voxel_idx: Voxel index (ix, iy, iz)
            occupied: True if obstacle detected, False if free space
            uncertainty: Uncertainty value for the measurement
        """
        # Get current log-odds value
        current_log_odds = self.grid.get(voxel_idx, 0.0)

        # Apply uncertainty weighting to the update
        weight = 1.0 - min(1.0, uncertainty / (2 * self.resolution))  # Normalize uncertainty
        if occupied:
            update_value = self.log_odds_occupied * weight
        else:
            update_value = self.log_odds_free * weight

        # Apply temporal filtering if enabled
        if self.temporal_filtering:
            # Add to history
            self.observation_history[voxel_idx].append((update_value, time.time()))
            
            # Keep only recent observations
            current_time = time.time()
            recent_obs = [
                obs for obs in self.observation_history[voxel_idx]
                if current_time - obs[1] < 2.0  # Keep last 2 seconds of observations
            ]
            self.observation_history[voxel_idx] = recent_obs
            
            # Compute weighted average of recent observations
            if len(recent_obs) >= self.min_observations:
                # Weight more recent observations higher
                total_weight = 0
                weighted_sum = 0
                for i, (obs_val, obs_time) in enumerate(recent_obs):
                    time_weight = self.history_decay ** (len(recent_obs) - i - 1)
                    weighted_sum += obs_val * time_weight
                    total_weight += time_weight
                
                if total_weight > 0:
                    effective_update = weighted_sum / total_weight
                else:
                    effective_update = update_value
            else:
                effective_update = update_value
        else:
            effective_update = update_value

        # Bayesian update with clamping
        new_log_odds = current_log_odds + effective_update
        new_log_odds = np.clip(new_log_odds, self.log_odds_min, self.log_odds_max)

        # Store updated value
        self.grid[voxel_idx] = new_log_odds

        # Update sets for quick access
        if new_log_odds > self.confirmation_threshold:
            self.occupied_voxels.add(voxel_idx)
            self.free_voxels.discard(voxel_idx)
        elif new_log_odds < -self.confirmation_threshold:
            self.free_voxels.add(voxel_idx)
            self.occupied_voxels.discard(voxel_idx)

    def _apply_temporal_decay(self):
        """
        Apply slow temporal decay to maintain consistency over time.
        """
        if self.decay_rate <= 0:
            return

        # Decay all log-odds values toward the neutral value (0)
        current_time = time.time()
        decay_factor = np.exp(-self.decay_rate * self.update_interval)

        # Only decay if we have moved significantly since last update
        if len(self.last_positions) >= 2:
            last_pos = self.last_positions[-1]
            prev_pos = self.last_positions[-2]
            displacement = np.linalg.norm(last_pos - prev_pos)
            
            # Only decay if minimal movement (likely static scene)
            if displacement < 0.05:  # 5cm threshold
                # Only decay voxels near the camera (likely to be "seen" again)
                camera_pos = last_pos
                decay_radius = 2.0  # meters

                for voxel_idx, log_odds in list(self.grid.items()):
                    voxel_world = self.voxel_to_world(voxel_idx)
                    distance = np.linalg.norm(voxel_world - camera_pos)
                    
                    if distance <= decay_radius:
                        # Apply decay toward neutral (0)
                        decayed_value = log_odds * decay_factor
                        self.grid[voxel_idx] = decayed_value
                        
                        # Update sets if needed
                        if abs(decayed_value) < self.confirmation_threshold:
                            self.occupied_voxels.discard(voxel_idx)
                            self.free_voxels.discard(voxel_idx)

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
                self._update_voxel_with_temporal_filtering(voxel, occupied=False)
                updates += 1

        # Mark last voxel as occupied (obstacle endpoint)
        if ray_voxels and self.is_valid_voxel(ray_voxels[-1]):
            self._update_voxel_with_temporal_filtering(ray_voxels[-1], occupied=True)
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

    def get_occupied_voxels_with_probability(self, min_probability: float = 0.7) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get occupied voxel positions with their occupancy probabilities.

        Args:
            min_probability: Minimum occupancy probability threshold

        Returns:
            Tuple of (positions, probabilities) where positions is Nx3 array and 
            probabilities is N array
        """
        occupied_positions = []
        occupied_probabilities = []

        min_log_odds = np.log(min_probability / (1 - min_probability)) if min_probability < 1.0 else float('inf')
        
        for voxel_idx, log_odds in self.grid.items():
            if log_odds >= min_log_odds:
                world_pos = self.voxel_to_world(voxel_idx)
                probability = 1.0 / (1.0 + np.exp(-log_odds))
                
                occupied_positions.append(world_pos)
                occupied_probabilities.append(probability)

        if not occupied_positions:
            return np.zeros((0, 3)), np.zeros(0)

        return np.array(occupied_positions), np.array(occupied_probabilities)

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

        # Resize for better visibility with larger voxels
        vis_colored = cv2.resize(vis_colored, (600, 600), interpolation=cv2.INTER_NEAREST)

        # Add text overlay with larger font
        cv2.putText(vis_colored, f"Occupancy Grid @ z={z_height:.1f}m",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_colored, f"Updates: {self.total_updates}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_colored, f"Occupied: {len(self.occupied_voxels)}",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis_colored

    def visualize_3d_interactive(self, camera_position: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create interactive 3D visualization with larger voxels.

        Args:
            camera_position: Current camera position for highlighting

        Returns:
            Visualization image for display
        """
        # Create larger canvas with WHITE background
        img_size = 800
        canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # White background

        # Draw grid lines for depth perception
        grid_spacing = 40  # pixels
        grid_color = (230, 230, 230)  # Light gray
        for i in range(0, img_size, grid_spacing):
            cv2.line(canvas, (i, 0), (i, img_size), grid_color, 1)
            cv2.line(canvas, (0, i), (img_size, i), grid_color, 1)

        # Get occupied and free voxels
        occupied_voxels = []
        free_voxels = []

        for voxel_idx, log_odds in self.grid.items():
            if log_odds > 0.5:
                occupied_voxels.append(voxel_idx)
            elif log_odds < -0.5:
                free_voxels.append(voxel_idx)

        if not occupied_voxels and not free_voxels:
            # Draw empty grid message (black text on white background)
            cv2.putText(canvas, "No voxels observed yet",
                       (img_size//2 - 150, img_size//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            cv2.putText(canvas, "Move camera to build map",
                       (img_size//2 - 180, img_size//2 + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)
            return canvas

        # Rotation and scale parameters (can be adjusted)
        if not hasattr(self, 'view_angle_x'):
            self.view_angle_x = 30.0  # degrees
            self.view_angle_z = 45.0  # degrees
            self.view_scale = 30.0    # Increased from 20 for larger voxels
            self.view_offset_x = 0.0
            self.view_offset_y = 0.0

        # Project 3D voxels to 2D
        def project_voxel(voxel_idx):
            world_pos = self.voxel_to_world(voxel_idx)

            # Apply rotation
            angle_x = np.radians(self.view_angle_x)
            angle_z = np.radians(self.view_angle_z)

            # Rotate around Z axis
            x_rot = world_pos[0] * np.cos(angle_z) - world_pos[1] * np.sin(angle_z)
            y_rot = world_pos[0] * np.sin(angle_z) + world_pos[1] * np.cos(angle_z)
            z_rot = world_pos[2]

            # Rotate around X axis (tilt)
            y_final = y_rot * np.cos(angle_x) - z_rot * np.sin(angle_x)
            z_final = y_rot * np.sin(angle_x) + z_rot * np.cos(angle_x)

            # Isometric projection
            screen_x = int((x_rot + self.view_offset_x) * self.view_scale + img_size / 2)
            screen_y = int((-y_final + self.view_offset_y) * self.view_scale + img_size / 2)

            return screen_x, screen_y, z_final

        # SKIP free voxels for speed - only show occupied
        # Drawing 1000 voxels is too slow!

        # Draw occupied voxels - LIMIT for speed
        occupied_to_draw = list(occupied_voxels)[:500]  # Max 500 voxels
        voxel_depths = [(v, project_voxel(v)) for v in occupied_to_draw]
        # SKIP SORTING for speed - depth sorting is expensive

        for voxel_idx, (sx, sy, depth) in voxel_depths:
            if 0 <= sx < img_size and 0 <= sy < img_size:
                # MUCH larger voxel size for grid-like appearance
                voxel_size = max(8, int(self.view_scale * self.resolution * 1.5))

                # Color based on probability
                log_odds = self.grid.get(voxel_idx, 0.0)
                prob = 1.0 / (1.0 + np.exp(-log_odds))

                # Bright red for occupied voxels (more visible)
                color_intensity = int(prob * 180 + 75)  # 75-255 range
                color = (40, 40, color_intensity)  # Bright red (BGR format)

                # Draw voxel as THICK rectangle for grid effect
                cv2.rectangle(canvas,
                            (sx - voxel_size, sy - voxel_size),
                            (sx + voxel_size, sy + voxel_size),
                            color, -1)

                # Add THICK dark border for strong grid lines
                cv2.rectangle(canvas,
                            (sx - voxel_size, sy - voxel_size),
                            (sx + voxel_size, sy + voxel_size),
                            (0, 0, 0), 2)  # Thicker black border

                # Add inner grid lines for 3D cube effect
                cv2.line(canvas, (sx - voxel_size, sy), (sx + voxel_size, sy), (0, 0, 0), 1)
                cv2.line(canvas, (sx, sy - voxel_size), (sx, sy + voxel_size), (0, 0, 0), 1)

        # Draw camera position if provided (bright blue on white background)
        if camera_position is not None:
            cam_voxel = self.world_to_voxel(camera_position)
            sx, sy, _ = project_voxel(cam_voxel)
            if 0 <= sx < img_size and 0 <= sy < img_size:
                cv2.circle(canvas, (sx, sy), 10, (255, 200, 0), -1)  # Orange/yellow center
                cv2.circle(canvas, (sx, sy), 12, (0, 0, 0), 2)       # Black border
                cv2.putText(canvas, "YOU", (sx + 15, sy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Black text

        # Add info overlay with light background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 10), (380, 200), (240, 240, 240), -1)  # Light gray
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (10, 10), (380, 200), (100, 100, 100), 2)  # Border

        # Info text (black on light background)
        cv2.putText(canvas, "Voxel Grid Map", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(canvas, f"Occupied Voxels: {len(occupied_voxels)}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 50, 50), 2)  # Dark red
        cv2.putText(canvas, f"Free Voxels: {len(free_voxels)}", (20, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 150, 50), 2)  # Dark green
        cv2.putText(canvas, f"Voxel Size: {self.resolution*100:.0f}cm", (20, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)  # Dark gray
        cv2.putText(canvas, f"Zoom: {self.view_scale:.1f}x", (20, 165),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        cv2.putText(canvas, "GRID VIEW:", (20, 190),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 200), 1)  # Blue

        # Draw camera trajectory for movement visualization
        if not hasattr(self, 'camera_trajectory'):
            self.camera_trajectory = deque(maxlen=100)  # Store last 100 positions
        if camera_position is not None:
            self.camera_trajectory.append(camera_position.copy())

        # Draw camera trajectory if available
        if len(self.camera_trajectory) > 1:
            points = []
            for pos in self.camera_trajectory:
                sx, sy, _ = project_voxel(self.world_to_voxel(pos))
                if 0 <= sx < img_size and 0 <= sy < img_size:
                    points.append((sx, sy))
            
            # Draw trajectory line
            if len(points) > 1:
                for i in range(len(points) - 1):
                    alpha = i / len(points)  # Fade effect
                    color = (int(255 * alpha), 100, int(255 * (1 - alpha)))
                    cv2.line(canvas, points[i], points[i+1], color, 2)
            
            # Draw camera path dots
            for i, point in enumerate(points):
                alpha = i / len(points) if len(points) > 1 else 1.0
                size = 3 if i == len(points) - 1 else 2  # Larger for current position
                color = (0, 255, 255) if i == len(points) - 1 else (100, 100, 100)  # Yellow for current
                cv2.circle(canvas, point, size, color, -1)

        # Current camera position visualization
        if camera_position is not None:
            cam_voxel = self.world_to_voxel(camera_position)
            sx, sy, _ = project_voxel(cam_voxel)
            if 0 <= sx < img_size and 0 <= sy < img_size:
                # Camera indicator with direction
                cv2.circle(canvas, (sx, sy), 12, (0, 255, 255), -1)  # Yellow
                cv2.circle(canvas, (sx, sy), 14, (0, 0, 0), 2)       # Black border
                cv2.putText(canvas, "CAM", (sx + 18, sy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Red text
                # Draw direction indicator (simplified - just show as arrow)
                cv2.arrowedLine(canvas, (sx, sy), (sx + 20, sy), (0, 100, 255), 2, tipLength=0.3)

        # Info overlay
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 10), (450, 250), (245, 245, 245), -1)  # Light gray
        cv2.addWeighted(overlay, 0.9, canvas, 0.1, 0, canvas)
        cv2.rectangle(canvas, (10, 10), (450, 250), (128, 128, 128), 2)  # Border

        # Info text (black on light background)
        cv2.putText(canvas, "Simplified 3D Occupancy Grid", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        cv2.putText(canvas, f"Voxels: {len(occupied_voxels):,}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 200), 2)
        cv2.putText(canvas, f"Resolution: {self.resolution*100:.1f}cm", (20, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 150, 0), 2)
        cv2.putText(canvas, f"Grid Size: {self.grid_size[0]}x{self.grid_size[1]}x{self.grid_size[2]}m", (20, 135),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 0, 150), 2)
        cv2.putText(canvas, f"Updates: {self.total_updates}", (20, 165),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
        cv2.putText(canvas, f"Frame: {self.frame_count}", (20, 195),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 2)
        cv2.putText(canvas, f"Zoom: {self.view_scale:.1f}x", (20, 225),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return canvas

    def update_view_controls(self, key: int) -> bool:
        """
        Update view parameters based on keyboard input with controls.

        Args:
            key: OpenCV key code

        Returns:
            True if view was updated
        """
        updated = False

        # Zoom
        if key == ord('+') or key == ord('='):
            self.view_scale *= 1.2
            updated = True
        elif key == ord('-') or key == ord('_'):
            self.view_scale *= 0.8
            updated = True

        # Pan
        elif key == 82:  # Up arrow
            self.view_offset_y += 0.5
            updated = True
        elif key == 84:  # Down arrow
            self.view_offset_y -= 0.5
            updated = True
        elif key == 81:  # Left arrow
            self.view_offset_x -= 0.5
            updated = True
        elif key == 83:  # Right arrow
            self.view_offset_x += 0.5
            updated = True

        # Rotate Y-axis (yaw)
        elif key == ord('q') or key == ord('Q'):
            self.view_angle_z -= 5.0
            updated = True
        elif key == ord('e') or key == ord('E'):
            self.view_angle_z += 5.0
            updated = True

        # Rotate X-axis (pitch)
        elif key == ord('w') or key == ord('W'):
            self.view_angle_x += 5.0
            updated = True
        elif key == ord('s') or key == ord('S'):
            self.view_angle_x -= 5.0
            updated = True

        # Rotate Z-axis (roll)
        elif key == ord('a') or key == ord('A'):
            self.view_angle_y = getattr(self, 'view_angle_y', 0.0) - 5.0
            updated = True
        elif key == ord('d') or key == ord('D'):
            self.view_angle_y = getattr(self, 'view_angle_y', 0.0) + 5.0
            updated = True

        # Reset
        elif key == ord('r') or key == ord('R'):
            self.view_angle_x = 25.0
            self.view_angle_z = 45.0
            self.view_angle_y = 0.0
            self.view_scale = 40.0
            self.view_offset_x = 0.0
            self.view_offset_y = 0.0
            updated = True

        # Clear trajectory
        elif key == ord('c') or key == ord('C'):
            if hasattr(self, 'camera_trajectory'):
                self.camera_trajectory.clear()
            updated = False  # Not a view change, just a data clear

        return updated

    def handle_mouse_wheel(self, delta: int):
        """
        Handle mouse wheel for zooming with sensitivity.

        Args:
            delta: Mouse wheel delta (positive = zoom in, negative = zoom out)
        """
        if delta > 0:
            self.view_scale *= 1.15  # Zoom sensitivity
        else:
            self.view_scale *= 0.85

        # Clamp zoom with extended range for better exploration
        self.view_scale = np.clip(self.view_scale, 10.0, 200.0)

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
