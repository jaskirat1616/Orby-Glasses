"""
SLAM System without IMU for OrbyGlasses
Visual SLAM with 3D mapping and movement visualization
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import deque, defaultdict
import time
import json
import os
from dataclasses import dataclass, asdict


@dataclass
class MapPoint:
    """3D point in the map."""
    id: int
    position: np.ndarray  # 3D coordinates [x, y, z]
    descriptor: np.ndarray  # ORB descriptor
    observations: int = 0  # Number of times observed
    reprojection_error: float = float('inf')  # Error of 3D-2D reprojected point

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'descriptor': self.descriptor.tolist(),
            'observations': self.observations,
            'reprojection_error': self.reprojection_error
        }


@dataclass
class KeyFrame:
    """Key frame with pose and features."""
    id: int
    timestamp: float
    pose: np.ndarray  # 4x4 transformation matrix
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    frame: Optional[np.ndarray] = None  # Optional: store frame for visualization
    tracked_features: int = 0  # Number of features successfully tracked

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'pose': self.pose.tolist(),
            'keypoints': [(kp.pt[0], kp.pt[1], kp.size, kp.angle) for kp in self.keypoints],
            'descriptors': self.descriptors.tolist(),
            'tracked_features': self.tracked_features
        }


class SLAMTracking:
    """
    Monocular SLAM without IMU for 3D mapping.
    Uses visual tracking and pose estimation without requiring IMU data.
    """

    def __init__(self, config):
        """
        Initialize SLAM system.

        Args:
            config: ConfigManager instance
        """
        self.config = config
        self.enabled = config.get('advanced_slam.enabled', True)

        # Camera intrinsics (estimated from config)
        self.fx = config.get('mapping3d.fx', 500)
        self.fy = config.get('mapping3d.fy', 500)
        self.cx = config.get('camera.width', 320) / 2
        self.cy = config.get('camera.height', 320) / 2

        # Camera matrix
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Inverse camera matrix for back-projection
        self.K_inv = np.linalg.inv(self.K)

        # ORB feature detector
        nfeatures = config.get('advanced_slam.orb_features', 2000)
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,  # Default value for better feature detection
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=20
        )

        # Feature matcher with cross-check disabled (for better matches)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # FLANN matcher for better performance with many features
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6, key_size = 12, multi_probe_level = 1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # SLAM state
        self.map_points = {}  # id -> MapPoint
        self.keyframes = []  # List of KeyFrame
        self.current_pose = np.eye(4, dtype=np.float32)  # Current camera pose
        self.previous_pose = np.eye(4, dtype=np.float32)  # Previous camera pose
        self.next_point_id = 0
        self.next_keyframe_id = 0
        self.reference_keyframe_id = 0

        # Tracking state
        self.last_frame = None
        self.last_keypoints = None
        self.last_descriptors = None
        self.is_initialized = False

        # Parameters
        self.min_matches = config.get('advanced_slam.min_matches', 20)
        self.min_tracked_features = config.get('advanced_slam.min_tracked_features', 15)
        self.keyframe_threshold = config.get('advanced_slam.keyframe_threshold', 25)
        self.scale_threshold = config.get('advanced_slam.scale_threshold', 0.1)
        self.reprojection_threshold = config.get('advanced_slam.reprojection_threshold', 3.0)
        self.min_depth = config.get('advanced_slam.min_depth', 0.1)
        self.max_depth = config.get('advanced_slam.max_depth', 10.0)

        # Motion model for pose prediction (no IMU needed)
        self.velocity = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]
        self.last_pose_update_time = time.time()
        self.last_positions = deque(maxlen=5)  # Last 5 positions for smoothing

        # Pose smoothing parameters
        self.pose_alpha = config.get('advanced_slam.pose_alpha', 0.7)
        self.smoothed_pose = np.eye(4, dtype=np.float32)

        # Loop closure detection
        self.enable_loop_closure = config.get('advanced_slam.loop_closure', False)
        self.loop_closure_threshold = config.get('advanced_slam.loop_closure_threshold', 0.5)
        self.keyframe_database = []  # For loop closure detection

        # Bundle adjustment (simplified)
        self.enable_bundle_adjustment = config.get('advanced_slam.bundle_adjustment', False)

        # Map saving
        self.map_save_dir = "data/maps"
        os.makedirs(self.map_save_dir, exist_ok=True)

        # Position history for navigation and trajectory
        self.position_history = deque(maxlen=1000)
        self.pose_history = deque(maxlen=100)  # Keep last 100 poses for analysis

        # Statistics
        self.frame_count = 0
        self.total_features = 0
        self.tracked_features = 0
        self.lost_frames = 0
        self.keyframe_count = 0
        
        # Relative pose for movement tracking
        self.relative_pose_history = deque(maxlen=20)

        logging.info("Simplified SLAM initialized (camera-only, without IMU)")
        logging.info(f"Camera matrix: fx={self.fx}, fy={self.fy}, cx={self.cx:.1f}, cy={self.cy:.1f}")
        logging.info(f"ORB features: {nfeatures}, Min matches: {self.min_matches}")

    def process_frame(self, frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> Dict:
        """
        Process a new frame through SLAM pipeline.

        Args:
            frame: Input frame (BGR)
            depth_map: Optional depth map for scale initialization (H x W, normalized 0-1)

        Returns:
            Dictionary with tracking info:
                - pose: Current 4x4 pose matrix
                - position: [x, y, z] position
                - tracking_quality: 0-1 quality score
                - num_matches: Number of feature matches
                - is_keyframe: Whether this is a keyframe
                - relative_movement: [dx, dy, dz, drx, dry, drz] relative movement
        """
        if not self.enabled:
            return self._empty_result()

        self.frame_count += 1

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ORB features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 10:
            if self.is_initialized:
                # If already initialized, try to keep the last pose
                position = self.current_pose[:3, 3]
                self.position_history.append(position)
                
                # Calculate relative movement
                relative_movement = np.zeros(6)
                if len(self.position_history) > 1:
                    pos1 = self.position_history[-2]
                    pos2 = self.position_history[-1]
                    # Calculate translation
                    translation = pos2 - pos1
                    # For rotation, we'll use the difference in pose rotation matrices
                    prev_rot = self.previous_pose[:3, :3] if hasattr(self, 'previous_pose') else np.eye(3)
                    curr_rot = self.current_pose[:3, :3]
                    # Simple approximation of rotation change
                    rot_diff = cv2.Rodrigues(np.dot(curr_rot, prev_rot.T))[0].flatten()
                    
                    relative_movement = np.array([translation[0], translation[1], translation[2],
                                                 rot_diff[0], rot_diff[1], rot_diff[2]])
                
                return {
                    'pose': self.current_pose,
                    'position': position.tolist(),
                    'tracking_quality': 0.1,
                    'num_matches': 0,
                    'is_keyframe': False,
                    'num_map_points': len(self.map_points),
                    'relative_movement': relative_movement.tolist()
                }
            else:
                logging.warning(f"Insufficient features for initialization: {len(keypoints) if keypoints else 0}")
                return self._empty_result()

        # Store current descriptors and keypoints for next frame
        current_keypoints = keypoints
        current_descriptors = descriptors

        # Initialize or track
        if not self.is_initialized:
            result = self._initialize(gray, current_keypoints, current_descriptors, depth_map)
        else:
            result = self._track(gray, current_keypoints, current_descriptors, depth_map)

        # Store for next frame
        self.last_frame = gray
        self.last_keypoints = current_keypoints
        self.last_descriptors = current_descriptors

        # Update position history
        position = self.current_pose[:3, 3]
        self.position_history.append(position)
        self.pose_history.append(self.current_pose.copy())
        
        # Store relative pose for movement analysis
        if hasattr(self, 'previous_pose'):
            rel_pose = np.linalg.inv(self.previous_pose) @ self.current_pose
            self.relative_pose_history.append(rel_pose)
        
        self.previous_pose = self.current_pose.copy()

        # Calculate relative movement
        relative_movement = np.zeros(6)
        if len(self.position_history) > 1:
            pos1 = self.position_history[-2]
            pos2 = self.position_history[-1]
            # Calculate translation
            translation = pos2 - pos1
            # For rotation, calculate from pose matrices
            if len(self.pose_history) > 1:
                prev_rot = self.pose_history[-2][:3, :3]
                curr_rot = self.current_pose[:3, :3]
                # Use Rodrigues to get rotation vector
                rot_diff_matrix = np.dot(curr_rot, prev_rot.T)
                rot_diff = cv2.Rodrigues(rot_diff_matrix)[0].flatten()
            else:
                rot_diff = np.zeros(3)
                
            relative_movement = np.array([translation[0], translation[1], translation[2],
                                         rot_diff[0], rot_diff[1], rot_diff[2]])

        # Update result with relative movement
        result['relative_movement'] = relative_movement.tolist()
        result['position'] = position.tolist()

        return result

    def _initialize(self, frame: np.ndarray, keypoints: List, descriptors: np.ndarray,
                    depth_map: Optional[np.ndarray] = None) -> Dict:
        """
        Initialize SLAM with first frame and more stable feature initialization.
        """
        logging.info(f"Initializing SLAM with {len(keypoints)} features...")

        # Create initial keyframe at origin
        initial_pose = np.eye(4, dtype=np.float32)
        keyframe = KeyFrame(
            id=self.next_keyframe_id,
            timestamp=time.time(),
            pose=initial_pose,
            keypoints=keypoints,
            descriptors=descriptors,
            tracked_features=len(keypoints)
        )
        self.keyframes.append(keyframe)
        self.next_keyframe_id += 1

        # Initialize map points with depth information if available
        points_added = 0
        if depth_map is not None:
            # Use depth map to create initial 3D points
            for i, (kp, desc) in enumerate(zip(keypoints[:500], descriptors[:500])):
                u, v = int(kp.pt[0]), int(kp.pt[1])
                if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                    depth_norm = depth_map[v, u]
                    depth = depth_norm * self.max_depth  # Convert to meters
                    
                    if self.min_depth <= depth <= self.max_depth:
                        # Back-project to 3D point in camera frame
                        x_cam = (u - self.cx) * depth / self.fx
                        y_cam = (v - self.cy) * depth / self.fy
                        z_cam = depth
                        
                        # Transform to world frame (initially identity)
                        point_3d = np.array([x_cam, y_cam, z_cam])
                        
                        map_point = MapPoint(
                            id=self.next_point_id,
                            position=point_3d,
                            descriptor=desc,
                            observations=1,
                            reprojection_error=0.0
                        )
                        self.map_points[self.next_point_id] = map_point
                        self.next_point_id += 1
                        points_added += 1

        # If no depth map, create initial points with assumed depth
        if points_added == 0:
            for i, (kp, desc) in enumerate(zip(keypoints[:500], descriptors[:500])):
                # Use a reasonable default depth
                depth = 2.0  # meters
                u, v = int(kp.pt[0]), int(kp.pt[1])
                
                # Back-project to 3D point in camera frame
                x_cam = (u - self.cx) * depth / self.fx
                y_cam = (v - self.cy) * depth / self.fy
                z_cam = depth
                
                point_3d = np.array([x_cam, y_cam, z_cam])
                
                map_point = MapPoint(
                    id=self.next_point_id,
                    position=point_3d,
                    descriptor=desc,
                    observations=1,
                    reprojection_error=0.0
                )
                self.map_points[self.next_point_id] = map_point
                self.next_point_id += 1
                points_added += 1
                if points_added >= 100:  # Limit initial points
                    break

        self.is_initialized = True
        self.position_history.append([0, 0, 0])
        self.pose_history.append(initial_pose.copy())

        logging.info(f"SLAM initialized with {points_added} map points")

        return {
            'pose': initial_pose,
            'position': [0, 0, 0],
            'tracking_quality': 1.0,
            'num_matches': points_added,
            'is_keyframe': True,
            'num_map_points': len(self.map_points),
            'relative_movement': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }

    def _track(self, frame: np.ndarray, keypoints: List, descriptors: np.ndarray,
               depth_map: Optional[np.ndarray] = None) -> Dict:
        """
        Track camera motion by matching features with previous frame.
        """
        # Match features with last frame using FLANN
        try:
            matches = self.flann.knnMatch(descriptors, self.last_descriptors, k=2)
        except:
            # Fallback to brute force matcher
            matches = self.matcher.knnMatch(descriptors, self.last_descriptors, k=2)

        # Apply Lowe's ratio test with adaptive threshold
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # Use adaptive threshold based on image content
                ratio_threshold = 0.75
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        num_matches = len(good_matches)

        # Initialize defaults
        position = self.current_pose[:3, 3].tolist()
        tracking_quality = 0.0
        is_keyframe = False

        # Predict pose using motion model
        current_time = time.time()
        dt = current_time - self.last_pose_update_time
        predicted_pose = self._predict_pose(dt)

        if num_matches < self.min_matches:
            # Use predicted pose when tracking is weak
            if num_matches >= 10:
                # Partial tracking - blend prediction with last pose
                self.current_pose = predicted_pose
                tracking_quality = max(0.1, num_matches / self.min_matches)
            else:
                # Very weak tracking - use prediction only
                self.current_pose = predicted_pose
                tracking_quality = 0.1
                self.lost_frames += 1
                logging.debug(f"Low tracking: {num_matches} matches (using motion prediction)")
        else:
            # Good tracking - estimate pose from features
            if len(good_matches) >= 5:  # Need at least 5 points for Essential matrix
                # Get matched points
                pts_current = np.float32([keypoints[m.queryIdx].pt for m in good_matches])
                pts_last = np.float32([self.last_keypoints[m.trainIdx].pt for m in good_matches])
                
                # Estimate essential matrix with RANSAC
                E, mask = cv2.findEssentialMat(
                    pts_current, pts_last, self.K,
                    method=cv2.RANSAC, 
                    prob=0.999, 
                    threshold=1.0  # Was 0.8, increased for more stability
                )

                if E is not None and mask is not None:
                    # Count inliers
                    inliers = np.sum(mask)
                    
                    if inliers >= 5:  # Need at least 5 inliers
                        # Recover pose from essential matrix
                        success, R, t, mask_pose = cv2.recoverPose(
                            E, pts_current, pts_last, self.K, mask=mask
                        )

                        if success:
                            # Scale estimation for monocular SLAM using depth information
                            scale = self._estimate_scale_from_depth(
                                pts_current, pts_last, t.flatten(), depth_map
                            )

                            # Apply scale to translation
                            t_scaled = t * scale

                            # Create transformation matrix
                            T = np.eye(4, dtype=np.float32)
                            T[:3, :3] = R
                            T[:3, 3] = t_scaled.flatten()

                            # Update pose (relative to last pose)
                            new_pose = self.current_pose @ T

                            # Apply pose smoothing
                            self.current_pose = self._smooth_pose(new_pose)

                            # Update velocity for motion model
                            if dt > 0.001:  # Avoid division by zero
                                self._update_velocity(dt)
                                self.last_pose_update_time = current_time

                            # Extract position
                            position = self.current_pose[:3, 3].tolist()
                            
                            # Update statistics
                            self.tracked_features += inliers
                            self.total_features += len(good_matches)

                            # Tracking quality based on inliers
                            tracking_quality = min(1.0, inliers / self.min_matches)
                        else:
                            # Pose recovery failed
                            self.current_pose = predicted_pose
                            tracking_quality = 0.3
                    else:
                        # Not enough inliers
                        self.current_pose = predicted_pose
                        tracking_quality = 0.3
                else:
                    # Essential matrix estimation failed
                    self.current_pose = predicted_pose
                    tracking_quality = 0.3
            else:
                # Not enough matches to estimate pose
                self.current_pose = predicted_pose
                tracking_quality = 0.3

        # Check if we need a new keyframe based on motion and tracking quality
        is_keyframe = self._should_insert_keyframe(num_matches, tracking_quality)

        if is_keyframe:
            self._insert_keyframe(frame, keypoints, descriptors, tracking_quality)

        return {
            'pose': self.current_pose,
            'position': position,
            'tracking_quality': tracking_quality,
            'num_matches': num_matches,
            'is_keyframe': is_keyframe,
            'num_map_points': len(self.map_points),
            'relative_movement': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Will be updated in main function
        }

    def _estimate_scale_from_depth(self, pts_current: np.ndarray, pts_last: np.ndarray,
                                  translation: np.ndarray, depth_map: Optional[np.ndarray]) -> float:
        """
        Estimate scale factor from depth information for monocular SLAM.
        """
        if depth_map is None:
            # Default scale for monocular SLAM
            return 0.1  # meters per frame (baseline scale)

        # Estimate scale based on depth information
        scale = 0.1  # default
        
        if len(pts_current) >= 3:
            # Calculate distances in current frame
            current_depths = []
            for pt in pts_current[:10]:  # Use first 10 points for efficiency
                u, v = int(pt[0]), int(pt[1])
                if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                    depth_norm = depth_map[v, u]
                    depth = depth_norm * self.max_depth
                    if self.min_depth <= depth <= self.max_depth:
                        current_depths.append(depth)
            
            if len(current_depths) > 0:
                avg_depth = np.mean(current_depths)
                # Calculate expected translation based on depth
                expected_translation = np.linalg.norm(translation) * avg_depth
                # Use this to estimate scale
                if expected_translation > 0.01:  # Avoid very small values
                    scale = min(0.5, expected_translation)  # Limit scale to reasonable range
                else:
                    scale = 0.1

        return scale

    def _should_insert_keyframe(self, num_matches: int, tracking_quality: float) -> bool:
        """
        Decide whether to insert a new keyframe based on multiple factors.
        """
        # Always insert if quality is very high but not done recently
        if tracking_quality > 0.7 and self.frame_count % 50 == 0:
            return True

        # Insert if tracking quality is low and matches are few
        if tracking_quality < 0.3 and num_matches < self.min_matches:
            return True

        # Insert if we have enough good matches and quality is decent
        if num_matches > self.min_matches * 2 and tracking_quality > 0.5:
            return True

        # Check if camera moved significantly
        if len(self.position_history) >= 2:
            last_pos = np.array(self.position_history[-1])
            prev_pos = np.array(self.position_history[-2])
            displacement = np.linalg.norm(last_pos - prev_pos)
            
            if displacement > 0.3:  # 30cm movement threshold
                return True

        # Insert every N frames if conditions are met
        if self.frame_count % self.keyframe_threshold == 0:
            return True

        return False

    def _insert_keyframe(self, frame: np.ndarray, keypoints: List, descriptors: np.ndarray, 
                        tracking_quality: float):
        """
        Insert a new keyframe with mapping.
        """
        keyframe = KeyFrame(
            id=self.next_keyframe_id,
            timestamp=time.time(),
            pose=self.current_pose.copy(),
            keypoints=keypoints,
            descriptors=descriptors,
            tracked_features=len(keypoints)
        )
        self.keyframes.append(keyframe)
        self.keyframe_count += 1
        self.next_keyframe_id += 1

        # Add to keyframe database for potential loop closure
        if self.enable_loop_closure:
            self.keyframe_database.append(keyframe)

        logging.info(f"Inserted keyframe {keyframe.id} at position {self.current_pose[:3, 3]}, "
                    f"quality: {tracking_quality:.2f}")

    def get_position(self) -> np.ndarray:
        """Get current camera position in world coordinates."""
        return self.current_pose[:3, 3]

    def get_orientation(self) -> np.ndarray:
        """Get current camera orientation (rotation matrix)."""
        return self.current_pose[:3, :3]

    def get_pose(self) -> np.ndarray:
        """Get current camera pose (4x4 matrix)."""
        return self.current_pose.copy()

    def get_velocity(self) -> np.ndarray:
        """Get current camera velocity."""
        return self.velocity.copy()

    def get_map_points_array(self) -> np.ndarray:
        """Get all map points as Nx3 array for visualization."""
        if not self.map_points:
            return np.zeros((0, 3))
        return np.array([mp.position for mp in self.map_points.values()])

    def get_trajectory(self) -> np.ndarray:
        """Get camera trajectory as Nx3 array."""
        if not self.position_history:
            return np.zeros((0, 3))
        return np.array(list(self.position_history))

    def get_relative_movement(self) -> np.ndarray:
        """Get relative movement from last frame."""
        if not self.relative_pose_history:
            return np.zeros(6)
        
        last_rel_pose = self.relative_pose_history[-1]
        # Extract translation
        translation = last_rel_pose[:3, 3]
        
        # Extract rotation (Rodrigues vector)
        rotation_matrix = last_rel_pose[:3, :3]
        rotation_vec = cv2.Rodrigues(rotation_matrix)[0].flatten()
        
        return np.array([translation[0], translation[1], translation[2],
                        rotation_vec[0], rotation_vec[1], rotation_vec[2]])

    def get_tracking_stats(self) -> Dict:
        """Get tracking statistics."""
        return {
            'frame_count': self.frame_count,
            'keyframe_count': self.keyframe_count,
            'total_features': self.total_features,
            'tracked_features': self.tracked_features,
            'lost_frames': self.lost_frames,
            'map_points': len(self.map_points),
            'current_position': self.current_pose[:3, 3].tolist(),
            'tracking_quality': self._calculate_tracking_quality()
        }

    def _calculate_tracking_quality(self) -> float:
        """Calculate overall tracking quality."""
        if not self.position_history or len(self.position_history) < 2:
            return 0.5
            
        # Calculate average displacement between consecutive frames
        displacements = []
        pos_list = list(self.position_history)
        for i in range(1, len(pos_list)):
            disp = np.linalg.norm(np.array(pos_list[i]) - np.array(pos_list[i-1]))
            displacements.append(disp)
            
        if not displacements:
            return 0.5
            
        avg_displacement = np.mean(displacements)
        # Quality based on movement consistency (not too much, not too little)
        quality = min(1.0, avg_displacement * 2)  # Scale factor to normalize
        return min(1.0, quality)

    def save_map(self, filename: str = None):
        """
        Save map to file with information.

        Args:
            filename: Output filename (default: auto-generated)
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"advanced_map_{timestamp}.json"

        filepath = os.path.join(self.map_save_dir, filename)

        # Prepare data with current state
        map_data = {
            'timestamp': time.time(),
            'num_keyframes': len(self.keyframes),
            'num_map_points': len(self.map_points),
            'camera_matrix': self.K.tolist(),
            'current_pose': self.current_pose.tolist(),
            'position_history': [pos.tolist() if isinstance(pos, np.ndarray) else pos 
                                for pos in list(self.position_history)],
            'keyframes': [kf.to_dict() for kf in self.keyframes],
            'map_points': [mp.to_dict() for mp in self.map_points.values()],
            'tracking_stats': self.get_tracking_stats()
        }

        # Save
        with open(filepath, 'w') as f:
            json.dump(map_data, f, indent=2)

        logging.info(f"Advanced map saved to {filepath}")
        return filepath

    def load_map(self, filepath: str) -> bool:
        """
        Load map from file.

        Args:
            filepath: Path to map file

        Returns:
            True if successful
        """
        try:
            with open(filepath, 'r') as f:
                map_data = json.load(f)

            # Clear current map
            self.map_points = {}
            self.keyframes = []

            # Reconstruct map points
            for mp_data in map_data.get('map_points', []):
                mp = MapPoint(
                    id=mp_data['id'],
                    position=np.array(mp_data['position']),
                    descriptor=np.array(mp_data['descriptor'], dtype=np.uint8),
                    observations=mp_data['observations'],
                    reprojection_error=mp_data.get('reprojection_error', float('inf'))
                )
                self.map_points[mp.id] = mp

            # Reconstruct keyframes
            for kf_data in map_data.get('keyframes', []):
                # Reconstruct keypoints
                keypoints = [
                    cv2.KeyPoint(x=pt[0], y=pt[1], size=pt[2], angle=pt[3])
                    for pt in kf_data['keypoints']
                ]

                kf = KeyFrame(
                    id=kf_data['id'],
                    timestamp=kf_data['timestamp'],
                    pose=np.array(kf_data['pose']),
                    keypoints=keypoints,
                    descriptors=np.array(kf_data['descriptors'], dtype=np.uint8),
                    tracked_features=kf_data.get('tracked_features', len(keypoints))
                )
                self.keyframes.append(kf)

            # Restore current pose and position history
            if 'current_pose' in map_data:
                self.current_pose = np.array(map_data['current_pose'])
            if 'position_history' in map_data:
                for pos in map_data['position_history']:
                    self.position_history.append(np.array(pos) if not isinstance(pos, np.ndarray) else pos)

            # Update IDs
            self.is_initialized = True
            self.next_point_id = max([mp.id for mp in self.map_points.values()], default=-1) + 1
            self.next_keyframe_id = len(self.keyframes)

            logging.info(f"Advanced map loaded: {len(self.keyframes)} keyframes, {len(self.map_points)} points")
            return True

        except Exception as e:
            logging.error(f"Failed to load advanced map: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _predict_pose(self, dt: float) -> np.ndarray:
        """
        Predict next pose using constant velocity motion model.

        Args:
            dt: Time since last update

        Returns:
            Predicted 4x4 pose matrix
        """
        # Predict translation
        predicted_pose = self.current_pose.copy()
        
        # Calculate translation based on linear velocity
        translation_delta = self.velocity[:3] * dt
        predicted_pose[:3, 3] += translation_delta

        # Calculate rotation based on angular velocity
        if np.linalg.norm(self.velocity[3:]) > 0.001:  # Avoid small rotation calculations
            angle = np.linalg.norm(self.velocity[3:]) * dt
            if angle > 0.001:  # Only apply significant rotations
                axis = self.velocity[3:] / np.linalg.norm(self.velocity[3:])
                
                # Rodrigues rotation formula
                K = np.array([[0, -axis[2], axis[1]],
                             [axis[2], 0, -axis[0]],
                             [-axis[1], axis[0], 0]])
                R_delta = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
                
                predicted_pose[:3, :3] = predicted_pose[:3, :3] @ R_delta

        return predicted_pose

    def _update_velocity(self, dt: float):
        """
        Update velocity estimate from pose changes.

        Args:
            dt: Time delta
        """
        if dt < 0.001 or len(self.position_history) < 2:
            return

        # Calculate linear velocity
        current_pos = self.current_pose[:3, 3]
        if len(self.position_history) >= 2:
            prev_pos = np.array(self.position_history[-2])
            linear_vel = (current_pos - prev_pos) / dt

            # Exponential moving average for smoothing
            alpha = 0.3
            self.velocity[:3] = alpha * linear_vel + (1 - alpha) * self.velocity[:3]

        # Calculate angular velocity from rotation changes
        if len(self.pose_history) >= 2:
            prev_rot = self.pose_history[-2][:3, :3]
            curr_rot = self.current_pose[:3, :3]
            
            # Compute rotation difference
            rot_diff = np.dot(curr_rot, prev_rot.T)
            rot_vec = cv2.Rodrigues(rot_diff)[0].flatten()
            
            angular_vel = rot_vec / dt
            
            # Apply smoothing
            self.velocity[3:] = alpha * angular_vel + (1 - alpha) * self.velocity[3:]

    def _smooth_pose(self, new_pose: np.ndarray) -> np.ndarray:
        """
        Apply exponential moving average to smooth pose.

        Args:
            new_pose: Newly estimated pose

        Returns:
            Smoothed pose
        """
        # Smooth translation using exponential moving average
        self.smoothed_pose[:3, 3] = (self.pose_alpha * new_pose[:3, 3] +
                                    (1 - self.pose_alpha) * self.smoothed_pose[:3, 3])

        # Smooth rotation - blend rotation matrices
        R_new = new_pose[:3, :3]
        R_old = self.smoothed_pose[:3, :3]
        
        # Blend rotations using Rodrigues vector interpolation
        rot_diff = np.dot(R_new, R_old.T)
        rot_vec = cv2.Rodrigues(rot_diff)[0].flatten()
        rot_vec_smooth = rot_vec * self.pose_alpha
        
        R_blend = np.dot(cv2.Rodrigues(rot_vec_smooth)[0], R_old)
        self.smoothed_pose[:3, :3] = R_blend

        return self.smoothed_pose.copy()

    def reset(self):
        """Reset SLAM to initial state."""
        self.map_points = {}
        self.keyframes = []
        self.current_pose = np.eye(4, dtype=np.float32)
        self.next_point_id = 0
        self.next_keyframe_id = 0
        self.is_initialized = False
        self.position_history.clear()
        self.pose_history.clear()
        self.velocity = np.zeros(6)
        self.smoothed_pose = np.eye(4, dtype=np.float32)
        self.frame_count = 0
        self.keyframe_count = 0
        self.lost_frames = 0
        self.total_features = 0
        self.tracked_features = 0
        self.relative_pose_history.clear()
        
        logging.info("Advanced SLAM reset")

    def _empty_result(self) -> Dict:
        """Return empty result when SLAM is disabled or fails."""
        return {
            'pose': np.eye(4, dtype=np.float32),
            'position': [0, 0, 0],
            'tracking_quality': 0.0,
            'num_matches': 0,
            'is_keyframe': False,
            'num_map_points': 0,
            'relative_movement': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }

    def visualize_tracking(self, frame: np.ndarray, tracking_result: Dict) -> np.ndarray:
        """
        Simplified visualization of SLAM tracking on frame with movement info.

        Args:
            frame: Input frame
            tracking_result: Result from process_frame()

        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()

        # Draw features if available
        if self.last_keypoints:
            for kp in self.last_keypoints[:50]:  # Draw top 50 features
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(vis_frame, (x, y), 2, (0, 255, 0), -1)

        # Draw tracking info with details
        h, w = frame.shape[:2]

        # Background overlay for text
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (5, h - 150), (350, h - 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)

        # Extract information
        position = tracking_result['position']
        quality = tracking_result['tracking_quality']
        relative_movement = tracking_result.get('relative_movement', [0]*6)
        dx, dy, dz = relative_movement[0], relative_movement[1], relative_movement[2]
        drx, dry, drz = relative_movement[3], relative_movement[4], relative_movement[5]

        # Display position
        cv2.putText(vis_frame, f"Pos: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})",
                   (10, h - 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Tracking quality with color coding
        quality_color = (0, 255, 0) if quality > 0.7 else (0, 165, 255) if quality > 0.4 else (0, 0, 255)
        cv2.putText(vis_frame, f"Quality: {quality:.2f}",
                   (10, h - 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, quality_color, 1)

        # Movement information
        cv2.putText(vis_frame, f"Δ: ({dx:.3f}, {dy:.3f}, {dz:.3f})m",
                   (10, h - 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
        
        cv2.putText(vis_frame, f"R: ({np.degrees(drx):.1f}, {np.degrees(dry):.1f}, {np.degrees(drz):.1f})°",
                   (10, h - 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        # Feature information
        cv2.putText(vis_frame, f"Matches: {tracking_result['num_matches']}",
                   (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(vis_frame, f"Map Points: {tracking_result['num_map_points']}",
                   (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Keyframe indicator
        if tracking_result['is_keyframe']:
            cv2.putText(vis_frame, "KEYFRAME", (w - 130, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return vis_frame

    def get_movement_summary(self) -> Dict:
        """Get summary of movement and trajectory."""
        if not self.position_history:
            return {
                'total_distance': 0.0,
                'average_speed': 0.0,
                'current_speed': 0.0,
                'trajectory_length': 0
            }

        positions = list(self.position_history)
        if len(positions) < 2:
            return {
                'total_distance': 0.0,
                'average_speed': 0.0,
                'current_speed': 0.0,
                'trajectory_length': len(positions)
            }

        # Calculate total distance traveled
        total_distance = 0.0
        for i in range(1, len(positions)):
            dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
            total_distance += dist

        # Calculate current speed (if enough history)
        current_speed = 0.0
        if len(positions) >= 2:
            last_pos = positions[-1]
            prev_pos = positions[-2]
            displacement = np.linalg.norm(np.array(last_pos) - np.array(prev_pos))
            # Assuming regular frame rate, we can estimate speed
            current_speed = displacement * 10  # Rough estimate at 10 FPS

        # Calculate average speed
        avg_speed = total_distance / len(positions) if len(positions) > 0 else 0.0

        return {
            'total_distance': total_distance,
            'average_speed': avg_speed,
            'current_speed': current_speed,
            'trajectory_length': len(positions)
        }