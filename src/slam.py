"""
OrbyGlasses - Enhanced Visual SLAM for Indoor Navigation
Advanced monocular SLAM using ORB features with improved accuracy for camera-only localization and mapping.
No IMU required - works with just a USB webcam with enhanced tracking accuracy.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import deque, defaultdict
import json
import os
from dataclasses import dataclass, asdict
import time


@dataclass
class MapPoint:
    """3D point in the map with tracking information."""
    id: int
    position: np.ndarray  # 3D coordinates [x, y, z]
    descriptor: np.ndarray  # ORB descriptor
    observations: int = 0  # Number of times observed
    reprojection_error: float = float('inf')  # Reprojection error
    last_observed_frame: int = 0  # Frame number of last observation
    is_tracked: bool = True  # Whether the point is currently being tracked

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'descriptor': self.descriptor.tolist(),
            'observations': self.observations,
            'reprojection_error': self.reprojection_error,
            'last_observed_frame': self.last_observed_frame,
            'is_tracked': self.is_tracked
        }


@dataclass
class KeyFrame:
    """Simplified Key frame with pose and features."""
    id: int
    timestamp: float
    pose: np.ndarray  # 4x4 transformation matrix
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    frame: Optional[np.ndarray] = None  # Optional: store frame for visualization
    tracked_features: int = 0  # Number of successfully tracked features
    feature_quality: float = 0.0  # Quality score of features
    motion_consistency: float = 0.0  # How consistent this frame's motion is

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'pose': self.pose.tolist(),
            'keypoints': [(kp.pt[0], kp.pt[1], kp.size, kp.angle) for kp in self.keypoints],
            'descriptors': self.descriptors.tolist(),
            'tracked_features': self.tracked_features,
            'feature_quality': self.feature_quality,
            'motion_consistency': self.motion_consistency
        }


class MonocularSLAM:
    """
    Simplified monocular SLAM for indoor navigation with better accuracy.
    Uses ORB features with tracking and mapping without IMU.
    Includes temporal consistency, scale estimation, and tracking.

    This implementation focuses on accuracy and stability for navigation purposes
    without requiring IMU data.
    """

    def __init__(self, config):
        """
        Initialize SLAM system.

        Args:
            config: ConfigManager instance
        """
        self.config = config
        self.enabled = config.get('slam.enabled', True)

        # Camera intrinsics (estimated from config)
        self.fx = config.get('mapping3d.fx', 500)
        self.fy = config.get('mapping3d.fy', 500)
        self.cx = config.get('camera.width', 320) / 2
        self.cy = config.get('camera.height', 320) / 2

        # Camera matrix and inverse
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        self.K_inv = np.linalg.inv(self.K)  # For back-projection

        # ORB feature detector
        nfeatures = config.get('slam.orb_features', 2000)  # Configurable number of features
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=config.get('slam.scale_factor', 1.2),
            nlevels=config.get('slam.nlevels', 8),
            edgeThreshold=config.get('slam.edge_threshold', 10),
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=config.get('slam.fast_threshold', 20)
        )

        # Feature matcher with multiple strategies
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # FLANN matcher for better performance with many features
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
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

        # Parameters for accuracy
        self.min_matches = config.get('slam.min_matches', 20)
        self.min_tracked_features = config.get('slam.min_tracked_features', 15)
        self.keyframe_threshold = config.get('slam.keyframe_threshold', 25)
        self.scale_threshold = config.get('slam.scale_threshold', 0.1)
        self.reprojection_threshold = config.get('slam.reprojection_threshold', 3.0)
        self.min_depth = config.get('slam.min_depth', 0.1)
        self.max_depth = config.get('slam.max_depth', 10.0)
        self.depth_variance = config.get('slam.depth_variance', 0.05)  # For uncertainty modeling

        # Motion model for pose prediction (no IMU needed)
        self.velocity = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]
        self.last_pose_update_time = time.time()

        # Pose smoothing parameters
        self.pose_alpha = config.get('slam.pose_alpha', 0.7)  # More aggressive smoothing for stability
        self.smoothed_pose = np.eye(4, dtype=np.float32)

        # Loop closure detection
        self.enable_loop_closure = config.get('slam.loop_closure', False)
        self.loop_closure_threshold = config.get('slam.loop_closure_threshold', 0.5)
        self.keyframe_database = []  # For loop closure detection

        # Bundle adjustment (simplified)
        self.enable_bundle_adjustment = config.get('slam.bundle_adjustment', False)
        self.ba_interval = config.get('slam.ba_interval', 10)  # Perform BA every N keyframes

        # Map saving
        self.map_save_dir = "data/maps"
        os.makedirs(self.map_save_dir, exist_ok=True)

        # Position and trajectory history for navigation
        self.position_history = deque(maxlen=1000)  # Longer history for trajectory analysis
        self.pose_history = deque(maxlen=100)  # Keep last 100 poses for analysis
        self.relative_poses = deque(maxlen=20)  # Store relative poses for motion analysis

        # Feature tracking statistics
        self.frame_count = 0
        self.total_features = 0
        self.tracked_features = 0
        self.lost_frames = 0
        
        # Temporal consistency checks
        self.temporal_consistency_check = config.get('slam.temporal_consistency_check', True)
        self.max_position_jump = config.get('slam.max_position_jump', 0.5)  # meters
        self.max_rotation_jump = config.get('slam.max_rotation_jump', 0.5)  # radians

        # Statistics
        self.last_positions = deque(maxlen=5)  # For velocity estimation
        self.velocity_history = deque(maxlen=10)  # For velocity smoothing

        logging.info("Monocular SLAM initialized (camera-only, no IMU)")
        logging.info(f"Camera matrix: fx={self.fx}, fy={self.fy}, cx={self.cx:.1f}, cy={self.cy:.1f}")
        logging.info(f"ORB features: {nfeatures}, Min matches: {self.min_matches}")
        logging.info(f"Temporal consistency: {self.temporal_consistency_check}")

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
                - relative_movement: [dx, dy, dz, drx, dry, drz]
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
                # If already initialized, try to predict pose based on motion model
                predicted_pose = self._predict_pose(0.1)  # Assume 0.1s since last frame
                position = predicted_pose[:3, 3].tolist()
                
                # Calculate relative movement
                relative_movement = np.zeros(6)
                if len(self.position_history) > 0:
                    prev_pos = self.position_history[-1]
                    translation = np.array(position) - prev_pos
                    rotation = np.zeros(3)  # Assume no rotation if no tracking
                    relative_movement = np.concatenate([translation, rotation])
                
                return {
                    'pose': predicted_pose,
                    'position': position,
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
        self.position_history.append(position.copy())
        self.pose_history.append(self.current_pose.copy())
        
        # Calculate relative movement from previous pose
        relative_movement = np.zeros(6)
        if len(self.pose_history) > 1:
            prev_pose = self.pose_history[-2]
            curr_pose = self.current_pose
            
            # Calculate relative transformation
            rel_transform = np.linalg.inv(prev_pose) @ curr_pose
            
            # Extract translation
            translation = rel_transform[:3, 3]
            
            # Extract rotation using Rodrigues formula
            rotation_matrix = rel_transform[:3, :3]
            rotation_vec = cv2.Rodrigues(rotation_matrix)[0].flatten()
            
            relative_movement = np.concatenate([translation, rotation_vec])
        
        # Update result with relative movement
        if not np.all(np.isfinite(relative_movement)):
            logging.warning(f"Non-finite relative movement calculated: {relative_movement}. Resetting to zero.")
            relative_movement = np.zeros(6)

        result['relative_movement'] = relative_movement.tolist()
        result['position'] = position.tolist()
        
        # Update velocity estimation
        current_time = time.time()
        if len(self.position_history) > 1 and (current_time - self.last_pose_update_time) > 0.001:
            self._update_velocity(current_time - self.last_pose_update_time)
            self.last_pose_update_time = current_time

        return result

    def _initialize(self, frame: np.ndarray, keypoints: List, descriptors: np.ndarray,
                    depth_map: Optional[np.ndarray] = None) -> Dict:
        """
        Initialize SLAM with first frame.
        """
        logging.info("Initializing SLAM with first frame...")

        # Create initial keyframe at origin
        initial_pose = np.eye(4, dtype=np.float32)
        keyframe = KeyFrame(
            id=self.next_keyframe_id,
            timestamp=time.time(),
            pose=initial_pose,
            keypoints=keypoints,
            descriptors=descriptors
        )
        self.keyframes.append(keyframe)
        self.next_keyframe_id += 1

        # Initialize map points with proper depth
        for kp, desc in zip(keypoints[:500], descriptors[:500]):
            depth = 2.0  # Default depth
            # Use depth map if available for initialization
            if depth_map is not None:
                u, v = int(kp.pt[0]), int(kp.pt[1])
                if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                    depth_norm = depth_map[v, u]
                    estimated_depth = depth_norm * 5.0  # Convert to meters (assuming max 5m)
                    if np.isfinite(estimated_depth) and estimated_depth > 0.1:
                        depth = estimated_depth
                    else:
                        logging.debug(f"Invalid depth value ({estimated_depth}) at ({u},{v}), using default.")

            point_3d = self._pixel_to_3d(kp.pt, depth=depth)
            if not np.all(np.isfinite(point_3d)):
                logging.warning(f"Failed to create valid 3D point for kp at {kp.pt}. Skipping.")
                continue

            map_point = MapPoint(
                id=self.next_point_id,
                position=point_3d,
                descriptor=desc,
                observations=1
            )
            self.map_points[self.next_point_id] = map_point
            self.next_point_id += 1

        self.is_initialized = True
        self.position_history.append([0, 0, 0])

        logging.info(f"SLAM initialized with {len(self.map_points)} map points")

        return {
            'pose': initial_pose,
            'position': [0, 0, 0],
            'tracking_quality': 1.0,
            'num_matches': len(keypoints),
            'is_keyframe': True,
            'num_map_points': len(self.map_points)
        }

    def _track(self, frame: np.ndarray, keypoints: List, descriptors: np.ndarray,
               depth_map: Optional[np.ndarray] = None) -> Dict:
        """
        Simplified track camera motion by matching features with previous frame.
        Includes temporal consistency checks, multiple matcher strategies, and pose estimation.
        """
        # Try multiple matcher strategies
        try:
            # Try FLANN matcher first (better performance with many features)
            matches = self.flann.knnMatch(descriptors, self.last_descriptors, k=2)
        except:
            # Fallback to brute force matcher
            matches = self.matcher.knnMatch(descriptors, self.last_descriptors, k=2)

        # Apply Lowe's ratio test with adaptive threshold
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # Adaptive threshold based on image content
                adaptive_threshold = 0.75
                if m.distance < adaptive_threshold * n.distance:
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
                # Partial tracking - blend prediction with motion model
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

                # Estimate essential matrix with RANSAC and better parameters
                E, mask = cv2.findEssentialMat(
                    pts_current, pts_last, self.K,
                    method=cv2.RANSAC, 
                    prob=0.999, 
                    threshold=1.0  # Was 0.8, increased for more stability
                )

                if E is not None and mask is not None:
                    # Count inliers
                    inliers = np.sum(mask)
                    
                    if inliers >= self.min_tracked_features:  # Use a configurable threshold
                        # Recover pose from essential matrix
                        success, R, t, mask_pose = cv2.recoverPose(
                            E, pts_current, pts_last, self.K, mask=mask
                        )

                        if success and t is not None and np.all(np.isfinite(t)):
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

                            # Apply temporal consistency check
                            if self.temporal_consistency_check:
                                if self._is_pose_consistent(new_pose):
                                    self.current_pose = new_pose
                                    # Apply pose smoothing
                                    self.current_pose = self._smooth_pose(new_pose)
                                else:
                                    # Use prediction instead of inconsistent pose
                                    self.current_pose = predicted_pose
                                    tracking_quality = 0.4  # Lower quality for inconsistent tracking
                            else:
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

                            # Tracking quality based on inliers and matches
                            tracking_quality = min(1.0, (inliers / self.min_matches) * 0.7 + 
                                                 (len(good_matches) / (self.min_matches * 2)) * 0.3)
                        else:
                            # Pose recovery failed
                            logging.warning(f"Pose recovery failed. Success: {success}, t: {t}. Inliers: {inliers}")
                            if E is not None:
                                logging.debug(f"Essential matrix:\n{E}")
                            self.current_pose = predicted_pose
                            tracking_quality = 0.3
                    else:
                        # Not enough inliers
                        logging.debug(f"Not enough inliers ({inliers}) to recover pose.")
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
            'num_map_points': len(self.map_points)
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
        default_scale = 0.1
        
        if len(pts_current) < 3 or not np.all(np.isfinite(translation)):
            return default_scale

        # Calculate distances in current frame
        current_depths = []
        for pt in pts_current[:20]:  # Use more points for better estimate
            u, v = int(pt[0]), int(pt[1])
            if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                depth_norm = depth_map[v, u]
                depth = depth_norm * self.max_depth
                if self.min_depth <= depth <= self.max_depth and np.isfinite(depth):
                    current_depths.append(depth)
        
        if not current_depths:
            return default_scale

        avg_depth = np.mean(current_depths)
        translation_norm = np.linalg.norm(translation)

        if avg_depth > 0 and translation_norm > 1e-6:
            # Calculate expected translation based on depth
            expected_translation = translation_norm * avg_depth
            # Use this to estimate scale
            scale = np.clip(expected_translation, 0.01, 0.5) # Limit scale to reasonable range
            return scale

        return default_scale

    def _is_pose_consistent(self, new_pose: np.ndarray) -> bool:
        """
        Check if the new pose is temporally consistent with previous poses.
        
        Args:
            new_pose: New 4x4 pose matrix
            
        Returns:
            True if pose is consistent, False otherwise
        """
        if len(self.pose_history) < 2:
            return True  # Not enough history to check consistency

        # Get previous pose
        prev_pose = self.pose_history[-1]
        
        # Calculate relative transformation
        try:
            rel_transform = np.linalg.inv(prev_pose) @ new_pose
        except np.linalg.LinAlgError:
            logging.warning("Failed to compute inverse of previous pose, assuming inconsistency.")
            return False

        if not np.all(np.isfinite(rel_transform)):
            logging.warning("Non-finite relative transform, assuming inconsistency.")
            return False
        
        # Check translation magnitude
        translation = rel_transform[:3, 3]
        trans_magnitude = np.linalg.norm(translation)
        
        # Check rotation magnitude
        rotation_matrix = rel_transform[:3, :3]
        # Use Rodrigues to get rotation angle
        try:
            rotation_vec = cv2.Rodrigues(rotation_matrix)[0]
        except cv2.error:
            logging.warning("Failed to compute Rodrigues vector for consistency check.")
            return False

        if not np.all(np.isfinite(rotation_vec)):
            logging.warning("Non-finite rotation vector, assuming inconsistency.")
            return False

        rot_magnitude = np.linalg.norm(rotation_vec)
        
        # Check if the movement is within reasonable bounds
        is_consistent = (trans_magnitude <= self.max_position_jump and 
                        rot_magnitude <= self.max_rotation_jump)
        
        if not is_consistent:
            logging.debug(f"Inconsistent movement detected: trans={trans_magnitude:.3f}, rot={rot_magnitude:.3f}")

        return is_consistent

    def _should_insert_keyframe(self, num_matches: int, tracking_quality: float) -> bool:
        """
        Decide whether to insert a new keyframe based on multiple criteria.
        """
        if not np.isfinite(tracking_quality):
            logging.warning(f"Invalid tracking quality ({tracking_quality}), skipping keyframe insertion check.")
            return False

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
            try:
                last_pos = np.array(self.position_history[-1])
                prev_pos = np.array(self.position_history[-2])
                if np.all(np.isfinite(last_pos)) and np.all(np.isfinite(prev_pos)):
                    displacement = np.linalg.norm(last_pos - prev_pos)
                    if displacement > 0.3:  # 30cm movement threshold
                        return True
            except IndexError:
                pass  # Not enough history

        # Insert every N frames if conditions are met
        if self.frame_count % self.keyframe_threshold == 0:
            return True

        return False

    def _insert_keyframe(self, frame: np.ndarray, keypoints: List, descriptors: np.ndarray, tracking_quality: float):
        """
        Insert a new keyframe with mapping and quality assessment.
        """
        keyframe = KeyFrame(
            id=self.next_keyframe_id,
            timestamp=time.time(),
            pose=self.current_pose.copy(),
            keypoints=keypoints,
            descriptors=descriptors,
            tracked_features=len(keypoints),
            feature_quality=tracking_quality,
            motion_consistency=self._calculate_motion_consistency()
        )
        self.keyframes.append(keyframe)
        
        # Add to keyframe database for potential loop closure
        if self.enable_loop_closure:
            self.keyframe_database.append(keyframe)

        self.next_keyframe_id += 1

        # Create new map points with depth information if available
        for kp, desc in zip(keypoints[:50], descriptors[:50]):  # Reduced to avoid overloading
            # Estimate depth if possible
            avg_depth = 2.0  # Default depth assumption
            point_3d = self._pixel_to_3d(kp.pt, depth=avg_depth, pose=self.current_pose)

            if not np.all(np.isfinite(point_3d)):
                logging.warning(f"Skipping non-finite 3D point for keypoint at {kp.pt}")
                continue

            map_point = MapPoint(
                id=self.next_point_id,
                position=point_3d,
                descriptor=desc,
                observations=1,
                reprojection_error=float('inf'),
                last_observed_frame=self.frame_count
            )
            self.map_points[self.next_point_id] = map_point
            self.next_point_id += 1

        logging.info(f"Inserted keyframe {keyframe.id} at position {self.current_pose[:3, 3]}, "
                    f"quality: {tracking_quality:.2f}")

    def _calculate_motion_consistency(self) -> float:
        """
        Calculate motion consistency based on recent poses.
        
        Returns:
            Consistency score (0-1) where 1 is very consistent
        """
        if len(self.pose_history) < 3:
            return 1.0
            
        # Calculate average movement between consecutive poses
        displacements = []
        for i in range(1, len(self.pose_history)):
            pos1 = self.pose_history[i-1][:3, 3]
            pos2 = self.pose_history[i][:3, 3]
            if np.all(np.isfinite(pos1)) and np.all(np.isfinite(pos2)):
                disp = np.linalg.norm(pos2 - pos1)
                if np.isfinite(disp):
                    displacements.append(disp)
            
        if len(displacements) < 2:
            return 1.0
            
        # Calculate consistency as inverse of variance (lower variance = more consistent)
        avg_disp = np.mean(displacements)
        if avg_disp < 1e-6:
            return 1.0
            
        variance = np.var(displacements)
        consistency = 1.0 / (1.0 + variance / avg_disp)
        
        return np.clip(consistency, 0.0, 1.0)  # Clamp between 0 and 1

    def _pixel_to_3d(self, pixel: Tuple[float, float], depth: float, pose: np.ndarray = None) -> np.ndarray:
        """
        Convert pixel coordinates to 3D point.

        Args:
            pixel: (u, v) pixel coordinates
            depth: Depth value
            pose: Optional camera pose (defaults to identity)

        Returns:
            3D point [x, y, z] in world coordinates
        """
        if pose is None:
            pose = np.eye(4, dtype=np.float32)

        u, v = pixel

        # Back-project to camera coordinates
        x_cam = (u - self.cx) * depth / self.fx
        y_cam = (v - self.cy) * depth / self.fy
        z_cam = depth

        # Transform to world coordinates
        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        point_world = pose @ point_cam

        return point_world[:3]

    def get_position(self) -> np.ndarray:
        """Get current camera position in world coordinates."""
        return self.current_pose[:3, 3]

    def get_orientation(self) -> np.ndarray:
        """Get current camera orientation (rotation matrix)."""
        return self.current_pose[:3, :3]

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

    def save_map(self, filename: str = None):
        """
        Save map to file.

        Args:
            filename: Output filename (default: auto-generated)
        """
        if filename is None:
            timestamp = int(time.time())
            filename = f"map_{timestamp}.json"

        filepath = os.path.join(self.map_save_dir, filename)

        # Prepare data
        map_data = {
            'timestamp': time.time(),
            'num_keyframes': len(self.keyframes),
            'num_map_points': len(self.map_points),
            'keyframes': [kf.to_dict() for kf in self.keyframes],
            'map_points': [mp.to_dict() for mp in self.map_points.values()],
            'camera_matrix': self.K.tolist()
        }

        # Save
        with open(filepath, 'w') as f:
            json.dump(map_data, f, indent=2)

        logging.info(f"Map saved to {filepath}")
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

            # Reconstruct map points
            self.map_points = {}
            for mp_data in map_data['map_points']:
                mp = MapPoint(
                    id=mp_data['id'],
                    position=np.array(mp_data['position']),
                    descriptor=np.array(mp_data['descriptor'], dtype=np.uint8),
                    observations=mp_data['observations']
                )
                self.map_points[mp.id] = mp

            # Reconstruct keyframes
            self.keyframes = []
            for kf_data in map_data['keyframes']:
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
                    descriptors=np.array(kf_data['descriptors'], dtype=np.uint8)
                )
                self.keyframes.append(kf)

            self.is_initialized = True
            self.next_point_id = max(self.map_points.keys()) + 1 if self.map_points else 0
            self.next_keyframe_id = len(self.keyframes)

            logging.info(f"Map loaded: {len(self.keyframes)} keyframes, {len(self.map_points)} points")
            return True

        except Exception as e:
            logging.error(f"Failed to load map: {e}")
            return False

    def _predict_pose(self, dt: float) -> np.ndarray:
        """
        Predict next pose using constant velocity motion model.
        No IMU needed - uses previous motion.

        Args:
            dt: Time since last update

        Returns:
            Predicted 4x4 pose matrix
        """
        if not np.all(np.isfinite(self.velocity)):
            logging.warning("Non-finite velocity in prediction, returning current pose.")
            return self.current_pose.copy()

        # Predict translation
        predicted_pose = self.current_pose.copy()
        translation_delta = self.velocity[:3] * dt
        predicted_pose[:3, 3] += translation_delta

        # Predict rotation (small angle approximation)
        angular_velocity_norm = np.linalg.norm(self.velocity[3:])
        if angular_velocity_norm > 1e-6:
            angle = angular_velocity_norm * dt
            axis = self.velocity[3:] / angular_velocity_norm

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
        if dt < 0.001:
            return

        # Calculate linear velocity
        current_pos = self.current_pose[:3, 3]
        if len(self.position_history) >= 2:
            prev_pos = np.array(self.position_history[-2])

            if not np.all(np.isfinite(current_pos)) or not np.all(np.isfinite(prev_pos)):
                logging.warning("Non-finite position in velocity update, skipping.")
                return
            
            linear_vel = (current_pos - prev_pos) / dt

            if not np.all(np.isfinite(linear_vel)):
                logging.warning(f"Non-finite linear velocity calculated, resetting to zero.")
                linear_vel = np.zeros(3)

            # Exponential moving average for smoothing
            alpha = 0.3
            self.velocity[:3] = alpha * linear_vel + (1 - alpha) * self.velocity[:3]
            
            # Add to velocity history for additional smoothing
            self.velocity_history.append(self.velocity[:3].copy())

        # Calculate angular velocity from rotation changes
        if len(self.pose_history) >= 2:
            prev_rot = self.pose_history[-2][:3, :3]
            curr_rot = self.current_pose[:3, :3]

            if not np.all(np.isfinite(prev_rot)) or not np.all(np.isfinite(curr_rot)):
                logging.warning("Non-finite rotation in velocity update, skipping.")
                return
            
            # Compute rotation difference
            rot_diff = np.dot(curr_rot, prev_rot.T)
            try:
                rot_vec = cv2.Rodrigues(rot_diff)[0].flatten()
            except cv2.error:
                logging.warning("Failed to compute Rodrigues vector, skipping angular velocity update.")
                return
            
            angular_vel = rot_vec / dt

            if not np.all(np.isfinite(angular_vel)):
                logging.warning("Non-finite angular velocity calculated, resetting to zero.")
                angular_vel = np.zeros(3)
            
            # Apply smoothing to angular velocity as well
            self.velocity[3:] = alpha * angular_vel + (1 - alpha) * self.velocity[3:]

    def _smooth_pose(self, new_pose: np.ndarray) -> np.ndarray:
        """
        Apply pose smoothing with rotation handling.
        Reduces jitter without IMU.

        Args:
            new_pose: Newly estimated pose

        Returns:
            Smoothed pose
        """
        # Smooth translation using exponential moving average
        self.smoothed_pose[:3, 3] = (self.pose_alpha * new_pose[:3, 3] +
                                    (1 - self.pose_alpha) * self.smoothed_pose[:3, 3])

        # Rotation smoothing using quaternion interpolation approach
        R_new = new_pose[:3, :3]
        R_old = self.smoothed_pose[:3, :3]
        
        # Compute rotation difference
        R_diff = np.dot(R_new, R_old.T)

        if not np.all(np.isfinite(R_diff)):
            logging.warning("Non-finite rotation difference in pose smoothing. Returning unsmoothed pose.")
            return new_pose
        
        # Convert to axis-angle representation and blend
        angle, axis = self._matrix_to_axis_angle(R_diff)

        if not np.isfinite(angle) or not np.all(np.isfinite(axis)):
            logging.warning("Non-finite angle or axis in pose smoothing. Returning unsmoothed pose.")
            return new_pose
        
        # Apply blending to the rotation
        blended_angle = angle * self.pose_alpha
        R_blend = self._axis_angle_to_matrix(axis, blended_angle)
        
        # Apply to old rotation to get the smoothed rotation
        self.smoothed_pose[:3, :3] = np.dot(R_blend, R_old)

        return self.smoothed_pose.copy()
    
    def _matrix_to_axis_angle(self, R: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Convert rotation matrix to axis-angle representation.
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            Tuple of (angle, axis)
        """
        # Calculate rotation angle
        angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
        
        if np.isclose(angle, 0):
            return 0.0, np.array([1, 0, 0])  # Arbitrarily return x-axis
        
        # Calculate rotation axis
        axis = np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]])
        
        # Normalize the axis
        norm = np.linalg.norm(axis)
        if np.isclose(norm, 0):
            return angle, np.array([1, 0, 0]) # Return default axis
        axis = axis / norm
        
        return angle, axis
    
    def _axis_angle_to_matrix(self, axis: np.ndarray, angle: float) -> np.ndarray:
        """
        Convert axis-angle representation to rotation matrix (Rodrigues formula).
        
        Args:
            axis: 3D rotation axis (normalized)
            angle: Rotation angle in radians
            
        Returns:
            3x3 rotation matrix
        """
        norm = np.linalg.norm(axis)
        if np.isclose(norm, 0):
            return np.eye(3) # Return identity matrix for zero axis

        axis = axis / norm  # Ensure normalized
        
        K = np.array([[0, -axis[2], axis[1]],
                     [axis[2], 0, -axis[0]],
                     [-axis[1], axis[0], 0]])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        return R

    def reset(self):
        """Reset SLAM to initial state."""
        self.map_points = {}
        self.keyframes = []
        self.current_pose = np.eye(4, dtype=np.float32)
        self.next_point_id = 0
        self.next_keyframe_id = 0
        self.is_initialized = False
        self.position_history.clear()
        self.velocity = np.zeros(6)
        self.smoothed_pose = np.eye(4, dtype=np.float32)
        logging.info("SLAM reset")

    def get_movement_summary(self) -> Dict:
        """Get summary of movement and trajectory for visualization."""
        if not self.position_history:
            return self._empty_movement_summary()

        positions = [p for p in self.position_history if np.all(np.isfinite(p))]
        
        if len(positions) < 2:
            return self._empty_movement_summary(trajectory_length=len(positions))

        # Calculate total distance traveled along trajectory
        total_distance = 0.0
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i-1])
            total_distance += dist

        # Calculate displacement (straight-line distance from start to end)
        displacement_vector = positions[-1] - positions[0]
        displacement_magnitude = np.linalg.norm(displacement_vector)

        # Calculate current speed (if enough history)
        last_pos = positions[-1]
        prev_pos = positions[-2]
        displacement = np.linalg.norm(last_pos - prev_pos)
        # Assuming regular frame rate, we can estimate speed
        current_speed = displacement * 10  # Rough estimate at 10 FPS

        # Calculate average speed
        avg_speed = total_distance / (len(positions) -1) if len(positions) > 1 else 0.0

        return {
            'total_distance': total_distance,
            'average_speed': avg_speed,
            'current_speed': current_speed,
            'trajectory_length': len(positions),
            'displacement_vector': displacement_vector.tolist(),
            'displacement_magnitude': displacement_magnitude
        }

    def _empty_movement_summary(self, trajectory_length: int = 0) -> Dict:
        return {
            'total_distance': 0.0,
            'average_speed': 0.0,
            'current_speed': 0.0,
            'trajectory_length': trajectory_length,
            'displacement_vector': [0.0, 0.0, 0.0],
            'displacement_magnitude': 0.0
        }

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
        Simplified visualization of SLAM tracking on frame with movement information.

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

        return vis_frame

    def visualize_3d_map(self) -> np.ndarray:
        """
        Create a 3D visualization of the current SLAM map.
        
        Returns:
            Annotated frame showing 3D map
        """
        # Create a visualization canvas
        canvas_size = 600
        vis_map = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 240  # Light gray background
        
        # Draw coordinate grid
        center_x, center_y = canvas_size // 2, canvas_size // 2
        scale = 20  # pixels per meter
        
        # Draw grid lines
        for i in range(-10, 11):
            # Vertical lines
            x = int(center_x + i * scale)
            if 0 <= x < canvas_size:
                cv2.line(vis_map, (x, 0), (x, canvas_size), (220, 220, 220), 1)
            # Horizontal lines  
            y = int(center_y + i * scale)
            if 0 <= y < canvas_size:
                cv2.line(vis_map, (0, y), (canvas_size, y), (220, 220, 220), 1)
        
        # Draw center cross
        cv2.line(vis_map, (center_x, 0), (center_x, canvas_size), (200, 200, 200), 2)
        cv2.line(vis_map, (0, center_y), (canvas_size, center_y), (200, 200, 200), 2)
        
        # Draw trajectory
        if len(self.position_history) > 1:
            points = []
            for i, pos in enumerate(self.position_history):
                x = int(center_x + pos[0] * scale)
                y = int(center_y - pos[1] * scale)  # Flip Y axis
                if 0 <= x < canvas_size and 0 <= y < canvas_size:
                    points.append((x, y))
            
            # Draw trajectory line
            for i in range(len(points) - 1):
                pt1 = points[i]
                pt2 = points[i+1]
                if (isinstance(pt1, tuple) and len(pt1) == 2 and 
                    isinstance(pt2, tuple) and len(pt2) == 2 and
                    all(isinstance(coord, (int, np.integer)) for coord in pt1 + pt2)):
                    alpha = i / len(points) if len(points) > 0 else 1.0
                    color = (int(200 * alpha), 100, int(255 * (1 - alpha)))  # Gradient from blue to red
                    cv2.line(vis_map, pt1, pt2, color, 2)
            
            # Draw trajectory dots (more recent positions are brighter)
            for i, point in enumerate(points):
                if (isinstance(point, tuple) and len(point) == 2 and
                    all(isinstance(coord, (int, np.integer)) for coord in point) and
                    0 <= point[0] < canvas_size and 0 <= point[1] < canvas_size):
                    alpha = i / len(points) if len(points) > 1 else 1.0
                    size = 2 if i < len(points) - 15 else 4  # Larger for recent positions
                    color = (0, int(255 * alpha), int(255 * (1 - alpha)))  # Green for recent
                    cv2.circle(vis_map, point, size, color, -1)
        
        # Draw current position
        if len(self.position_history) > 0:
            current_pos = self.position_history[-1]
            x = int(center_x + current_pos[0] * scale)
            y = int(center_y - current_pos[1] * scale)
            if 0 <= x < canvas_size and 0 <= y < canvas_size:
                cv2.circle(vis_map, (x, y), 8, (0, 255, 0), -1)  # Green
                cv2.circle(vis_map, (x, y), 10, (0, 100, 0), 2)   # Darker green border
                cv2.putText(vis_map, "YOU", (x + 12, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 2)
        
        # Draw map points if available
        if len(self.map_points) > 0:
            map_points = self.get_map_points_array()
            if map_points.shape[0] > 0:
                for i, point in enumerate(map_points[:200]):  # Limit to 200 points for performance
                    x = int(center_x + point[0] * scale)
                    y = int(center_y - point[1] * scale)
                    if 0 <= x < canvas_size and 0 <= y < canvas_size:
                        # Draw map points as smaller circles
                        cv2.circle(vis_map, (x, y), 2, (255, 0, 0), -1)  # Blue
        
        # Add title
        cv2.putText(vis_map, "SLAM 3D Map", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Add stats
        cv2.putText(vis_map, f"Map Points: {len(self.map_points)}", 
                   (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(vis_map, f"Trajectory: {len(self.position_history)}", 
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return vis_map
