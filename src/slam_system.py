"""
SLAM for OrbyGlasses
Camera-only SLAM with accuracy and performance
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import deque
import time
from dataclasses import dataclass


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
    keyframe_ref: int = -1 # Reference to the keyframe where it was created


@dataclass
class KeyFrame:
    """Key frame with pose and features."""
    id: int
    timestamp: float
    pose: np.ndarray  # 4x4 transformation matrix
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    frame: Optional[np.ndarray] = None  # Optional: store frame for visualization
    tracked_features: int = 0  # Number of successfully tracked features
    feature_quality: float = 0.0  # Quality score of features
    motion_consistency: float = 0.0  # How consistent this frame's motion is


class SLAMSystem:
    """
    SLAM with accuracy for navigation without IMU.
    Uses feature tracking, temporal consistency, and pose estimation.
    """
    
    def __init__(self, config):
        """
        Initialize SLAM system.

        Args:
            config: ConfigManager instance
        """
        self.config = config
        self.enabled = config.get('slam.enabled', True)

        # Camera intrinsics
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
        
        self.K_inv = np.linalg.inv(self.K)

        # Feature detector with parameters
        nfeatures = config.get('slam.orb_features', 3000)
        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=1.1,  # Smaller scale factor for more precise features
            nlevels=16,       # More levels for better scale invariance
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=10,  # Lower threshold for more features
            scoreType=cv2.ORB_HARRIS_SCORE
        )

        # Multiple matcher strategies
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # FLANN matcher for performance with many features
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=12, key_size=20, multi_probe_level=2)
        search_params = dict(checks=100)
        self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # SLAM state
        self.map_points = {}  # id -> MapPoint
        self.keyframes = []   # List of KeyFrame
        self.current_pose = np.eye(4, dtype=np.float32)
        self.previous_pose = np.eye(4, dtype=np.float32)
        self.next_point_id = 0
        self.next_keyframe_id = 0

        # Tracking state
        self.last_frame = None
        self.last_keypoints = None
        self.last_descriptors = None
        self.is_initialized = False

        # Parameters for accuracy
        self.min_matches = config.get('slam.min_matches', 15)
        self.min_tracked_features = config.get('slam.min_tracked_features', 10)
        self.keyframe_threshold = config.get('slam.keyframe_threshold', 20)
        self.reprojection_threshold = 3.0
        self.min_depth = config.get('slam.min_depth', 0.1)
        self.max_depth = config.get('slam.max_depth', 10.0)

        # Motion model for pose prediction
        self.velocity = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]
        self.last_pose_update_time = time.time()

        # Pose smoothing parameters
        self.pose_alpha = config.get('slam.pose_alpha', 0.8)  # More aggressive smoothing for stability
        self.smoothed_pose = np.eye(4, dtype=np.float32)

        # Loop closure detection
        self.enable_loop_closure = config.get('slam.loop_closure', False)
        self.loop_closure_threshold = config.get('slam.loop_closure_threshold', 0.6)
        self.keyframe_database = []  # For loop closure detection

        # Map saving
        self.map_save_dir = "data/maps"
        import os
        os.makedirs(self.map_save_dir, exist_ok=True)

        # Position and trajectory history
        self.position_history = deque(maxlen=1000)
        self.pose_history = deque(maxlen=100)
        self.relative_poses = deque(maxlen=20)

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
        self.last_positions = deque(maxlen=5)
        self.velocity_history = deque(maxlen=10)

        logging.info("SLAM System initialized (camera-only, no IMU)")
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
            Dictionary with tracking info
        """
        if not self.enabled:
            return self._empty_result()

        self.frame_count += 1

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ORB features with parameters
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 10:
            if self.is_initialized:
                # Predict pose based on motion model
                predicted_pose = self._predict_pose(0.1)
                position = predicted_pose[:3, 3].tolist()
                
                # Calculate relative movement
                relative_movement = np.zeros(6)
                if len(self.position_history) > 0:
                    prev_pos = self.position_history[-1]
                    translation = np.array(position) - prev_pos
                    rotation = np.zeros(3)
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
            if depth_map is not None:
                u, v = int(kp.pt[0]), int(kp.pt[1])
                if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                    depth_norm = depth_map[v, u]
                    estimated_depth = depth_norm * 5.0  # Convert to meters
                    if np.isfinite(estimated_depth) and estimated_depth > 0.1:
                        depth = estimated_depth

            point_3d = self._pixel_to_3d(kp.pt, depth=depth)
            if not np.all(np.isfinite(point_3d)):
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
        Includes temporal consistency checks and pose estimation.
        """
        # Use FLANN-based matcher for better performance
        matches = self.flann_matcher.knnMatch(descriptors, self.last_descriptors, k=2)

        # Apply Lowe's ratio test to filter out bad matches
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
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
        else:
            # Good tracking - estimate pose from features
            if len(good_matches) >= 5:  # Need at least 5 points for PnP
                # Get 3D-2D correspondences
                object_points = []
                image_points = []
                for m in good_matches:
                    if m.trainIdx in self.map_points:
                        object_points.append(self.map_points[m.trainIdx].position)
                        image_points.append(keypoints[m.queryIdx].pt)

                if len(object_points) >= 5:
                    object_points = np.float32(object_points)
                    image_points = np.float32(image_points)

                    # Estimate pose with PnP
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        object_points, image_points, self.K, None,
                        iterationsCount=100, reprojectionError=self.reprojection_threshold
                    )

                    if success:
                        # Create transformation matrix
                        R, _ = cv2.Rodrigues(rvec)
                        T = np.eye(4, dtype=np.float32)
                        T[:3, :3] = R
                        T[:3, 3] = tvec.flatten()

                        # Update pose
                        self.current_pose = T

                        # Update velocity for motion model
                        if dt > 0.001:
                            self._update_velocity(dt)
                            self.last_pose_update_time = current_time

                        # Extract position
                        position = self.current_pose[:3, 3].tolist()
                        
                        # Update statistics
                        self.tracked_features += len(inliers)
                        self.total_features += len(good_matches)

                        # Tracking quality based on inliers and matches
                        tracking_quality = min(1.0, (len(inliers) / self.min_matches) * 0.7 + 
                                             (len(good_matches) / (self.min_matches * 2)) * 0.3)
                    else:
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
            return 0.1

        default_scale = 0.1
        
        if len(pts_current) < 3 or not np.all(np.isfinite(translation)):
            return default_scale

        # Use a sliding window of recent scale estimates
        if not hasattr(self, 'scale_history'):
            self.scale_history = deque(maxlen=10)

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
            if len(self.scale_history) > 0:
                return np.median(self.scale_history)
            return default_scale

        avg_depth = np.mean(current_depths)
        translation_norm = np.linalg.norm(translation)

        if avg_depth > 0 and translation_norm > 1e-6:
            # Calculate expected translation based on depth
            expected_translation = translation_norm * avg_depth
            # Use this to estimate scale
            scale = np.clip(expected_translation, 0.01, 0.5) # Limit scale to reasonable range
            self.scale_history.append(scale)
            return np.median(self.scale_history)

        if len(self.scale_history) > 0:
            return np.median(self.scale_history)

        return default_scale

    def _is_pose_consistent(self, new_pose: np.ndarray) -> bool:
        """
        Check if the new pose is temporally consistent with previous poses.
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
        if tracking_quality > 0.8 and self.frame_count % 30 == 0:
            return True

        # Insert if tracking quality is low and matches are few
        if tracking_quality < 0.4 and num_matches < self.min_matches:
            return True

        # Insert if we have enough good matches and quality is decent
        if num_matches > self.min_matches * 3 and tracking_quality > 0.6:
            return True

        # Check if camera moved significantly
        if len(self.position_history) >= 2:
            try:
                last_pos = np.array(self.position_history[-1])
                prev_pos = np.array(self.position_history[-2])
                if np.all(np.isfinite(last_pos)) and np.all(np.isfinite(prev_pos)):
                    displacement = np.linalg.norm(last_pos - prev_pos)
                    if displacement > 0.5:  # 50cm movement threshold
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

    def _predict_pose(self, dt: float) -> np.ndarray:
        """
        Predict next pose using a Kalman filter-based motion model.

        Args:
            dt: Time since last update

        Returns:
            Predicted 4x4 pose matrix
        """
        if not hasattr(self, 'kalman_filter'):
            # Initialize Kalman filter
            self.kalman_filter = cv2.KalmanFilter(18, 6, 0)
            self.kalman_filter.transitionMatrix = np.eye(18, dtype=np.float32)
            for i in range(6):
                self.kalman_filter.transitionMatrix[i, i+6] = dt
                self.kalman_filter.transitionMatrix[i+6, i+12] = dt
            self.kalman_filter.measurementMatrix = np.zeros((6, 18), dtype=np.float32)
            for i in range(6):
                self.kalman_filter.measurementMatrix[i, i] = 1
            self.kalman_filter.processNoiseCov = np.eye(18, dtype=np.float32) * 1e-5
            self.kalman_filter.measurementNoiseCov = np.eye(6, dtype=np.float32) * 1e-4
            self.kalman_filter.errorCovPost = np.eye(18, dtype=np.float32)

            # Initialize state
            self.kalman_filter.statePost = np.zeros(18, dtype=np.float32)
            self.kalman_filter.statePost[0:3] = self.current_pose[:3, 3]
            rvec, _ = cv2.Rodrigues(self.current_pose[:3, :3])
            self.kalman_filter.statePost[3:6] = rvec.flatten()

        # Predict
        predicted_state = self.kalman_filter.predict()

        # Update transition matrix with new dt
        for i in range(6):
            self.kalman_filter.transitionMatrix[i, i+6] = dt
            self.kalman_filter.transitionMatrix[i+6, i+12] = dt

        # Create predicted pose
        predicted_pose = np.eye(4, dtype=np.float32)
        predicted_pose[:3, 3] = predicted_state[0:3].flatten()
        rvec = predicted_state[3:6].flatten()
        R, _ = cv2.Rodrigues(rvec)
        predicted_pose[:3, :3] = R

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
                return
            
            linear_vel = (current_pos - prev_pos) / dt

            if not np.all(np.isfinite(linear_vel)):
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
                return
            
            # Compute rotation difference
            rot_diff = np.dot(curr_rot, prev_rot.T)
            try:
                rot_vec = cv2.Rodrigues(rot_diff)[0].flatten()
            except cv2.error:
                return
            
            angular_vel = rot_vec / dt

            if not np.all(np.isfinite(angular_vel)):
                angular_vel = np.zeros(3)
            
            # Apply smoothing to angular velocity as well
            self.velocity[3:] = alpha * angular_vel + (1 - alpha) * self.velocity[3:]

    def _smooth_pose(self, new_pose: np.ndarray) -> np.ndarray:
        """
        Apply pose smoothing with rotation handling.

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
            return new_pose
        
        # Convert to axis-angle representation and blend
        angle, axis = self._matrix_to_axis_angle(R_diff)

        if not np.isfinite(angle) or not np.all(np.isfinite(axis)):
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

    def get_movement_summary(self) -> Dict:
        """Get summary of movement and trajectory for visualization with enhanced accuracy."""
        if not self.position_history:
            return self._empty_movement_summary()

        positions = [p for p in self.position_history if np.all(np.isfinite(p))]
        
        if len(positions) < 2:
            return self._empty_movement_summary(trajectory_length=len(positions))

        # Calculate total distance traveled along trajectory with enhanced accuracy
        total_distance = 0.0
        instantaneous_speeds = []
        
        # Calculate distance and speed for each step
        for i in range(1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[i-1])
            total_distance += dist
            
            # Calculate instantaneous speed (assuming 10 FPS)
            time_delta = 0.1  # 0.1 seconds per frame at 10 FPS
            instantaneous_speeds.append(dist / time_delta)

        # Calculate displacement (straight-line distance from start to end)
        displacement_vector = positions[-1] - positions[0]
        displacement_magnitude = np.linalg.norm(displacement_vector)

        # Calculate current speed with better estimation using recent positions
        recent_positions = positions[-5:] if len(positions) >= 5 else positions  # Use last 5 positions
        if len(recent_positions) >= 2:
            # Calculate speed based on recent movements for better accuracy
            recent_distance = 0.0
            for i in range(len(recent_positions) - 1):
                recent_distance += np.linalg.norm(recent_positions[i+1] - recent_positions[i])
            # Average speed over recent positions
            time_span = len(recent_positions) * 0.1  # 0.1s per frame at 10 FPS
            current_speed = recent_distance / time_span if time_span > 0 else 0.0
        else:
            current_speed = 0.0

        # Calculate average speed with outlier filtering
        if len(instantaneous_speeds) > 0:
            # Use trimmed mean to exclude outliers
            speeds_array = np.array(instantaneous_speeds)
            if len(speeds_array) > 4:  # Only filter if we have enough data
                p10 = np.percentile(speeds_array, 10)
                p90 = np.percentile(speeds_array, 90)
                filtered_speeds = speeds_array[(speeds_array >= p10) & (speeds_array <= p90)]
                avg_speed = np.mean(filtered_speeds) if len(filtered_speeds) > 0 else np.mean(speeds_array)
            else:
                avg_speed = np.mean(speeds_array)
        else:
            avg_speed = 0.0

        # Calculate trajectory smoothness (1.0 = perfectly smooth, 0.0 = very jagged)
        if len(positions) > 2:
            # Calculate the deviation from a straight line between multiple points
            smoothness_sum = 0.0
            smoothness_count = 0
            for i in range(0, len(positions) - 2, 3):  # Check every 3rd point to reduce computation
                if i + 2 < len(positions):
                    p1, p2, p3 = positions[i], positions[i+1], positions[i+2]
                    # Calculate how much p2 deviates from the line between p1 and p3
                    line_vec = p3 - p1
                    line_length = np.linalg.norm(line_vec)
                    if line_length > 0.01:  # Avoid division by zero
                        # Project p2 onto the line between p1 and p3
                        line_unit = line_vec / line_length
                        projected = p1 + np.dot(p2 - p1, line_unit) * line_unit
                        deviation = np.linalg.norm(p2 - projected)
                        # Normalize deviation by line length to get relative smoothness
                        relative_deviation = deviation / line_length
                        # Smoothness is inverse of deviation (clamped between 0 and 1)
                        smoothness = max(0.0, min(1.0, 1.0 - 5.0 * relative_deviation))
                        smoothness_sum += smoothness
                        smoothness_count += 1
            
            trajectory_smoothness = smoothness_sum / smoothness_count if smoothness_count > 0 else 1.0
        else:
            trajectory_smoothness = 1.0

        return {
            'total_distance': total_distance,
            'average_speed': avg_speed,
            'current_speed': current_speed,
            'trajectory_length': len(positions),
            'displacement_vector': displacement_vector.tolist(),
            'displacement_magnitude': displacement_magnitude,
            'trajectory_smoothness': trajectory_smoothness,
            'max_instantaneous_speed': max(instantaneous_speeds) if instantaneous_speeds else 0.0,
            'speed_variance': np.var(instantaneous_speeds) if instantaneous_speeds else 0.0
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

        # Draw SLAM status information
        h, w = vis_frame.shape[:2]
        
        # Position
        position = tracking_result.get('position', [0, 0, 0])
        cv2.putText(vis_frame, f"Pos: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})", 
                   (w - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Quality
        quality = tracking_result.get('tracking_quality', 0.0)
        quality_color = (0, 255, 0) if quality > 0.7 else (0, 165, 255) if quality > 0.4 else (0, 0, 255)
        cv2.putText(vis_frame, f"Quality: {quality:.2f}", 
                   (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, quality_color, 1)
        
        # Number of matches
        num_matches = tracking_result.get('num_matches', 0)
        cv2.putText(vis_frame, f"Matches: {num_matches}", 
                   (w - 200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Number of map points
        num_map_points = tracking_result.get('num_map_points', 0)
        cv2.putText(vis_frame, f"Map Points: {num_map_points}", 
                   (w - 200, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Keyframe indicator
        if tracking_result.get('is_keyframe', False):
            cv2.putText(vis_frame, "KEYFRAME", 
                       (w - 200, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

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