"""
OrbyGlasses - Visual SLAM for Indoor Navigation
Monocular SLAM using ORB features for camera-only localization and mapping.
No IMU required - works with just a USB webcam.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import deque
import json
import os
from dataclasses import dataclass, asdict
import time


@dataclass
class MapPoint:
    """3D point in the map."""
    id: int
    position: np.ndarray  # 3D coordinates [x, y, z]
    descriptor: np.ndarray  # ORB descriptor
    observations: int = 0  # Number of times observed

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'descriptor': self.descriptor.tolist(),
            'observations': self.observations
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

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'pose': self.pose.tolist(),
            'keypoints': [(kp.pt[0], kp.pt[1], kp.size, kp.angle) for kp in self.keypoints],
            'descriptors': self.descriptors.tolist()
        }


class MonocularSLAM:
    """
    Simplified monocular SLAM for indoor navigation.
    Uses ORB features for tracking and mapping without IMU.

    This is a lightweight implementation suitable for real-time navigation
    on embedded systems. For production, consider ORB-SLAM3 with proper
    installation.
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

        # Camera matrix
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # ORB feature detector (more features, more robust)
        self.orb = cv2.ORB_create(
            nfeatures=3000,      # Increased from 2000
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=10,    # Lower edge threshold = more features
            firstLevel=0,
            WTA_K=2,
            patchSize=31
        )

        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # SLAM state
        self.map_points = {}  # id -> MapPoint
        self.keyframes = []  # List of KeyFrame
        self.current_pose = np.eye(4, dtype=np.float32)  # Current camera pose
        self.next_point_id = 0
        self.next_keyframe_id = 0

        # Tracking state
        self.last_frame = None
        self.last_keypoints = None
        self.last_descriptors = None
        self.is_initialized = False

        # Parameters (more lenient for better tracking)
        self.min_matches = 30  # Reduced from 50 for better tracking
        self.keyframe_threshold = 15  # More frequent keyframes
        self.frame_count = 0

        # Motion model for pose prediction (no IMU needed)
        self.velocity = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]
        self.last_pose_update_time = time.time()

        # Pose smoothing (exponential moving average)
        self.pose_alpha = 0.5  # Smoothing factor (higher = more responsive)
        self.smoothed_pose = np.eye(4, dtype=np.float32)

        # Map saving
        self.map_save_dir = "data/maps"
        os.makedirs(self.map_save_dir, exist_ok=True)

        # Position history for navigation
        self.position_history = deque(maxlen=100)

        logging.info("Monocular SLAM initialized (camera-only, no IMU)")
        logging.info(f"Camera matrix: fx={self.fx}, fy={self.fy}, cx={self.cx:.1f}, cy={self.cy:.1f}")

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
        """
        if not self.enabled:
            return self._empty_result()

        self.frame_count += 1

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ORB features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 20:  # Reduced threshold
            if self.is_initialized:
                # If already initialized, keep last pose rather than failing
                return {
                    'pose': self.current_pose,
                    'position': self.current_pose[:3, 3].tolist(),
                    'tracking_quality': 0.1,
                    'num_matches': 0,
                    'is_keyframe': False,
                    'num_map_points': len(self.map_points)
                }
            else:
                logging.warning(f"Insufficient features for initialization: {len(keypoints) if keypoints else 0}")
                return self._empty_result()

        # Initialize or track
        if not self.is_initialized:
            result = self._initialize(gray, keypoints, descriptors, depth_map)
        else:
            result = self._track(gray, keypoints, descriptors, depth_map)

        # Store for next frame
        self.last_frame = gray
        self.last_keypoints = keypoints
        self.last_descriptors = descriptors

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
            # Use depth map if available for accurate initialization
            if depth_map is not None:
                u, v = int(kp.pt[0]), int(kp.pt[1])
                if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                    depth_norm = depth_map[v, u]
                    depth = depth_norm * 5.0  # Convert to meters (assuming max 5m)
                    if depth < 0.1:
                        depth = 2.0  # Default depth for invalid measurements
                else:
                    depth = 2.0
            else:
                depth = 2.0  # Default depth without depth map

            point_3d = self._pixel_to_3d(kp.pt, depth=depth)
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
        Track camera motion by matching features with previous frame.
        Enhanced with motion model and pose smoothing (no IMU needed).
        """
        # Match features with last frame
        matches = self.matcher.knnMatch(descriptors, self.last_descriptors, k=2)

        # Apply Lowe's ratio test (stricter for better accuracy)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:  # Stricter (was 0.75)
                    good_matches.append(m)

        num_matches = len(good_matches)

        # Initialize defaults
        position = self.current_pose[:3, 3].tolist()
        tracking_quality = 0.0

        # Predict pose using motion model (constant velocity assumption)
        current_time = time.time()
        dt = current_time - self.last_pose_update_time
        predicted_pose = self._predict_pose(dt)

        if num_matches < self.min_matches:
            # Use predicted pose when tracking is weak
            if num_matches >= 10:
                # Partial tracking - blend prediction with last pose
                self.current_pose = predicted_pose
                tracking_quality = num_matches / self.min_matches
            else:
                # Very weak tracking - use prediction only
                self.current_pose = predicted_pose
                tracking_quality = 0.1
                logging.debug(f"Low tracking: {num_matches} matches (using motion prediction)")
        else:
            # Good tracking - estimate pose from features
            pts_current = np.float32([keypoints[m.queryIdx].pt for m in good_matches])
            pts_last = np.float32([self.last_keypoints[m.trainIdx].pt for m in good_matches])

            # Estimate essential matrix with RANSAC
            E, mask = cv2.findEssentialMat(pts_current, pts_last, self.K,
                                          method=cv2.RANSAC, prob=0.999, threshold=0.8)

            if E is not None and mask is not None:
                # Recover pose from essential matrix
                _, R, t, mask_pose = cv2.recoverPose(E, pts_current, pts_last, self.K, mask=mask)

                # IMPORTANT: Fix monocular scale using realistic assumptions
                # Default scale for monocular SLAM (assume ~0.1m movement per frame at normal speed)
                scale = 0.1  # meters per frame (baseline scale)

                # If we have velocity history, use it for scale
                if len(self.position_history) >= 3:
                    # Estimate scale from recent velocity
                    vel_magnitude = np.linalg.norm(self.velocity[:3])
                    if vel_magnitude > 0.01:  # Minimum movement
                        scale = vel_magnitude * dt
                        scale = np.clip(scale, 0.02, 0.5)  # Reasonable range: 2cm to 50cm per frame

                # Apply scale to translation
                t = t * scale

                # Create transformation matrix
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = R
                T[:3, 3] = t.flatten()

                # Update pose
                new_pose = self.current_pose @ T

                # Apply pose smoothing (exponential moving average)
                self.current_pose = self._smooth_pose(new_pose)

                # Update velocity for motion model
                self._update_velocity(dt)
                self.last_pose_update_time = current_time

                # Extract position
                position = self.current_pose[:3, 3].tolist()
                self.position_history.append(position)

                # Tracking quality based on inliers
                tracking_quality = np.sum(mask_pose) / len(good_matches)
            else:
                # Essential matrix failed - use prediction
                self.current_pose = predicted_pose
                tracking_quality = 0.3

        # Check if we need a new keyframe
        is_keyframe = self._should_insert_keyframe(num_matches)

        if is_keyframe:
            self._insert_keyframe(frame, keypoints, descriptors)

        return {
            'pose': self.current_pose,
            'position': position,
            'tracking_quality': tracking_quality,
            'num_matches': num_matches,
            'is_keyframe': is_keyframe,
            'num_map_points': len(self.map_points)
        }

    def _should_insert_keyframe(self, num_matches: int) -> bool:
        """
        Decide whether to insert a new keyframe.
        """
        # Insert keyframe every N frames or if tracking quality drops
        if self.frame_count % self.keyframe_threshold == 0:
            return True

        if num_matches < self.min_matches * 1.5:
            return True

        # Check if camera moved significantly
        if len(self.position_history) >= 2:
            last_pos = np.array(self.position_history[-1])
            prev_pos = np.array(self.position_history[-2])
            displacement = np.linalg.norm(last_pos - prev_pos)

            if displacement > 0.5:  # 0.5 meters
                return True

        return False

    def _insert_keyframe(self, frame: np.ndarray, keypoints: List, descriptors: np.ndarray):
        """
        Insert a new keyframe and create new map points.
        """
        keyframe = KeyFrame(
            id=self.next_keyframe_id,
            timestamp=time.time(),
            pose=self.current_pose.copy(),
            keypoints=keypoints,
            descriptors=descriptors
        )
        self.keyframes.append(keyframe)
        self.next_keyframe_id += 1

        # Create new map points (simplified - assume depth from last known depth)
        avg_depth = 2.0  # Default depth assumption
        for kp, desc in zip(keypoints[:100], descriptors[:100]):
            point_3d = self._pixel_to_3d(kp.pt, depth=avg_depth, pose=self.current_pose)
            map_point = MapPoint(
                id=self.next_point_id,
                position=point_3d,
                descriptor=desc,
                observations=1
            )
            self.map_points[self.next_point_id] = map_point
            self.next_point_id += 1

        logging.info(f"Inserted keyframe {keyframe.id} at position {self.current_pose[:3, 3]}")

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
        # Predict translation
        predicted_pose = self.current_pose.copy()
        translation_delta = self.velocity[:3] * dt
        predicted_pose[:3, 3] += translation_delta

        # Predict rotation (small angle approximation)
        if np.linalg.norm(self.velocity[3:]) > 0.01:
            angle = np.linalg.norm(self.velocity[3:]) * dt
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
        if len(self.position_history) < 2 or dt < 0.001:
            return

        # Linear velocity
        current_pos = self.current_pose[:3, 3]
        if len(self.position_history) >= 2:
            prev_pos = np.array(self.position_history[-2])
            linear_vel = (current_pos - prev_pos) / dt

            # Exponential moving average for smoothing
            alpha = 0.3
            self.velocity[:3] = alpha * linear_vel + (1 - alpha) * self.velocity[:3]

    def _smooth_pose(self, new_pose: np.ndarray) -> np.ndarray:
        """
        Apply exponential moving average to smooth pose.
        Reduces jitter without IMU.

        Args:
            new_pose: Newly estimated pose

        Returns:
            Smoothed pose
        """
        # Smooth translation
        smoothed = self.smoothed_pose.copy()
        smoothed[:3, 3] = (self.pose_alpha * new_pose[:3, 3] +
                          (1 - self.pose_alpha) * self.smoothed_pose[:3, 3])

        # Smooth rotation using SLERP-like interpolation
        # Simple linear interpolation of rotation matrices
        smoothed[:3, :3] = (self.pose_alpha * new_pose[:3, :3] +
                           (1 - self.pose_alpha) * self.smoothed_pose[:3, :3])

        # Re-orthogonalize rotation matrix
        U, _, Vt = np.linalg.svd(smoothed[:3, :3])
        smoothed[:3, :3] = U @ Vt

        self.smoothed_pose = smoothed
        return smoothed

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

    def _empty_result(self) -> Dict:
        """Return empty result when SLAM is disabled or fails."""
        return {
            'pose': np.eye(4, dtype=np.float32),
            'position': [0, 0, 0],
            'tracking_quality': 0.0,
            'num_matches': 0,
            'is_keyframe': False,
            'num_map_points': 0
        }

    def visualize_tracking(self, frame: np.ndarray, tracking_result: Dict) -> np.ndarray:
        """
        Visualize SLAM tracking on frame.

        Args:
            frame: Input frame
            tracking_result: Result from process_frame()

        Returns:
            Annotated frame
        """
        vis_frame = frame.copy()

        # Draw features
        if self.last_keypoints:
            for kp in self.last_keypoints[:100]:  # Draw top 100
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(vis_frame, (x, y), 2, (0, 255, 0), -1)

        # Draw tracking info
        h, w = frame.shape[:2]

        # Background overlay
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (5, h - 100), (300, h - 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)

        # Text info
        position = tracking_result['position']
        quality = tracking_result['tracking_quality']

        cv2.putText(vis_frame, f"SLAM Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})",
                   (10, h - 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        quality_color = (0, 255, 0) if quality > 0.7 else (0, 165, 255) if quality > 0.4 else (0, 0, 255)
        cv2.putText(vis_frame, f"Tracking Quality: {quality:.2f}",
                   (10, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, quality_color, 1)

        cv2.putText(vis_frame, f"Matches: {tracking_result['num_matches']}",
                   (10, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.putText(vis_frame, f"Map Points: {tracking_result['num_map_points']}",
                   (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Keyframe indicator
        if tracking_result['is_keyframe']:
            cv2.putText(vis_frame, "KEYFRAME", (w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return vis_frame
