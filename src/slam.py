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

        # Map saving
        self.map_save_dir = "data/maps"
        os.makedirs(self.map_save_dir, exist_ok=True)

        # Position history for navigation
        self.position_history = deque(maxlen=100)

        logging.info("Monocular SLAM initialized (camera-only, no IMU)")
        logging.info(f"Camera matrix: fx={self.fx}, fy={self.fy}, cx={self.cx:.1f}, cy={self.cy:.1f}")

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a new frame through SLAM pipeline.

        Args:
            frame: Input frame (BGR)

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
            result = self._initialize(gray, keypoints, descriptors)
        else:
            result = self._track(gray, keypoints, descriptors)

        # Store for next frame
        self.last_frame = gray
        self.last_keypoints = keypoints
        self.last_descriptors = descriptors

        return result

    def _initialize(self, frame: np.ndarray, keypoints: List, descriptors: np.ndarray) -> Dict:
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

        # Initialize map points (assume depth = 1.0 for first frame)
        for kp, desc in zip(keypoints[:500], descriptors[:500]):
            point_3d = self._pixel_to_3d(kp.pt, depth=1.0)
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

    def _track(self, frame: np.ndarray, keypoints: List, descriptors: np.ndarray) -> Dict:
        """
        Track camera motion by matching features with previous frame.
        """
        # Match features with last frame
        matches = self.matcher.knnMatch(descriptors, self.last_descriptors, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        num_matches = len(good_matches)

        # Initialize defaults
        position = self.current_pose[:3, 3].tolist()
        tracking_quality = 0.0

        if num_matches < self.min_matches:
            # Don't spam warnings, keep last pose
            if num_matches < 10:  # Only warn if very few matches
                logging.debug(f"Low tracking: only {num_matches} matches (keeping last pose)")
            tracking_quality = num_matches / self.min_matches  # Proportional quality
        else:
            # Extract matched points
            pts_current = np.float32([keypoints[m.queryIdx].pt for m in good_matches])
            pts_last = np.float32([self.last_keypoints[m.trainIdx].pt for m in good_matches])

            # Estimate essential matrix (assumes calibrated camera)
            E, mask = cv2.findEssentialMat(pts_current, pts_last, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

            if E is not None and mask is not None:
                # Recover pose from essential matrix
                _, R, t, mask_pose = cv2.recoverPose(E, pts_current, pts_last, self.K, mask=mask)

                # Update pose (relative motion)
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = R
                T[:3, 3] = t.flatten()

                # Update current pose
                self.current_pose = self.current_pose @ T

                # Extract position
                position = self.current_pose[:3, 3].tolist()
                self.position_history.append(position)

                # Tracking quality based on inliers
                tracking_quality = np.sum(mask_pose) / len(good_matches)
            else:
                position = self.current_pose[:3, 3].tolist()
                tracking_quality = 0.5

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

    def reset(self):
        """Reset SLAM to initial state."""
        self.map_points = {}
        self.keyframes = []
        self.current_pose = np.eye(4, dtype=np.float32)
        self.next_point_id = 0
        self.next_keyframe_id = 0
        self.is_initialized = False
        self.position_history.clear()
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
