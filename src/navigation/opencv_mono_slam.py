"""
Lightweight OpenCV-based Monocular SLAM for OrbyGlasses
Production-ready visual odometry with pose estimation.

This implementation uses:
- ORB features for tracking
- Essential matrix decomposition for pose estimation
- Triangulation for 3D reconstruction
- Keyframe-based mapping
"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List
from collections import deque
import time


class OpenCVMonocularSLAM:
    """
    Lightweight monocular SLAM using OpenCV.
    Suitable for real-time applications on macOS.
    """

    def __init__(self, config):
        """
        Initialize OpenCV monocular SLAM.

        Args:
            config: ConfigManager instance
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Camera parameters
        width = config.get('camera.width', 320)
        height = config.get('camera.height', 240)
        self.fx = config.get('mapping3d.fx', 500)
        self.fy = config.get('mapping3d.fy', 500)
        self.cx = width / 2
        self.cy = height / 2

        # Camera intrinsic matrix
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Feature detector and matcher
        self.orb = cv2.ORB_create(
            nfeatures=config.get('slam.orb_features', 2000),
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=config.get('slam.fast_threshold', 20)
        )

        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # State
        self.is_initialized = False
        self.frame_count = 0
        self.current_pose = np.eye(4, dtype=np.float32)  # 4x4 transformation matrix
        self.tracking_state = "NOT_INITIALIZED"

        # Previous frame data
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None

        # Map points (3D points in world coordinates)
        self.map_points = []  # List of [x, y, z] points
        self.map_point_colors = []  # Colors for visualization

        # Keyframes
        self.keyframes = []  # List of poses
        self.keyframe_threshold = 30  # Create keyframe every N frames

        # History for visualization
        self.pose_history = deque(maxlen=1000)
        self.position_history = deque(maxlen=1000)

        # RANSAC parameters
        self.ransac_threshold = 1.0  # pixels
        self.ransac_confidence = 0.999

        self.logger.info("OpenCV Monocular SLAM initialized")
        self.logger.info(f"Camera: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> Dict:
        """
        Process frame through monocular SLAM.

        Args:
            frame: Grayscale or BGR frame
            timestamp: Frame timestamp (seconds)

        Returns:
            Dictionary with pose, position, tracking quality
        """
        self.frame_count += 1

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 8:
            self.logger.warning(f"Insufficient features detected: {len(keypoints) if keypoints else 0}")
            return self._get_tracking_lost_result()

        # First frame - initialize
        if not self.is_initialized:
            self.prev_frame = gray.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.is_initialized = True
            self.tracking_state = "INITIALIZING"

            return {
                'pose': self.current_pose,
                'position': [0, 0, 0],
                'tracking_quality': 0.0,
                'tracking_state': self.tracking_state,
                'num_map_points': 0,
                'num_tracked_points': 0,
                'is_keyframe': False,
                'initialized': False
            }

        # Match features with previous frame
        matches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 8:
            self.logger.warning(f"Insufficient good matches: {len(good_matches)}")
            return self._get_tracking_lost_result()

        # Extract matched point coordinates
        pts_prev = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
        pts_curr = np.float32([keypoints[m.trainIdx].pt for m in good_matches])

        # Estimate essential matrix and recover pose
        E, mask = cv2.findEssentialMat(
            pts_prev, pts_curr, self.K,
            method=cv2.RANSAC,
            prob=self.ransac_confidence,
            threshold=self.ransac_threshold
        )

        if E is None:
            self.logger.warning("Failed to compute essential matrix")
            return self._get_tracking_lost_result()

        # Recover relative pose (R, t)
        _, R, t, mask_pose = cv2.recoverPose(E, pts_prev, pts_curr, self.K, mask=mask)

        # Filter inliers
        inlier_matches = [m for i, m in enumerate(good_matches) if mask_pose[i]]
        num_inliers = len(inlier_matches)

        if num_inliers < 8:
            self.logger.warning(f"Insufficient inliers: {num_inliers}")
            return self._get_tracking_lost_result()

        # Update pose
        # Convert R, t to 4x4 transformation matrix
        T_relative = np.eye(4, dtype=np.float32)
        T_relative[:3, :3] = R
        T_relative[:3, 3] = t.flatten()

        # Update cumulative pose
        self.current_pose = self.current_pose @ T_relative

        # Extract position
        position = self.current_pose[:3, 3].copy()

        # Update history
        self.pose_history.append(self.current_pose.copy())
        self.position_history.append(position.tolist())

        # Triangulate 3D points for map building
        if self.frame_count % 5 == 0:  # Triangulate every 5 frames
            self._triangulate_points(pts_prev, pts_curr, R, t, inlier_matches, gray)

        # Check if we need a keyframe
        is_keyframe = (self.frame_count % self.keyframe_threshold == 0)
        if is_keyframe:
            self.keyframes.append(self.current_pose.copy())

        # Update previous frame data
        self.prev_frame = gray.copy()
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        # Tracking quality based on number of inliers
        tracking_quality = min(1.0, num_inliers / 100.0)
        self.tracking_state = "OK"

        return {
            'pose': self.current_pose,
            'position': position.tolist(),
            'tracking_quality': tracking_quality,
            'tracking_state': self.tracking_state,
            'num_map_points': len(self.map_points),
            'num_tracked_points': num_inliers,
            'is_keyframe': is_keyframe,
            'initialized': True
        }

    def _triangulate_points(self, pts_prev: np.ndarray, pts_curr: np.ndarray,
                           R: np.ndarray, t: np.ndarray, matches: List, gray: np.ndarray):
        """
        Triangulate 3D points from matched 2D points.

        Args:
            pts_prev: Previous frame points
            pts_curr: Current frame points
            R: Rotation matrix
            t: Translation vector
            matches: List of good matches
            gray: Current frame for color extraction
        """
        # Projection matrices
        P1 = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Previous frame at origin
        P2 = self.K @ np.hstack((R, t))  # Current frame

        # Triangulate
        pts_prev_matched = pts_prev[:len(matches)]
        pts_curr_matched = pts_curr[:len(matches)]

        points_4d = cv2.triangulatePoints(
            P1, P2,
            pts_prev_matched.T,
            pts_curr_matched.T
        )

        # Convert to 3D
        points_3d = points_4d[:3, :] / points_4d[3, :]
        points_3d = points_3d.T

        # Filter points (remove outliers)
        valid_mask = (points_3d[:, 2] > 0) & (points_3d[:, 2] < 50)  # Depth between 0 and 50 units
        points_3d = points_3d[valid_mask]

        # Transform to world coordinates
        points_3d_world = (self.current_pose[:3, :3] @ points_3d.T + self.current_pose[:3, 3:4]).T

        # Add to map (limit total points)
        max_map_points = 5000
        if len(self.map_points) < max_map_points:
            for pt in points_3d_world:
                self.map_points.append(pt.tolist())

    def get_map_points(self) -> np.ndarray:
        """
        Get all map points for visualization.

        Returns:
            Nx3 array of map point positions
        """
        if len(self.map_points) == 0:
            return np.array([]).reshape(0, 3)
        return np.array(self.map_points)

    def get_keyframes(self) -> List:
        """
        Get all keyframe poses.

        Returns:
            List of 4x4 pose matrices
        """
        return self.keyframes

    def reset(self):
        """Reset SLAM system."""
        self.is_initialized = False
        self.frame_count = 0
        self.current_pose = np.eye(4, dtype=np.float32)
        self.tracking_state = "NOT_INITIALIZED"
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.map_points = []
        self.keyframes = []
        self.pose_history.clear()
        self.position_history.clear()
        self.logger.info("SLAM reset")

    def get_position(self) -> np.ndarray:
        """Get current camera position."""
        return self.current_pose[:3, 3]

    def get_pose(self) -> np.ndarray:
        """Get current camera pose."""
        return self.current_pose

    def _get_tracking_lost_result(self) -> Dict:
        """Get default result when tracking fails."""
        self.tracking_state = "LOST"
        return {
            'pose': self.current_pose,
            'position': self.current_pose[:3, 3].tolist(),
            'tracking_quality': 0.0,
            'tracking_state': self.tracking_state,
            'num_map_points': len(self.map_points),
            'num_tracked_points': 0,
            'is_keyframe': False,
            'initialized': self.is_initialized
        }
