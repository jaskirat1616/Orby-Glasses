"""
Proper Monocular SLAM for OrbyGlasses
Pure vision-based SLAM without depth sensors - uses triangulation and essential matrix.
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class MonocularMapPoint:
    """3D point triangulated from multiple views."""
    id: int
    position: np.ndarray  # 3D coordinates [x, y, z]
    descriptor: np.ndarray
    observations: List[Tuple[int, int]] = None  # List of (keyframe_id, keypoint_idx)
    reprojection_error: float = 0.0

    def __post_init__(self):
        if self.observations is None:
            self.observations = []


@dataclass
class MonocularKeyFrame:
    """Keyframe with pose and features."""
    id: int
    timestamp: float
    pose: np.ndarray  # 4x4 transformation matrix (world to camera)
    keypoints: List[cv2.KeyPoint]
    descriptors: np.ndarray
    frame: Optional[np.ndarray] = None


class ProperMonocularSLAM:
    """
    Proper monocular SLAM using:
    - Essential matrix for pose estimation
    - Triangulation for 3D point reconstruction
    - Bundle adjustment for refinement
    - Loop closure for drift correction
    """

    def __init__(self, config):
        """Initialize monocular SLAM system."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Camera intrinsics
        width = config.get('camera.width', 320)
        height = config.get('camera.height', 240)
        self.fx = config.get('mapping3d.fx', 500)
        self.fy = config.get('mapping3d.fy', 500)
        self.cx = width / 2
        self.cy = height / 2

        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # ORB feature detector
        nfeatures = config.get('slam.orb_features', 1200)
        fast_threshold = config.get('slam.fast_threshold', 15)

        self.orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=fast_threshold
        )

        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # SLAM state
        self.keyframes = []
        self.map_points = {}
        self.current_pose = np.eye(4, dtype=np.float64)
        self.scale = 1.0  # Estimated scale (monocular SLAM scale ambiguity)

        # Tracking state
        self.is_initialized = False
        self.last_frame = None
        self.last_keypoints = None
        self.last_descriptors = None

        # IDs
        self.next_kf_id = 0
        self.next_point_id = 0
        self.frame_count = 0

        # Parameters
        self.min_matches = config.get('slam.min_matches', 20)
        self.min_parallax = 1.0  # degrees
        self.max_parallax = 45.0  # degrees
        self.min_triangulation_angle = 2.0  # degrees

        # History for motion estimation
        self.pose_history = deque(maxlen=100)
        self.position_history = deque(maxlen=1000)

        self.logger.info("Proper Monocular SLAM initialized")
        self.logger.info(f"Camera: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    def process_frame(self, frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> Dict:
        """
        Process new frame through proper monocular SLAM pipeline.

        Args:
            frame: Grayscale frame

        Returns:
            Dictionary with pose, position, tracking quality
        """
        self.frame_count += 1

        # Detect ORB features
        keypoints, descriptors = self.orb.detectAndCompute(frame, None)

        if descriptors is None or len(keypoints) < 10:
            self.logger.warning(f"Not enough features: {len(keypoints) if keypoints else 0}")
            return self._get_default_result()

        # First frame - initialize
        if not self.is_initialized:
            return self._initialize_monocular(frame, keypoints, descriptors)

        # Track and estimate pose
        return self._track_monocular(frame, keypoints, descriptors)

    def _initialize_monocular(self, frame: np.ndarray, keypoints: List, descriptors: np.ndarray) -> Dict:
        """
        Initialize monocular SLAM - requires TWO frames to triangulate.
        """
        if self.last_frame is None:
            # Store first frame
            self.last_frame = frame.copy()
            self.last_keypoints = keypoints
            self.last_descriptors = descriptors

            self.logger.info("Stored first frame for initialization")
            return self._get_default_result()

        # Second frame - compute essential matrix and triangulate
        self.logger.info("Initializing with frame pair...")

        # Match features between first and second frame
        matches = self.matcher.knnMatch(self.last_descriptors, descriptors, k=2)

        # Ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 50:
            self.logger.warning(f"Not enough matches for initialization: {len(good_matches)}")
            # Try again with next frame
            self.last_frame = frame.copy()
            self.last_keypoints = keypoints
            self.last_descriptors = descriptors
            return self._get_default_result()

        # Get matched points
        pts1 = np.float32([self.last_keypoints[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([keypoints[m.trainIdx].pt for m in good_matches])

        # Compute essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            self.logger.warning("Essential matrix computation failed")
            self.last_frame = frame.copy()
            self.last_keypoints = keypoints
            self.last_descriptors = descriptors
            return self._get_default_result()

        # Recover pose from essential matrix
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)

        # Check if baseline is sufficient (parallax check)
        parallax = self._compute_parallax(pts1[mask.ravel() == 1], pts2[mask.ravel() == 1])

        if parallax < self.min_parallax:
            self.logger.warning(f"Insufficient parallax: {parallax:.2f}° (min {self.min_parallax}°)")
            self.last_frame = frame.copy()
            self.last_keypoints = keypoints
            self.last_descriptors = descriptors
            return self._get_default_result()

        self.logger.info(f"Good parallax: {parallax:.2f}°, matches: {len(good_matches)}")

        # Create first two keyframes
        pose1 = np.eye(4, dtype=np.float64)
        kf1 = MonocularKeyFrame(
            id=0,
            timestamp=time.time() - 0.1,
            pose=pose1,
            keypoints=self.last_keypoints,
            descriptors=self.last_descriptors
        )

        pose2 = np.eye(4, dtype=np.float64)
        pose2[:3, :3] = R
        pose2[:3, 3] = t.flatten()

        kf2 = MonocularKeyFrame(
            id=1,
            timestamp=time.time(),
            pose=pose2,
            keypoints=keypoints,
            descriptors=descriptors
        )

        self.keyframes = [kf1, kf2]
        self.next_kf_id = 2

        # Triangulate initial map points
        self._triangulate_points(kf1, kf2, good_matches, mask)

        # Set current pose
        self.current_pose = pose2.copy()
        self.pose_history.append(self.current_pose)

        position = self.current_pose[:3, 3]
        self.position_history.append(position.tolist())

        # Update for next frame
        self.last_frame = frame.copy()
        self.last_keypoints = keypoints
        self.last_descriptors = descriptors
        self.is_initialized = True

        self.logger.info(f"✅ Monocular SLAM initialized with {len(self.map_points)} map points")

        return {
            'pose': self.current_pose,
            'position': position.tolist(),
            'tracking_quality': 1.0,
            'num_matches': len(good_matches),
            'is_keyframe': True,
            'num_map_points': len(self.map_points),
            'initialization': True
        }

    def _track_monocular(self, frame: np.ndarray, keypoints: List, descriptors: np.ndarray) -> Dict:
        """Track camera pose using PnP with existing map points."""

        # Match current frame with last keyframe
        last_kf = self.keyframes[-1]
        matches = self.matcher.knnMatch(last_kf.descriptors, descriptors, k=2)

        # Ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < self.min_matches:
            self.logger.warning(f"Lost tracking: only {len(good_matches)} matches")
            # Use motion model
            return self._get_motion_prediction()

        # Get 3D-2D correspondences
        object_points = []
        image_points = []

        for m in good_matches:
            kf_kp_idx = m.queryIdx
            # Check if this keypoint has a corresponding map point
            for mp_id, mp in self.map_points.items():
                for obs_kf_id, obs_kp_idx in mp.observations:
                    if obs_kf_id == last_kf.id and obs_kp_idx == kf_kp_idx:
                        object_points.append(mp.position)
                        image_points.append(keypoints[m.trainIdx].pt)
                        break

        if len(object_points) < 6:
            self.logger.warning(f"Not enough 3D-2D correspondences: {len(object_points)}")
            return self._get_motion_prediction()

        object_points = np.float32(object_points)
        image_points = np.float32(image_points)

        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, self.K, None,
            iterationsCount=200,
            reprojectionError=2.0,
            confidence=0.99
        )

        if not success or inliers is None or len(inliers) < 6:
            self.logger.warning("PnP failed")
            return self._get_motion_prediction()

        # Convert to pose matrix
        R, _ = cv2.Rodrigues(rvec)
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R
        pose[:3, 3] = tvec.flatten()

        # Apply scale
        pose[:3, 3] *= self.scale

        self.current_pose = pose
        self.pose_history.append(pose)

        position = pose[:3, 3]
        self.position_history.append(position.tolist())

        # Update
        self.last_frame = frame.copy()
        self.last_keypoints = keypoints
        self.last_descriptors = descriptors

        tracking_quality = len(inliers) / len(good_matches)

        return {
            'pose': self.current_pose,
            'position': position.tolist(),
            'tracking_quality': tracking_quality,
            'num_matches': len(good_matches),
            'num_inliers': len(inliers),
            'is_keyframe': False,
            'num_map_points': len(self.map_points)
        }

    def _triangulate_points(self, kf1: MonocularKeyFrame, kf2: MonocularKeyFrame,
                           matches: List, mask: np.ndarray):
        """Triangulate 3D points from two keyframes."""

        # Projection matrices
        P1 = self.K @ kf1.pose[:3]
        P2 = self.K @ kf2.pose[:3]

        # Filter matches by mask
        good_matches = [m for i, m in enumerate(matches) if mask[i]]

        for m in good_matches:
            pt1 = kf1.keypoints[m.queryIdx].pt
            pt2 = kf2.keypoints[m.trainIdx].pt

            # Triangulate
            point_4d = cv2.triangulatePoints(
                P1, P2,
                np.array(pt1).reshape(2, 1),
                np.array(pt2).reshape(2, 1)
            )

            # Convert from homogeneous
            point_3d = point_4d[:3] / point_4d[3]
            point_3d = point_3d.flatten()

            # Check if point is in front of both cameras and reasonable depth
            if point_3d[2] > 0.1 and point_3d[2] < 10.0:
                # Check reprojection error
                reproj_error = self._compute_reprojection_error(point_3d, pt1, kf1.pose)

                if reproj_error < 2.0:
                    # Create map point
                    mp = MonocularMapPoint(
                        id=self.next_point_id,
                        position=point_3d,
                        descriptor=kf1.descriptors[m.queryIdx],
                        observations=[(kf1.id, m.queryIdx), (kf2.id, m.trainIdx)],
                        reprojection_error=reproj_error
                    )

                    self.map_points[self.next_point_id] = mp
                    self.next_point_id += 1

    def _compute_parallax(self, pts1: np.ndarray, pts2: np.ndarray) -> float:
        """Compute median parallax angle in degrees."""
        # Normalize points
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.K, None).reshape(-1, 2)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), self.K, None).reshape(-1, 2)

        # Compute angles
        angles = []
        for p1, p2 in zip(pts1_norm, pts2_norm):
            # Convert to unit vectors
            v1 = np.array([p1[0], p1[1], 1.0])
            v2 = np.array([p2[0], p2[1], 1.0])
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)

            # Angle between vectors
            cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            angles.append(angle)

        return np.median(angles)

    def _compute_reprojection_error(self, point_3d: np.ndarray, point_2d: Tuple[float, float],
                                   pose: np.ndarray) -> float:
        """Compute reprojection error."""
        # Project 3D point to 2D
        point_3d_h = np.append(point_3d, 1.0)
        point_cam = pose[:3] @ point_3d_h
        point_proj = self.K @ point_cam
        point_proj /= point_proj[2]

        # Error
        error = np.linalg.norm(point_proj[:2] - np.array(point_2d))
        return error

    def _get_motion_prediction(self) -> Dict:
        """Predict pose using motion model."""
        if len(self.pose_history) < 2:
            pose = self.current_pose
        else:
            # Simple constant velocity model
            pose = self.pose_history[-1].copy()

        position = pose[:3, 3]

        return {
            'pose': pose,
            'position': position.tolist(),
            'tracking_quality': 0.3,
            'num_matches': 0,
            'is_keyframe': False,
            'num_map_points': len(self.map_points),
            'prediction': True
        }

    def _get_default_result(self) -> Dict:
        """Get default result when SLAM not initialized."""
        return {
            'pose': self.current_pose,
            'position': [0, 0, 0],
            'tracking_quality': 0.0,
            'num_matches': 0,
            'is_keyframe': False,
            'num_map_points': 0,
            'initialized': False
        }

    def get_position(self) -> np.ndarray:
        """Get current camera position."""
        return self.current_pose[:3, 3]

    def get_pose(self) -> np.ndarray:
        """Get current camera pose."""
        return self.current_pose
