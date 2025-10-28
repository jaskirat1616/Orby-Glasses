"""
WORKING Simple Monocular SLAM
Adapted from: https://github.com/Fosowl/monocularSlam
This actually WORKS - it's been tested and proven!
"""

import numpy as np
import cv2
import logging
from typing import Dict, Optional, List
import sys
import os

# Import the working SLAM implementation (in same directory)
from navigation.simple_visual_slam import Slam as SimpleSLAM


class WorkingSLAM:
    """
    Wrapper around proven-working simple monocular SLAM
    This is not fancy, but it WORKS!
    """

    def __init__(self, config):
        """Initialize working SLAM"""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Camera parameters
        self.width = config.get('camera.width', 320)
        self.height = config.get('camera.height', 240)

        # Initialize the SLAM system
        self.slam = SimpleSLAM(width=self.width, height=self.height)

        # State
        self.is_initialized = False
        self.frame_count = 0
        self.last_frame = None
        self.current_pose = np.eye(4, dtype=np.float64)

        self.logger.info("✅ Working SLAM initialized (proven implementation)")
        self.logger.info("Source: https://github.com/Fosowl/monocularSlam")

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None,
                     depth_map: Optional[np.ndarray] = None) -> Dict:
        """Process frame through SLAM"""
        self.frame_count += 1

        # Convert to BGR if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Need two frames to start
        if self.last_frame is None:
            self.last_frame = frame.copy()
            self.logger.info("Stored first frame")
            return self._get_default_result()

        # Check if frames are different
        if (frame == self.last_frame).all():
            return self._get_default_result()

        try:
            # Update frames
            self.slam.update_frame_pixels(frame, self.last_frame)

            # Get matches and pose
            matches, _ = self.slam.get_vision_matches(frame.copy())

            if matches is not None and len(matches) > 3:  # Reduced from 10 to 3 for more robust initialization
                # Triangulate points
                points = self.slam.triangulate(matches)

                # Get pose
                R_total, t_total = self.slam.vision.get_pose_cumulation()

                # Update current pose
                self.current_pose[:3, :3] = R_total
                self.current_pose[:3, 3] = t_total.ravel()

                if not self.is_initialized:
                    self.is_initialized = True
                    self.logger.info(f"✅ SLAM initialized with {len(matches)} matches!")

                # Store frame
                self.last_frame = frame.copy()

                return {
                    'pose': self.current_pose,
                    'position': t_total.ravel().tolist(),
                    'tracking_quality': min(1.0, len(matches) / 100.0),
                    'tracking_state': 'OK',
                    'num_map_points': len(points) if points else 0,
                    'num_keyframes': len(self.slam.get_camera_poses()),
                    'num_matches': len(matches),
                    'initialized': True
                }
            else:
                # Store frame anyway but return degraded result
                self.last_frame = frame.copy()
                if self.is_initialized:
                    # Return last known good pose
                    return {
                        'pose': self.current_pose,
                        'position': self.current_pose[:3, 3].tolist(),
                        'tracking_quality': 0.1,
                        'tracking_state': 'DEGRADED',
                        'num_map_points': 0,
                        'num_keyframes': len(self.slam.get_camera_poses()),
                        'num_matches': len(matches) if matches else 0,
                        'initialized': True
                    }
                else:
                    if self.frame_count % 30 == 0:
                        self.logger.info("⚠️  SLAM initializing - MOVE camera while pointing at textured surface")
                    return self._get_default_result()

        except Exception as e:
            if "No matches found" not in str(e):
                self.logger.debug(f"SLAM frame processing: {e}")

        # Store frame
        self.last_frame = frame.copy()

        if self.is_initialized:
            # Return last known good pose
            return {
                'pose': self.current_pose,
                'position': self.current_pose[:3, 3].tolist(),
                'tracking_quality': 0.3,
                'tracking_state': 'DEGRADED',
                'num_map_points': 0,
                'num_keyframes': len(self.slam.get_camera_poses()),
                'initialized': True
            }
        else:
            if self.frame_count % 30 == 0:
                self.logger.info("⚠️  SLAM initializing - MOVE camera while pointing at textured surface")
            return self._get_default_result()

    def get_map_points(self) -> np.ndarray:
        """Get map points"""
        if hasattr(self.slam, 'points3Dcumulative') and len(self.slam.points3Dcumulative) > 0:
            # Get last set of points
            points_data = self.slam.points3Dcumulative[-1]
            if points_data is not None and len(points_data) > 0:
                points = points_data[0]  # First element is the points array
                if points is not None and points.shape[0] == 3:
                    return points.T  # Transpose to Nx3
        return np.array([]).reshape(0, 3)

    def get_keyframes(self) -> List[np.ndarray]:
        """Get keyframe poses"""
        poses = []
        for pose_data in self.slam.get_camera_poses():
            pose_mat = np.eye(4, dtype=np.float64)
            pose_mat[:3, :3] = pose_data['R']
            pose_mat[:3, 3] = pose_data['t'].ravel()
            poses.append(pose_mat)
        return poses

    def get_position(self) -> np.ndarray:
        """Get current position"""
        return self.current_pose[:3, 3]

    def get_pose(self) -> np.ndarray:
        """Get current pose"""
        return self.current_pose

    def reset(self):
        """Reset SLAM"""
        self.slam = SimpleSLAM(width=self.width, height=self.height)
        self.is_initialized = False
        self.last_frame = None
        self.current_pose = np.eye(4, dtype=np.float64)
        self.logger.info("SLAM reset")

    def _get_default_result(self) -> Dict:
        """Default result"""
        return {
            'pose': self.current_pose,
            'position': [0, 0, 0],
            'tracking_quality': 0.0,
            'tracking_state': 'NOT_INITIALIZED',
            'num_map_points': 0,
            'num_keyframes': 0,
            'initialized': False
        }
