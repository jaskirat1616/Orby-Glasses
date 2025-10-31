"""
Proper pySLAM Integration for OrbyGlasses
Based on original main_slam.py and main_vo.py from pySLAM
"""

import os
import sys
import cv2
import numpy as np
import logging
import time
import threading
from typing import Dict, Optional, List
from collections import deque

# Add pySLAM to path
pyslam_path = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'pyslam')
if os.path.exists(pyslam_path) and pyslam_path not in sys.path:
    sys.path.insert(0, pyslam_path)

try:
    from pyslam.config import Config
    from pyslam.slam.slam import Slam
    from pyslam.slam.visual_odometry import VisualOdometryEducational
    from pyslam.slam.camera import PinholeCamera
    from pyslam.io.ground_truth import groundtruth_factory
    from pyslam.io.dataset_factory import dataset_factory
    from pyslam.io.dataset_types import DatasetType, SensorType
    from pyslam.viz.viewer3D import Viewer3D
    from pyslam.viz.rerun_interface import Rerun
    from pyslam.local_features.feature_tracker import feature_tracker_factory
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
    from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs
    PYSLAM_AVAILABLE = True
except Exception as e:
    PYSLAM_AVAILABLE = False
    print(f"pySLAM not available: {e}")


class ProperPySLAMIntegration:
    """
    Proper pySLAM integration following original main_slam.py and main_vo.py structure.
    Uses dataset_factory for live camera access.
    """

    def __init__(self, mode='slam', use_rerun=True):
        """
        Initialize pySLAM properly.

        Args:
            mode: 'slam' or 'vo'
            use_rerun: Use Rerun.io for visualization
        """
        self.logger = logging.getLogger(__name__)
        self.mode = mode
        self.use_rerun = use_rerun

        if not PYSLAM_AVAILABLE:
            raise RuntimeError("pySLAM not available")

        # Configuration - use config_live.yaml
        config_path = os.path.join(pyslam_path, 'config_live.yaml')
        self.config = Config(config_path)

        # Create dataset (handles live camera)
        self.dataset = dataset_factory(self.config)
        self.is_monocular = self.dataset.sensor_type == SensorType.MONOCULAR

        # Create camera
        self.camera = PinholeCamera(self.config)

        # Create groundtruth (None for live)
        self.groundtruth = groundtruth_factory(self.config.dataset_settings)

        # Feature tracker config
        self.feature_tracker_config = FeatureTrackerConfigs.ORB2
        num_features = self.config.num_features_to_extract if self.config.num_features_to_extract > 0 else 2000
        self.feature_tracker_config["num_features"] = num_features

        # Create feature tracker
        self.feature_tracker = feature_tracker_factory(**self.feature_tracker_config)

        # Initialize SLAM or VO
        if mode == 'slam':
            self._init_slam()
        else:  # mode == 'vo'
            self._init_vo()

        # State
        self.img_id = 0
        self.is_running = False
        self.trajectory = deque(maxlen=1000)

        # Visualization
        self.traj_img_size = 800
        self.traj_img = np.zeros((self.traj_img_size, self.traj_img_size, 3), dtype=np.uint8)
        self.draw_scale = 1

        self.logger.info(f"✅ pySLAM {mode.upper()} initialized properly!")

    def _init_slam(self):
        """Initialize SLAM following main_slam.py"""
        # Loop detector config
        loop_detection_config = LoopDetectorConfigs.DBOW3

        # Override from config if specified
        if self.config.loop_detection_config_name is not None:
            loop_detection_config = LoopDetectorConfigs.get_config_from_name(
                self.config.loop_detection_config_name
            )

        # Create SLAM object
        self.slam = Slam(
            self.camera,
            self.groundtruth,
            self.feature_tracker_config,
            loop_detection_config,
            None  # semantic_mapping_config
        )

        # Initialize Rerun if enabled
        if self.use_rerun and Rerun.is_ok:
            try:
                Rerun.init_slam()
                self.logger.info("✅ Rerun.io initialized for SLAM")
            except Exception as e:
                self.logger.warning(f"Rerun init failed: {e}")
                self.use_rerun = False

        # 3D Viewer
        try:
            self.viewer3d = Viewer3D(scale=1.0)
        except Exception as e:
            self.logger.warning(f"Viewer3D init failed: {e}")
            self.viewer3d = None

    def _init_vo(self):
        """Initialize VO following main_vo.py"""
        # Create VO object
        if self.dataset.sensor_type == SensorType.RGBD:
            from pyslam.slam.visual_odometry_rgbd import VisualOdometryRgbdTensor
            self.vo = VisualOdometryRgbdTensor(self.camera, self.groundtruth)
            self.logger.info("Using VisualOdometryRgbdTensor")
        else:
            self.vo = VisualOdometryEducational(self.camera, self.groundtruth, self.feature_tracker)
            self.logger.info("Using VisualOdometryEducational")

        # Initialize Rerun if enabled
        if self.use_rerun and Rerun.is_ok:
            try:
                Rerun.init_vo()
                self.logger.info("✅ Rerun.io initialized for VO")
            except Exception as e:
                self.logger.warning(f"Rerun init failed: {e}")
                self.use_rerun = False

        # 3D Viewer
        try:
            self.viewer3d = Viewer3D(scale=10.0)
        except Exception as e:
            self.logger.warning(f"Viewer3D init failed: {e}")
            self.viewer3d = None

    def process_frame(self) -> Optional[Dict]:
        """
        Process next frame from dataset.
        Returns tracking result or None if dataset is not ready.
        """
        if not self.dataset.is_ok:
            return None

        try:
            # Get frame from dataset (this handles the camera internally)
            timestamp = self.dataset.getTimestamp()
            img = self.dataset.getImageColor(self.img_id)

            if img is None:
                return None

            # Get depth and right image if available
            depth = self.dataset.getDepth(self.img_id)
            img_right = self.dataset.getImageColorRight(self.img_id) if self.dataset.sensor_type == SensorType.STEREO else None

            # Track with SLAM or VO
            if self.mode == 'slam':
                result = self._process_slam(img, img_right, depth, timestamp)
            else:
                result = self._process_vo(img, img_right, depth, timestamp)

            self.img_id += 1
            return result

        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return None

    def _process_slam(self, img, img_right, depth, timestamp):
        """Process frame with SLAM"""
        # Track
        self.slam.track(img, img_right, depth, self.img_id, timestamp)

        # Get pose
        pose = self.slam.tracking.get_current_pose() if hasattr(self.slam.tracking, 'get_current_pose') else np.eye(4)

        # Get trajectory
        if hasattr(self.slam, 'traj3d_est') and len(self.slam.traj3d_est) > 0:
            pos = self.slam.traj3d_est[-1]
            self.trajectory.append(pos)

        # Visualization
        if self.use_rerun:
            try:
                Rerun.log_slam_frame(self.img_id, self.slam)
            except Exception:
                pass

        if self.viewer3d and hasattr(self.slam, 'map'):
            try:
                self.viewer3d.draw_map(self.slam.map)
            except Exception:
                pass

        return {
            'pose': pose.copy(),
            'position': pose[:3, 3].copy() if pose is not None else np.zeros(3),
            'trajectory': list(self.trajectory),
            'num_keyframes': len(self.slam.map.keyframes) if hasattr(self.slam, 'map') else 0,
            'num_map_points': len(self.slam.map.points) if hasattr(self.slam, 'map') else 0,
            'tracking_state': str(self.slam.tracking.state) if hasattr(self.slam.tracking, 'state') else 'UNKNOWN'
        }

    def _process_vo(self, img, img_right, depth, timestamp):
        """Process frame with VO"""
        # Track
        self.vo.track(img, img_right, depth, self.img_id, timestamp)

        # Get trajectory
        if len(self.vo.traj3d_est) > 0:
            pos = self.vo.traj3d_est[-1]
            self.trajectory.append(pos)

            # Draw trajectory
            self._draw_trajectory_2d()

        # Get pose
        pose = np.eye(4)
        if len(self.vo.poses) > 0:
            pose = self.vo.poses[-1]

        # Visualization
        if self.use_rerun:
            try:
                Rerun.log_3d_camera_img_seq(self.img_id, self.vo.draw_img if hasattr(self.vo, 'draw_img') else img,
                                           None, self.camera, pose)
                if len(self.vo.traj3d_est) > 0:
                    Rerun.log_3d_trajectory(self.img_id, self.vo.traj3d_est, "estimated", color=[0, 0, 255])
            except Exception:
                pass

        if self.viewer3d and len(self.vo.traj3d_est) > 1:
            try:
                self.viewer3d.draw_vo_trajectory(self.vo.traj3d_est, color=(0, 0, 255))
            except Exception:
                pass

        return {
            'pose': pose.copy(),
            'position': pose[:3, 3].copy(),
            'trajectory': list(self.trajectory),
            'num_matches': self.vo.num_matched_kps if hasattr(self.vo, 'num_matched_kps') else 0,
            'num_inliers': self.vo.num_inliers if hasattr(self.vo, 'num_inliers') else 0
        }

    def _draw_trajectory_2d(self):
        """Draw 2D trajectory image"""
        if len(self.trajectory) < 2:
            return

        self.traj_img = np.zeros((self.traj_img_size, self.traj_img_size, 3), dtype=np.uint8)
        half_size = self.traj_img_size // 2

        for i in range(1, len(self.trajectory)):
            prev = self.trajectory[i-1]
            curr = self.trajectory[i]

            x1 = int(self.draw_scale * prev[0]) + half_size
            y1 = half_size - int(self.draw_scale * prev[2])
            x2 = int(self.draw_scale * curr[0]) + half_size
            y2 = half_size - int(self.draw_scale * curr[2])

            x1 = max(0, min(self.traj_img_size-1, x1))
            y1 = max(0, min(self.traj_img_size-1, y1))
            x2 = max(0, min(self.traj_img_size-1, x2))
            y2 = max(0, min(self.traj_img_size-1, y2))

            cv2.line(self.traj_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw current position
        if len(self.trajectory) > 0:
            curr = self.trajectory[-1]
            x = int(self.draw_scale * curr[0]) + half_size
            y = half_size - int(self.draw_scale * curr[2])
            x = max(0, min(self.traj_img_size-1, x))
            y = max(0, min(self.traj_img_size-1, y))
            cv2.circle(self.traj_img, (x, y), 3, (0, 0, 255), -1)

    def get_trajectory_image(self):
        """Get 2D trajectory visualization"""
        return self.traj_img.copy()

    def cleanup(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'dataset'):
                self.dataset.close()
            if hasattr(self, 'viewer3d') and self.viewer3d:
                self.viewer3d.quit = True
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
