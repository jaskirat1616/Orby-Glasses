#!/usr/bin/env python3
"""
pySLAM Wrapper for OrbyGlasses
Handles pySLAM integration without environment conflicts
"""

import os
import sys
import cv2
import numpy as np
import logging
import time
import subprocess
from typing import Dict, Optional, List, Tuple
from collections import deque

# Add pySLAM path to sys.path
pyslam_path = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'pyslam')
if os.path.exists(pyslam_path) and pyslam_path not in sys.path:
    sys.path.insert(0, pyslam_path)

# Try to import pySLAM core modules
PYSLAM_AVAILABLE = False
try:
    from pyslam.config import Config
    from pyslam.slam.slam import Slam, SlamState
    from pyslam.slam.camera import PinholeCamera
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs, FeatureTrackerTypes
    from pyslam.local_features.feature_types import FeatureDetectorTypes
    from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs
    PYSLAM_AVAILABLE = True
    print("✅ pySLAM core modules imported successfully!")
except ImportError as e:
    PYSLAM_AVAILABLE = False
    print(f"pySLAM not available: {e}")


class PySLAMWrapper:
    """
    pySLAM wrapper that handles environment conflicts and provides clean interface.
    """

    def __init__(self, config: Dict):
        """Initialize pySLAM wrapper."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Camera parameters
        self.width = config.get('camera.width', 640)
        self.height = config.get('camera.height', 480)
        self.fx = config.get('mapping3d.fx', 500)
        self.fy = config.get('mapping3d.fy', 500)
        self.cx = self.width / 2
        self.cy = self.height / 2

        # State variables
        self.frame_count = 0
        self.is_initialized = False
        self.current_pose = np.eye(4)
        self.trajectory = []
        self.map_points = []
        
        # Visualization settings
        self.enable_visualization = config.get('slam.visualize', True)
        
        # Initialize SLAM system
        if PYSLAM_AVAILABLE:
            self._initialize_pyslam()
        else:
            raise RuntimeError("pySLAM is not available. Please install pySLAM properly.")

    def _initialize_pyslam(self):
        """Initialize pySLAM system with proper configuration."""
        try:
            # Create camera configuration
            camera_config = Config()
            camera_config.cam_settings = {
                'Camera.width': self.width,
                'Camera.height': self.height,
                'Camera.fx': self.fx,
                'Camera.fy': self.fy,
                'Camera.cx': self.cx,
                'Camera.cy': self.cy,
                'Camera.fps': 30,
                'Camera.k1': 0.0,
                'Camera.k2': 0.0,
                'Camera.p1': 0.0,
                'Camera.p2': 0.0,
                'Camera.k3': 0.0
            }
            
            # Create camera
            self.camera = PinholeCamera(camera_config)

            # Feature detector configuration
            feature_type = self.config.get('slam.feature_type', 'ORB')
            if feature_type == 'ORB':
                slam_config = Config()
                slam_config.feature_detector_type = FeatureDetectorTypes.ORB
            elif feature_type == 'SIFT':
                slam_config = Config()
                slam_config.feature_detector_type = FeatureDetectorTypes.SIFT
            else:
                slam_config = Config()
                slam_config.feature_detector_type = FeatureDetectorTypes.ORB

            # SLAM configuration
            slam_config.num_features = self.config.get('slam.orb_features', 2000)
            slam_config.enable_loop_closing = self.config.get('slam.loop_closure', False)
            slam_config.enable_local_mapping = True

            # Create feature tracker config
            feature_tracker_config = FeatureTrackerConfigs.ORB2.copy()
            feature_tracker_config["num_features"] = self.config.get('slam.orb_features', 2000)

            # Configure loop closing/relocalization
            # Using IBOW (Incremental Bag of Words) - it builds vocabulary incrementally
            # and doesn't require a pre-existing vocabulary file
            loop_detector_config = LoopDetectorConfigs.IBOW

            # Initialize SLAM with loop closing enabled
            self.slam = Slam(self.camera, feature_tracker_config, loop_detector_config=loop_detector_config)
            self.is_initialized = True
            
            print("✅ pySLAM initialized successfully!")
            self.logger.info("✅ pySLAM initialized successfully!")
            
        except Exception as e:
            error_msg = f"Failed to initialize pySLAM: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def process_frame(self, frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> Dict:
        """
        Process a single frame through pySLAM.

        Args:
            frame: Input frame (BGR or grayscale)
            depth_map: Depth map (ignored for monocular SLAM)

        Returns:
            Dictionary with SLAM results
        """
        self.frame_count += 1
        
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            # Convert to 3-channel BGR for pySLAM
            if len(gray.shape) == 2:
                img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = gray
            
            # Process frame through pySLAM
            timestamp = time.time()
            self.slam.track(img_bgr, None, None, self.frame_count, timestamp)
            
            # Get tracking state
            tracking_state = "OK"
            if hasattr(self.slam, 'tracking') and hasattr(self.slam.tracking, 'state'):
                if self.slam.tracking.state == SlamState.LOST:
                    tracking_state = "LOST"
                elif self.slam.tracking.state == SlamState.NOT_INITIALIZED:
                    tracking_state = "NOT_INITIALIZED"
            
            # Get current pose
            if hasattr(self.slam, 'tracking') and hasattr(self.slam.tracking, 'cur_pose'):
                self.current_pose = self.slam.tracking.cur_pose
            elif hasattr(self.slam, 'tracking') and hasattr(self.slam.tracking, 'get_current_pose'):
                self.current_pose = self.slam.tracking.get_current_pose()
            
            # Add to trajectory
            self.trajectory.append(self.current_pose.copy())
            
            # Get map points
            self.map_points = self.get_map_points()
            
            # Create result
            result = {
                'pose': self.current_pose.copy(),
                'position': self.current_pose[:3, 3].copy(),
                'tracking_quality': 0.9 if tracking_state == "OK" else 0.0,
                'tracking_state': tracking_state,
                'message': f"pySLAM frame {self.frame_count}",
                'is_initialized': self.is_initialized,
                'trajectory_length': len(self.trajectory),
                'num_map_points': len(self.map_points),
                'performance': {}
            }
            
            return result
                
        except Exception as e:
            self.logger.error(f"SLAM processing error: {e}")
            return {
                'pose': self.current_pose.copy(),
                'position': self.current_pose[:3, 3].copy(),
                'tracking_quality': 0.0,
                'tracking_state': "ERROR",
                'message': f"SLAM error: {e}",
                'is_initialized': False,
                'trajectory_length': len(self.trajectory),
                'num_map_points': 0,
                'performance': {}
            }

    def get_map_points(self) -> np.ndarray:
        """Get all map points for visualization."""
        try:
            if hasattr(self.slam, 'map') and self.slam.map is not None:
                map_points = self.slam.map.get_points()
                if map_points is not None and len(map_points) > 0:
                    positions = []
                    for point in map_points:
                        if hasattr(point, 'get_pos'):
                            positions.append(point.get_pos())
                        elif hasattr(point, 'pos'):
                            positions.append(point.pos)
                    return np.array(positions)
        except Exception as e:
            self.logger.error(f"Error getting map points: {e}")
        return np.array([]).reshape(0, 3)

    def is_tracking_good(self) -> bool:
        """Check if SLAM tracking is good."""
        if hasattr(self.slam, 'tracking') and hasattr(self.slam.tracking, 'state'):
            return self.slam.tracking.state == SlamState.OK
        return True

    def get_current_pose(self) -> np.ndarray:
        """Get the current estimated camera pose."""
        return self.current_pose

    def reset(self):
        """Reset the SLAM system."""
        try:
            if hasattr(self.slam, 'reset'):
                self.slam.reset()
            self.frame_count = 0
            self.is_initialized = False
            self.current_pose = np.eye(4)
            self.trajectory = []
            self.map_points = []
            self.logger.info("SLAM system reset")
        except Exception as e:
            self.logger.error(f"Reset error: {e}")

    def shutdown(self):
        """Shutdown SLAM system."""
        try:
            # Shutdown SLAM
            if hasattr(self.slam, 'shutdown'):
                self.slam.shutdown()
            
            self.logger.info("SLAM system shutdown")
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")


# For backward compatibility
def create_pyslam_system(config: Dict) -> PySLAMWrapper:
    """Create a pySLAM wrapper instance."""
    return PySLAMWrapper(config)