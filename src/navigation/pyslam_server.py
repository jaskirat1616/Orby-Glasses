#!/usr/bin/env python3
"""
pySLAM Server - Runs pySLAM in its own environment to avoid OpenCV conflicts
"""

import os
import sys
import json
import time
import logging
import numpy as np
import cv2
import subprocess
from typing import Dict, Optional

# Check if we're running in the pySLAM environment
def is_pyslam_env():
    """Check if we're running in the pySLAM virtual environment"""
    return 'pyslam' in sys.executable or 'pyslam' in os.environ.get('VIRTUAL_ENV', '')

# Only try to import pySLAM if we're in the right environment
PYSLAM_AVAILABLE = False
if is_pyslam_env():
    try:
        from pyslam.config import Config
        from pyslam.slam.slam import Slam, SlamState
        from pyslam.slam.camera import PinholeCamera
        from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs, FeatureTrackerTypes
        from pyslam.local_features.feature_types import FeatureDetectorTypes
        PYSLAM_AVAILABLE = True
    except ImportError as e:
        print(f"pySLAM not available in current environment: {e}")
        PYSLAM_AVAILABLE = False
else:
    print("Not running in pySLAM environment - pySLAM not available")
    PYSLAM_AVAILABLE = False

class PySLAMServer:
    """pySLAM server that runs in its own environment"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Camera parameters
        self.width = config.get('camera.width', 640)
        self.height = config.get('camera.height', 480)
        fps = config.get('camera.fps', 30)
        self.fx = config.get('mapping3d.fx', 500)
        self.fy = config.get('mapping3d.fy', 500)
        self.cx = self.width / 2
        self.cy = self.height / 2

        # Initialize pySLAM if available
        if PYSLAM_AVAILABLE:
            self._initialize_pyslam()
        else:
            self.logger.warning("pySLAM not available - using fallback implementation")
            self._initialize_fallback()
        
        # State
        self.frame_count = 0
        self.is_initialized = False
        self.current_pose = np.eye(4)
        self.trajectory = []
        self.performance_stats = {}

    def _initialize_pyslam(self):
        """Initialize pySLAM system"""
        try:
            # Create camera config
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
            
            # Initialize SLAM
            self.slam = Slam(self.camera, feature_tracker_config)
            self.is_initialized = True
            
            print("✅ pySLAM server initialized successfully")
            
        except Exception as e:
            print(f"Failed to initialize pySLAM: {e}")
            sys.exit(1)

    def _initialize_fallback(self):
        """Initialize fallback SLAM system using OpenCV"""
        self.logger.info("Initializing fallback SLAM system (OpenCV-based)")
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=self.config.get('slam.orb_features', 2000))
        
        # Initialize matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Initialize camera matrix
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Initialize distortion coefficients (assuming no distortion)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
        
        # Previous frame data
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        
        self.is_initialized = True
        self.logger.info("Fallback SLAM system initialized")

    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame"""
        try:
            if PYSLAM_AVAILABLE:
                return self._process_pyslam_frame(frame)
            else:
                return self._process_fallback_frame(frame)
        except Exception as e:
            return {
                'pose': self.current_pose.copy(),
                'position': self.current_pose[:3, 3].copy(),
                'tracking_quality': 0.0,
                'tracking_state': "ERROR",
                'message': f"SLAM error: {e}",
                'is_initialized': False,
                'trajectory_length': len(self.trajectory),
                'num_map_points': 0,
                'performance': self.performance_stats.copy()
            }

    def _process_pyslam_frame(self, frame: np.ndarray) -> Dict:
        """Process frame using pySLAM"""
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
        
        # Process through pySLAM
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
        map_points = self.get_map_points()
        
        # Create result
        result = {
            'pose': self.current_pose.copy(),
            'position': self.current_pose[:3, 3].copy(),
            'tracking_quality': 0.9 if tracking_state == "OK" else 0.0,
            'tracking_state': tracking_state,
            'message': f"pySLAM frame {self.frame_count}",
            'is_initialized': self.is_initialized,
            'trajectory_length': len(self.trajectory),
            'num_map_points': len(map_points),
            'performance': self.performance_stats.copy()
        }
        
        self.frame_count += 1
        return result

    def _process_fallback_frame(self, frame: np.ndarray) -> Dict:
        """Process frame using fallback OpenCV SLAM"""
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Detect keypoints and descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if self.prev_frame is not None and self.prev_keypoints is not None and self.prev_descriptors is not None:
            # Match features
            matches = self.matcher.match(self.prev_descriptors, descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Filter good matches
            good_matches = matches[:50]  # Take top 50 matches
            
            if len(good_matches) > 10:
                # Extract matched keypoints
                src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Estimate motion using essential matrix
                E, mask = cv2.findEssentialMat(src_pts, dst_pts, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                
                if E is not None:
                    # Recover pose
                    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, self.camera_matrix)
                    
                    # Update pose
                    pose_delta = np.eye(4)
                    pose_delta[:3, :3] = R
                    pose_delta[:3, 3] = t.flatten()
                    self.current_pose = self.current_pose @ pose_delta
                    
                    # Add to trajectory
                    self.trajectory.append(self.current_pose.copy())
        
        # Update previous frame data
        self.prev_frame = gray.copy()
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors
        
        # Create result
        result = {
            'pose': self.current_pose.copy(),
            'position': self.current_pose[:3, 3].copy(),
            'tracking_quality': 0.8 if len(keypoints) > 50 else 0.3,
            'tracking_state': "OK" if len(keypoints) > 50 else "LOST",
            'message': f"Fallback SLAM frame {self.frame_count}",
            'is_initialized': True,
            'trajectory_length': len(self.trajectory),
            'num_map_points': 0,  # Fallback doesn't maintain map points
            'performance': self.performance_stats.copy()
        }
        
        self.frame_count += 1
        return result

    def get_map_points(self) -> np.ndarray:
        """Get map points"""
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
            pass
        return np.array([]).reshape(0, 3)

    def reset(self):
        """Reset SLAM system"""
        self.frame_count = 0
        self.is_initialized = False
        self.current_pose = np.eye(4)
        self.trajectory = []
        self.performance_stats = {}
        if hasattr(self.slam, 'reset'):
            self.slam.reset()

    def shutdown(self):
        """Shutdown SLAM system"""
        if hasattr(self.slam, 'shutdown'):
            self.slam.shutdown()

if __name__ == "__main__":
    # Test the server
    config = {
        'camera.width': 640,
        'camera.height': 480,
        'camera.fps': 30,
        'mapping3d.fx': 500,
        'mapping3d.fy': 500,
        'slam.feature_type': 'ORB',
        'slam.orb_features': 2000,
        'slam.loop_closure': False
    }
    
    server = PySLAMServer(config)
    print("pySLAM server ready")
