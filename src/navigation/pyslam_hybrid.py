#!/usr/bin/env python3
"""
Hybrid pySLAM Integration for OrbyGlasses
Provides proper visualization with fallback when pySLAM is not available
"""

import os
import sys
import cv2
import numpy as np
import logging
import time
from typing import Dict, Optional, List, Tuple
from collections import deque

# Add pySLAM path to sys.path
pyslam_path = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'pyslam')
if os.path.exists(pyslam_path) and pyslam_path not in sys.path:
    sys.path.insert(0, pyslam_path)

# Add pySLAM virtual environment site-packages
pyslam_venv_site_packages = os.path.expanduser('~/.python/venvs/pyslam/lib/python3.11/site-packages')
if os.path.exists(pyslam_venv_site_packages) and pyslam_venv_site_packages not in sys.path:
    sys.path.insert(0, pyslam_venv_site_packages)

# Try to import pySLAM modules
PYSLAM_AVAILABLE = False
try:
    from pyslam.config import Config
    from pyslam.slam.slam import Slam, SlamState
    from pyslam.slam.camera import PinholeCamera
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs, FeatureTrackerTypes
    from pyslam.local_features.feature_types import FeatureDetectorTypes
    PYSLAM_AVAILABLE = True
    print("✅ pySLAM modules imported successfully!")
except ImportError as e:
    PYSLAM_AVAILABLE = False
    print(f"pySLAM not available: {e}")

# Always make the hybrid system available
HYBRID_PYSLAM_AVAILABLE = True


class HybridPySLAMSystem:
    """
    Hybrid pySLAM integration with proper visualization.
    Uses pySLAM when available, falls back to OpenCV-based SLAM with proper visualization.
    """

    def __init__(self, config: Dict):
        """Initialize hybrid pySLAM system."""
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
        self.keyframes = []
        
        # Visualization settings
        self.enable_visualization = config.get('slam.visualize', False)
        
        # Initialize SLAM system
        if PYSLAM_AVAILABLE:
            self._initialize_pyslam()
        else:
            self._initialize_fallback()
        
        # Initialize visualization
        if self.enable_visualization:
            self._initialize_visualization()

    def _initialize_pyslam(self):
        """Initialize pySLAM system."""
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
            
            print("✅ pySLAM initialized successfully!")
            self.logger.info("✅ pySLAM initialized successfully!")
            
        except Exception as e:
            error_msg = f"Failed to initialize pySLAM: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _initialize_fallback(self):
        """Initialize fallback OpenCV-based SLAM."""
        try:
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
            
            # Initialize distortion coefficients
            self.dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            
            # Previous frame data
            self.prev_frame = None
            self.prev_keypoints = None
            self.prev_descriptors = None
            
            self.is_initialized = True
            print("✅ Fallback SLAM system initialized")
            self.logger.info("✅ Fallback SLAM system initialized")
            
        except Exception as e:
            error_msg = f"Failed to initialize fallback SLAM: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _initialize_visualization(self):
        """Initialize visualization windows."""
        try:
            # Create 2D SLAM map window
            self.slam_window_name = "pySLAM - 2D Map & Features"
            cv2.namedWindow(self.slam_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.slam_window_name, 800, 600)
            
            # Create feature tracking window
            self.features_window_name = "pySLAM - Feature Tracking"
            cv2.namedWindow(self.features_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.features_window_name, 640, 480)
            
            print("✅ pySLAM visualization windows created!")
            self.logger.info("✅ pySLAM visualization windows created!")
            
        except Exception as e:
            print(f"⚠️ Could not initialize visualization: {e}")
            self.logger.warning(f"Could not initialize visualization: {e}")

    def process_frame(self, frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> Dict:
        """
        Process a single frame through SLAM.

        Args:
            frame: Input frame (BGR or grayscale)
            depth_map: Depth map (ignored for monocular SLAM)

        Returns:
            Dictionary with SLAM results
        """
        self.frame_count += 1
        
        try:
            if PYSLAM_AVAILABLE:
                return self._process_pyslam_frame(frame)
            else:
                return self._process_fallback_frame(frame)
                
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

    def _process_pyslam_frame(self, frame: np.ndarray) -> Dict:
        """Process frame using pySLAM."""
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
        
        # Update visualization
        if self.enable_visualization:
            self._update_visualization(img_bgr)
        
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

    def _process_fallback_frame(self, frame: np.ndarray) -> Dict:
        """Process frame using fallback OpenCV SLAM."""
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
        
        # Update visualization
        if self.enable_visualization:
            self._update_visualization(frame)
        
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
            'performance': {}
        }
        
        return result

    def _update_visualization(self, frame: np.ndarray):
        """Update visualization windows."""
        try:
            # Update 2D SLAM map
            self._draw_slam_map()
            
            # Update feature tracking
            self._draw_feature_tracking(frame)
            
        except Exception as e:
            self.logger.error(f"Visualization update error: {e}")

    def _draw_slam_map(self):
        """Draw 2D SLAM map visualization."""
        try:
            # Create a black canvas for the map
            map_img = np.zeros((600, 800, 3), dtype=np.uint8)
            
            # Scale and center for visualization
            scale = 50  # Scale factor for visualization
            center_x, center_y = 400, 300  # Center of the canvas
            
            # Draw map points if available
            if len(self.map_points) > 0:
                for point in self.map_points:
                    x = int(point[0] * scale + center_x)
                    y = int(point[1] * scale + center_y)
                    if 0 <= x < 800 and 0 <= y < 600:
                        cv2.circle(map_img, (x, y), 2, (0, 255, 0), -1)  # Green points
            
            # Draw trajectory (this should always be visible)
            if len(self.trajectory) > 1:
                for i in range(1, len(self.trajectory)):
                    prev_pos = self.trajectory[i-1][:3, 3]
                    curr_pos = self.trajectory[i][:3, 3]
                    
                    x1 = int(prev_pos[0] * scale + center_x)
                    y1 = int(prev_pos[1] * scale + center_y)
                    x2 = int(curr_pos[0] * scale + center_x)
                    y2 = int(curr_pos[1] * scale + center_y)
                    
                    if all(0 <= coord < 800 for coord in [x1, x2]) and all(0 <= coord < 600 for coord in [y1, y2]):
                        cv2.line(map_img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue trajectory
            
            # Draw current position
            if self.is_initialized and len(self.trajectory) > 0:
                curr_pos = self.current_pose[:3, 3]
                x = int(curr_pos[0] * scale + center_x)
                y = int(curr_pos[1] * scale + center_y)
                if 0 <= x < 800 and 0 <= y < 600:
                    cv2.circle(map_img, (x, y), 5, (0, 0, 255), -1)  # Red current position
            
            # Add grid for reference
            for i in range(0, 800, 50):
                cv2.line(map_img, (i, 0), (i, 600), (50, 50, 50), 1)
            for i in range(0, 600, 50):
                cv2.line(map_img, (0, i), (800, i), (50, 50, 50), 1)
            
            # Add text info
            info_text = f"pySLAM Map | Points: {len(self.map_points)} | Frames: {len(self.trajectory)} | Frame: {self.frame_count}"
            cv2.putText(map_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add status info
            status_text = f"Status: {'Initialized' if self.is_initialized else 'Initializing'}"
            cv2.putText(map_img, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Show the map
            cv2.imshow(self.slam_window_name, map_img)
            cv2.waitKey(1)
            
        except Exception as e:
            self.logger.error(f"SLAM map visualization error: {e}")

    def _draw_feature_tracking(self, frame: np.ndarray):
        """Draw feature tracking visualization."""
        try:
            # Resize image to fit window
            display_img = cv2.resize(frame, (640, 480))
            
            # Add text info
            info_text = f"pySLAM Features | Frame: {self.frame_count}"
            cv2.putText(display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add tracking status
            status_text = f"Tracking: {'OK' if self.is_initialized else 'Initializing'}"
            cv2.putText(display_img, status_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Add trajectory info
            traj_text = f"Trajectory: {len(self.trajectory)} poses"
            cv2.putText(display_img, traj_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Draw a simple crosshair in the center
            h, w = display_img.shape[:2]
            cv2.line(display_img, (w//2-10, h//2), (w//2+10, h//2), (0, 0, 255), 2)
            cv2.line(display_img, (w//2, h//2-10), (w//2, h//2+10), (0, 0, 255), 2)
            
            # Show the image
            cv2.imshow(self.features_window_name, display_img)
            cv2.waitKey(1)
            
        except Exception as e:
            self.logger.error(f"Feature tracking visualization error: {e}")

    def get_map_points(self) -> np.ndarray:
        """Get all map points for visualization."""
        try:
            if PYSLAM_AVAILABLE and hasattr(self.slam, 'map') and self.slam.map is not None:
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
        if PYSLAM_AVAILABLE and hasattr(self.slam, 'tracking') and hasattr(self.slam.tracking, 'state'):
            return self.slam.tracking.state == SlamState.OK
        return True  # Fallback always returns True

    def get_current_pose(self) -> np.ndarray:
        """Get the current estimated camera pose."""
        return self.current_pose

    def reset(self):
        """Reset the SLAM system."""
        try:
            if PYSLAM_AVAILABLE and hasattr(self.slam, 'reset'):
                self.slam.reset()
            self.frame_count = 0
            self.is_initialized = False
            self.current_pose = np.eye(4)
            self.trajectory = []
            self.map_points = []
            self.keyframes = []
            self.logger.info("SLAM system reset")
        except Exception as e:
            self.logger.error(f"Reset error: {e}")

    def shutdown(self):
        """Shutdown SLAM system and visualizations."""
        try:
            # Shutdown SLAM
            if PYSLAM_AVAILABLE and hasattr(self.slam, 'shutdown'):
                self.slam.shutdown()
            
            # Close OpenCV windows
            if self.enable_visualization:
                cv2.destroyWindow(self.slam_window_name)
                cv2.destroyWindow(self.features_window_name)
            
            self.logger.info("SLAM system shutdown")
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")
