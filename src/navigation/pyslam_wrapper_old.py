"""
pySLAM Integration for OrbyGlasses
Advanced monocular SLAM using the pyslam framework.

pySLAM provides:
- Multiple feature detectors (ORB, SIFT, SuperPoint, etc.)
- Loop closure detection
- Bundle adjustment
- Map persistence
- Production-quality monocular SLAM
"""

import cv2
import numpy as np
import logging
import sys
import os
from typing import Dict, Optional, Tuple, List
from collections import deque
import time

# Add pyslam to Python path
PYSLAM_PATH = os.path.join(os.path.dirname(__file__), '../../third_party/pyslam')
if os.path.exists(PYSLAM_PATH):
    sys.path.insert(0, PYSLAM_PATH)

# Check if pySLAM is available by testing the server
PYSLAM_AVAILABLE = False
try:
    # Test if we can import the pySLAM server
    from navigation.pyslam_server import PySLAMServer
    PYSLAM_AVAILABLE = True
    logging.info("✅ pySLAM server available!")
except ImportError as e:
    PYSLAM_AVAILABLE = False
    logging.warning(f"pySLAM server not available: {e}")
    logging.warning("Make sure pySLAM virtual environment is activated")
    logging.warning("Run: cd third_party/pyslam && source pyenv-activate.sh")


class PySLAMSystem:
    """
    Wrapper for pyslam monocular SLAM system.
    Provides production-quality visual SLAM with advanced features.
    """

    def __init__(self, config):
        """
        Initialize pySLAM system.

        Args:
            config: ConfigManager instance
        """
        if not PYSLAM_AVAILABLE:
            raise ImportError(
                "pyslam not installed. Please run:\n"
                "cd third_party/pyslam && ./scripts/install_all_venv.sh"
            )

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Camera parameters - use actual camera resolution
        self.width = config.get('camera.width', 640)  # Default to 640x480
        self.height = config.get('camera.height', 480)
        fps = config.get('camera.fps', 30)
        self.fx = config.get('mapping3d.fx', 500)
        self.fy = config.get('mapping3d.fy', 500)
        self.cx = self.width / 2
        self.cy = self.height / 2

        # Create camera config for pySLAM
        camera_config = Config()
        camera_config.cam_settings = {
            'Camera.width': self.width,
            'Camera.height': self.height,
            'Camera.fx': self.fx,
            'Camera.fy': self.fy,
            'Camera.cx': self.cx,
            'Camera.cy': self.cy,
            'Camera.fps': fps,
            'Camera.k1': 0.0,  # No distortion
            'Camera.k2': 0.0,
            'Camera.p1': 0.0,
            'Camera.p2': 0.0,
            'Camera.k3': 0.0
        }
        
        # Create pyslam camera model
        self.camera = PinholeCamera(camera_config)

        # Configure SLAM system
        slam_config = Config()
        slam_config.cam_settings = {
            'Camera.width': self.width,
            'Camera.height': self.height,
            'Camera.fx': self.fx,
            'Camera.fy': self.fy,
            'Camera.cx': self.cx,
            'Camera.cy': self.cy,
            'Camera.fps': fps
        }

        # Feature detector configuration
        feature_type = config.get('slam.feature_type', 'ORB')
        if feature_type == 'ORB':
            slam_config.feature_detector_type = FeatureDetectorTypes.ORB
        elif feature_type == 'SIFT':
            slam_config.feature_detector_type = FeatureDetectorTypes.SIFT
        elif feature_type == 'SUPERPOINT':
            slam_config.feature_detector_type = FeatureDetectorTypes.SUPERPOINT
        else:
            slam_config.feature_detector_type = FeatureDetectorTypes.ORB

        # SLAM configuration
        slam_config.num_features = config.get('slam.orb_features', 2000)
        slam_config.enable_loop_closing = config.get('slam.loop_closure', False)
        slam_config.enable_local_mapping = True

        # Create feature tracker config
        feature_tracker_config = FeatureTrackerConfigs.ORB2.copy()
        feature_tracker_config["num_features"] = config.get('slam.orb_features', 2000)
        
        # Initialize SLAM system
        try:
            # Pass the camera object as the first parameter
            self.slam = Slam(self.camera, feature_tracker_config)
            self.logger.info("✅ pySLAM initialized successfully")
            self.logger.info(f"Camera: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
            self.logger.info(f"Feature detector: {feature_type}")
        except Exception as e:
            error_msg = f"Failed to initialize pySLAM: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Initialize visualizations if enabled
        self.enable_visualization = config.get('slam.visualize', False)
        self.viewer3d = None
        self.display2d = None
        
        if self.enable_visualization and PYSLAM_AVAILABLE:
            try:
                # Initialize simple 2D SLAM viewer using OpenCV (no Pangolin)
                self.slam_window_name = "pySLAM - 2D Map & Features"
                cv2.namedWindow(self.slam_window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.slam_window_name, 800, 600)
                self.logger.info("✅ pySLAM 2D Viewer enabled - Real-time SLAM map!")
                
                # Initialize simple 2D display for features
                self.features_window_name = "pySLAM - Feature Tracking"
                cv2.namedWindow(self.features_window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.features_window_name, 640, 480)
                self.logger.info("✅ pySLAM Feature Display enabled - Real-time tracking!")
                
                self.logger.info("✅ pySLAM visualization windows should now be visible!")
                
            except Exception as e:
                self.logger.warning(f"Could not initialize pySLAM visualizations: {e}")
                self.viewer3d = None
                self.display2d = None

        # State
        self.is_initialized = False
        self.frame_count = 0
        self.current_pose = np.eye(4, dtype=np.float32)
        self.tracking_state = "NOT_INITIALIZED"

        # History
        self.pose_history = deque(maxlen=1000)
        self.position_history = deque(maxlen=1000)

        # Timestamp for frame processing
        self.start_time = time.time()

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> Dict:
        """
        Process frame through pySLAM.

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

        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = time.time() - self.start_time

        try:
            # Process frame through pySLAM
            # pySLAM expects 3-channel BGR image, not grayscale
            if len(gray.shape) == 2:
                # Convert grayscale to 3-channel BGR
                img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = gray
            
            # pySLAM track method needs: img, img_right, depth, img_id, timestamp
            img_right = None  # Monocular SLAM - no right image
            depth = None      # Monocular SLAM - no depth
            img_id = self.frame_count  # Use frame count as image ID
            
            self.slam.track(img_bgr, img_right, depth, img_id, timestamp)
            
            # Update visualizations if enabled
            if self.enable_visualization and self.slam is not None:
                try:
                    # Create 2D SLAM map visualization
                    self._draw_slam_map()
                    
                    # Create feature tracking visualization
                    self._draw_feature_tracking(img_bgr)
                    
                except Exception as e:
                    # Don't crash if visualization fails
                    pass

            # Get tracking state - pySLAM uses different method names
            # Try to get tracking state from the slam object
            tracking_state = "UNKNOWN"
            is_tracking_good = False
            
            # Check if we can get tracking state
            if hasattr(self.slam, 'tracking') and hasattr(self.slam.tracking, 'state'):
                tracking_state = str(self.slam.tracking.state)
                # pySLAM states: NO_IMAGES_YET, NOT_INITIALIZED, OK, LOST
                is_tracking_good = tracking_state == "OK"
            elif hasattr(self.slam, 'get_state'):
                tracking_state = str(self.slam.get_state())
                is_tracking_good = tracking_state == "OK"
            else:
                # Fallback: assume tracking is good if we got this far
                is_tracking_good = True
                tracking_state = "OK"

            if is_tracking_good:
                # Get current pose from SLAM
                # pySLAM may use different method names
                pose = None
                if hasattr(self.slam, 'get_current_pose'):
                    pose = self.slam.get_current_pose()
                elif hasattr(self.slam, 'tracking') and hasattr(self.slam.tracking, 'get_current_pose'):
                    pose = self.slam.tracking.get_current_pose()
                elif hasattr(self.slam, 'tracking') and hasattr(self.slam.tracking, 'cur_pose'):
                    pose = self.slam.tracking.cur_pose

                if pose is not None:
                    self.current_pose = pose.astype(np.float32)
                    self.is_initialized = True
                    self.tracking_state = "OK"

                    # Extract position
                    position = self.current_pose[:3, 3].copy()

                    # Update history
                    self.pose_history.append(self.current_pose.copy())
                    self.position_history.append(position.tolist())

                    # Get tracking statistics
                    num_tracked_points = self.slam.get_num_tracked_points()
                    num_map_points = self.slam.get_num_map_points()

                    # Calculate tracking quality
                    tracking_quality = min(1.0, num_tracked_points / 50.0) if num_tracked_points > 0 else 0.5

                    return {
                        'pose': self.current_pose,
                        'position': position.tolist(),
                        'tracking_quality': tracking_quality,
                        'tracking_state': self.tracking_state,
                        'num_map_points': num_map_points,
                        'num_tracked_points': num_tracked_points,
                        'is_keyframe': self.slam.is_keyframe(),
                        'initialized': True
                    }

            # Tracking lost or initializing
            if not self.is_initialized:
                self.tracking_state = "INITIALIZING"
            else:
                self.tracking_state = "LOST"

            return {
                'pose': self.current_pose,
                'position': self.current_pose[:3, 3].tolist(),
                'tracking_quality': 0.0,
                'tracking_state': self.tracking_state,
                'num_map_points': 0,
                'num_tracked_points': 0,
                'is_keyframe': False,
                'initialized': self.is_initialized
            }

        except Exception as e:
            self.logger.error(f"pySLAM processing error: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._get_default_result()

    def _draw_slam_map(self):
        """Draw 2D SLAM map visualization."""
        try:
            # Create a black canvas for the map
            map_img = np.zeros((600, 800, 3), dtype=np.uint8)
            
            # Get map points
            map_points = self.get_map_points()
            
            if len(map_points) > 0:
                # Scale and center the map points
                scale = 50  # Scale factor for visualization
                center_x, center_y = 400, 300  # Center of the canvas
                
                # Draw map points
                for point in map_points:
                    x = int(point[0] * scale + center_x)
                    y = int(point[1] * scale + center_y)
                    if 0 <= x < 800 and 0 <= y < 600:
                        cv2.circle(map_img, (x, y), 2, (0, 255, 0), -1)  # Green points
            
            # Draw trajectory
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
            if self.is_initialized:
                curr_pos = self.current_pose[:3, 3]
                x = int(curr_pos[0] * scale + center_x)
                y = int(curr_pos[1] * scale + center_y)
                if 0 <= x < 800 and 0 <= y < 600:
                    cv2.circle(map_img, (x, y), 5, (0, 0, 255), -1)  # Red current position
            
            # Add text info
            info_text = f"Map Points: {len(map_points)} | Keyframes: {len(self.trajectory)} | Frame: {self.frame_count}"
            cv2.putText(map_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show the map
            cv2.imshow(self.slam_window_name, map_img)
            cv2.waitKey(1)
            
        except Exception as e:
            pass  # Don't crash if visualization fails

    def _draw_feature_tracking(self, img):
        """Draw feature tracking visualization."""
        try:
            # Resize image to fit window
            display_img = cv2.resize(img, (640, 480))
            
            # Add text info
            info_text = f"pySLAM Features | Frame: {self.frame_count}"
            cv2.putText(display_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show the image
            cv2.imshow(self.features_window_name, display_img)
            cv2.waitKey(1)
            
        except Exception as e:
            pass  # Don't crash if visualization fails

    def get_map_points(self) -> np.ndarray:
        """
        Get all map points for visualization.

        Returns:
            Nx3 array of map point positions
        """
        try:
            # Get map points from the SLAM map
            if hasattr(self.slam, 'map') and self.slam.map is not None:
                map_points = self.slam.map.get_points()
                if map_points is not None and len(map_points) > 0:
                    # Extract 3D positions from map points
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

    def get_keyframes(self) -> List:
        """
        Get all keyframe poses.

        Returns:
            List of 4x4 pose matrices
        """
        try:
            keyframes = self.slam.get_keyframes()
            return keyframes if keyframes else []
        except Exception as e:
            self.logger.error(f"Error getting keyframes: {e}")
            return []

    def save_map(self, filename: str):
        """Save map to file."""
        try:
            self.slam.save_map(filename)
            self.logger.info(f"Map saved to: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save map: {e}")

    def load_map(self, filename: str):
        """Load map from file."""
        try:
            self.slam.load_map(filename)
            self.logger.info(f"Map loaded from: {filename}")
        except Exception as e:
            self.logger.error(f"Failed to load map: {e}")

    def reset(self):
        """Reset SLAM system."""
        try:
            self.slam.reset()
            self.is_initialized = False
            self.current_pose = np.eye(4, dtype=np.float32)
            self.tracking_state = "NOT_INITIALIZED"
            self.logger.info("pySLAM reset")
        except Exception as e:
            self.logger.error(f"Failed to reset: {e}")

    def shutdown(self):
        """Shutdown SLAM system and visualizations."""
        try:
            # Stop visualizations
            if self.viewer3d is not None:
                try:
                    self.viewer3d.stop()
                    self.logger.info("✅ 3D Viewer stopped")
                except Exception as e:
                    self.logger.warning(f"Could not stop 3D viewer: {e}")
            
            if self.display2d is not None:
                try:
                    self.display2d.quit()
                    self.logger.info("✅ 2D Display stopped")
                except Exception as e:
                    self.logger.warning(f"Could not stop 2D display: {e}")
            
            # Shutdown SLAM
            self.slam.shutdown()
            self.logger.info("pySLAM shutdown")
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

    def get_position(self) -> np.ndarray:
        """Get current camera position."""
        return self.current_pose[:3, 3]

    def get_pose(self) -> np.ndarray:
        """Get current camera pose."""
        return self.current_pose

    def _get_default_result(self) -> Dict:
        """Get default result when tracking fails."""
        return {
            'pose': self.current_pose,
            'position': [0, 0, 0],
            'tracking_quality': 0.0,
            'tracking_state': "ERROR",
            'num_map_points': 0,
            'num_tracked_points': 0,
            'is_keyframe': False,
            'initialized': False
        }
