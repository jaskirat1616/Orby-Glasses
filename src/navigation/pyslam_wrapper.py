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

try:
    # Import pyslam components
    from slam import Slam, SlamConfig
    from camera import Camera, PinholeCamera
    from feature_tracker import FeatureTrackerTypes
    
    # Import visualization components
    from viewer3D import Viewer3D
    from display2D import Display2D
    
    PYSLAM_AVAILABLE = True
except ImportError as e:
    PYSLAM_AVAILABLE = False
    logging.warning(f"pyslam not available: {e}")


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

        # Camera parameters
        width = config.get('camera.width', 320)
        height = config.get('camera.height', 240)
        fps = config.get('camera.fps', 30)
        self.fx = config.get('mapping3d.fx', 500)
        self.fy = config.get('mapping3d.fy', 500)
        self.cx = width / 2
        self.cy = height / 2

        # Create pyslam camera model
        self.camera = PinholeCamera(
            width=width,
            height=height,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            fps=fps
        )

        # Configure SLAM system
        slam_config = SlamConfig()
        slam_config.set_camera(self.camera)

        # Feature detector configuration
        feature_type = config.get('slam.feature_type', 'ORB')
        if feature_type == 'ORB':
            slam_config.feature_tracker_type = FeatureTrackerTypes.ORB
        elif feature_type == 'SIFT':
            slam_config.feature_tracker_type = FeatureTrackerTypes.SIFT
        elif feature_type == 'SUPERPOINT':
            slam_config.feature_tracker_type = FeatureTrackerTypes.SUPERPOINT
        else:
            slam_config.feature_tracker_type = FeatureTrackerTypes.ORB

        # SLAM configuration
        slam_config.num_features = config.get('slam.orb_features', 2000)
        slam_config.enable_loop_closing = config.get('slam.loop_closure', False)
        slam_config.enable_local_mapping = True

        # Initialize SLAM system
        try:
            self.slam = Slam(slam_config)
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
                # Initialize 3D viewer (map visualization)
                self.viewer3d = Viewer3D()
                self.viewer3d.start()
                self.logger.info("✅ pySLAM 3D Viewer enabled - Real-time 3D map!")
                
                # Initialize 2D display (feature tracking visualization)
                self.display2d = Display2D()
                self.logger.info("✅ pySLAM 2D Display enabled - Feature tracking!")
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
            self.slam.track(gray, timestamp)
            
            # Update visualizations if enabled
            if self.enable_visualization:
                try:
                    # Update 3D viewer with current map state
                    if self.viewer3d is not None and self.slam.map is not None:
                        self.viewer3d.draw_map(self.slam.map, self.slam.tracking)
                    
                    # Update 2D display with feature tracking
                    if self.display2d is not None:
                        self.display2d.draw(self.slam.tracking, gray)
                except Exception as e:
                    # Don't crash if visualization fails
                    pass

            # Get tracking state
            is_tracking_good = self.slam.is_tracking_good()

            if is_tracking_good:
                # Get current pose from SLAM
                # pySLAM returns camera pose as 4x4 matrix
                pose = self.slam.get_current_pose()

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
            return self._get_default_result()

    def get_map_points(self) -> np.ndarray:
        """
        Get all map points for visualization.

        Returns:
            Nx3 array of map point positions
        """
        try:
            map_points = self.slam.get_map_points()
            if map_points is not None and len(map_points) > 0:
                return np.array(map_points)
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
