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
    Wrapper for pyslam monocular SLAM system using server approach.
    """
    def __init__(self, config: Dict):
        """Initialize pySLAM system."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize pySLAM server
        try:
            self.server = PySLAMServer(config)
            self.logger.info("✅ pySLAM server initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize pySLAM server: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Initialize visualizations if enabled
        self.enable_visualization = config.get('slam.visualize', False)
        
        if self.enable_visualization:
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

        # State variables
        self.frame_count = 0
        self.is_initialized = False
        self.current_pose = np.eye(4)
        self.trajectory = []
        self.performance_stats = {}

    def process_frame(self, frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> Dict:
        """
        Process a single frame through pySLAM.

        Args:
            frame: Input frame (BGR or grayscale)
            depth_map: Depth map (ignored for monocular SLAM)

        Returns:
            Dictionary with SLAM results
        """
        try:
            # Process frame through pySLAM server
            result = self.server.process_frame(frame)
            
            # Update local state
            self.current_pose = result['pose']
            self.trajectory.append(self.current_pose.copy())
            self.is_initialized = result['is_initialized']
            self.frame_count += 1
            
            # Update visualizations if enabled
            if self.enable_visualization:
                try:
                    # Create 2D SLAM map visualization
                    self._draw_slam_map()
                    
                    # Create feature tracking visualization
                    self._draw_feature_tracking(frame)
                    
                except Exception as e:
                    # Don't crash if visualization fails
                    pass
            
            return result
            
        except Exception as e:
            self.logger.error(f"pySLAM processing error: {e}")
            return {
                'pose': self.current_pose.copy(),
                'position': self.current_pose[:3, 3].copy(),
                'tracking_quality': 0.0,
                'tracking_state': "ERROR",
                'message': f"pySLAM error: {e}",
                'is_initialized': False,
                'trajectory_length': len(self.trajectory),
                'num_map_points': 0,
                'performance': self.performance_stats.copy()
            }

    def get_map_points(self) -> np.ndarray:
        """
        Get all map points for visualization.

        Returns:
            Nx3 array of map point positions
        """
        try:
            return self.server.get_map_points()
        except Exception as e:
            self.logger.error(f"Error getting map points: {e}")
            return np.array([]).reshape(0, 3)

    def reset(self):
        """Reset the SLAM system."""
        try:
            self.server.reset()
            self.frame_count = 0
            self.is_initialized = False
            self.current_pose = np.eye(4)
            self.trajectory = []
            self.performance_stats = {}
            self.logger.info("pySLAM system reset")
        except Exception as e:
            self.logger.error(f"Reset error: {e}")

    def shutdown(self):
        """Shutdown SLAM system and visualizations."""
        try:
            if self.server:
                self.server.shutdown()
            
            # Close OpenCV windows
            if self.enable_visualization:
                cv2.destroyWindow(self.slam_window_name)
                cv2.destroyWindow(self.features_window_name)

            self.logger.info("pySLAM shutdown")
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

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
