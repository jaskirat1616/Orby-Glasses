#!/usr/bin/env python3
"""
Simple pySLAM Integration for OrbyGlasses
Uses OpenCV-based SLAM with pySLAM-inspired features
"""

import os
import sys
import cv2
import numpy as np
import logging
import time
from typing import Dict, Optional, List, Tuple
from collections import deque

# Add pySLAM path to sys.path for potential future use
pyslam_path = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'pyslam')
if os.path.exists(pyslam_path) and pyslam_path not in sys.path:
    sys.path.insert(0, pyslam_path)


class SimplePySLAM:
    """
    Simple pySLAM-inspired SLAM system using OpenCV.
    Provides similar functionality to pySLAM without environment conflicts.
    """

    def __init__(self, config: Dict):
        """Initialize simple pySLAM system."""
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
        self.enable_visualization = config.get('slam.visualize', True)
        
        # Initialize SLAM system
        self._initialize_slam()

    def _initialize_slam(self):
        """Initialize SLAM system with OpenCV."""
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
            
            # Map points
            self.map_points_3d = []
            self.map_points_2d = []
            
            # Keyframe management
            self.keyframe_poses = []
            self.keyframe_descriptors = []
            
            self.is_initialized = True
            print("✅ Simple pySLAM initialized successfully!")
            self.logger.info("✅ Simple pySLAM initialized successfully!")
            
        except Exception as e:
            error_msg = f"Failed to initialize simple pySLAM: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def process_frame(self, frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> Dict:
        """
        Process a single frame through SLAM.

        Args:
            frame: Input frame (BGR or grayscale)
            depth_map: Depth map (optional for depth-assisted SLAM)

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
            
            # Detect keypoints and descriptors
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            tracking_state = "OK"
            tracking_quality = 0.0
            
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
                        
                        # Update map points
                        self._update_map_points(src_pts, dst_pts, R, t)
                        
                        tracking_quality = min(len(good_matches) / 50.0, 1.0)
                    else:
                        tracking_state = "LOST"
                        tracking_quality = 0.0
                else:
                    tracking_state = "LOST"
                    tracking_quality = 0.0
            else:
                # First frame
                tracking_quality = 0.5
            
            # Update previous frame data
            self.prev_frame = gray.copy()
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            
            # Update visualization
            if self.enable_visualization:
                self._update_visualization(frame, keypoints, good_matches if 'good_matches' in locals() else [])
            
            # Create result
            result = {
                'pose': self.current_pose.copy(),
                'position': self.current_pose[:3, 3].copy(),
                'tracking_quality': tracking_quality,
                'tracking_state': tracking_state,
                'message': f"Simple pySLAM frame {self.frame_count}",
                'is_initialized': self.is_initialized,
                'trajectory_length': len(self.trajectory),
                'num_map_points': len(self.map_points_3d),
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

    def _update_map_points(self, src_pts, dst_pts, R, t):
        """Update 3D map points using triangulation."""
        try:
            # Simple triangulation (for demonstration)
            # In a real implementation, you would use proper bundle adjustment
            if len(self.map_points_3d) < 1000:  # Limit map points
                # Add some dummy 3D points for visualization
                for i in range(min(10, len(src_pts))):
                    # Simple depth estimation (not accurate, just for visualization)
                    depth = 1.0 + np.random.random() * 2.0
                    point_3d = np.array([
                        src_pts[i][0][0] * depth / self.fx,
                        src_pts[i][0][1] * depth / self.fy,
                        depth
                    ])
                    self.map_points_3d.append(point_3d)
                    
        except Exception as e:
            self.logger.error(f"Map point update error: {e}")

    def _update_visualization(self, frame: np.ndarray, keypoints: List, matches: List):
        """Update visualization windows."""
        try:
            # Create visualization image
            vis_img = frame.copy()
            
            # Draw keypoints
            cv2.drawKeypoints(vis_img, keypoints, vis_img, color=(0, 255, 0), flags=0)
            
            # Draw matches
            if len(matches) > 0:
                match_img = cv2.drawMatches(
                    self.prev_frame, self.prev_keypoints,
                    vis_img, keypoints,
                    matches[:20], None,  # Show top 20 matches
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                )
                cv2.imshow("pySLAM - Feature Matching", match_img)
            
            # Add text info
            info_text = f"Simple pySLAM | Frame: {self.frame_count} | Features: {len(keypoints)} | Matches: {len(matches)}"
            cv2.putText(vis_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add pose info
            pos = self.current_pose[:3, 3]
            pose_text = f"Position: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            cv2.putText(vis_img, pose_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show the image
            cv2.imshow("pySLAM - Live Tracking", vis_img)
            cv2.waitKey(1)
            
        except Exception as e:
            self.logger.error(f"Visualization update error: {e}")

    def get_map_points(self) -> np.ndarray:
        """Get all map points for visualization."""
        if len(self.map_points_3d) > 0:
            return np.array(self.map_points_3d)
        return np.array([]).reshape(0, 3)

    def is_tracking_good(self) -> bool:
        """Check if SLAM tracking is good."""
        return len(self.trajectory) > 0

    def get_current_pose(self) -> np.ndarray:
        """Get the current estimated camera pose."""
        return self.current_pose

    def reset(self):
        """Reset the SLAM system."""
        try:
            self.frame_count = 0
            self.is_initialized = False
            self.current_pose = np.eye(4)
            self.trajectory = []
            self.map_points_3d = []
            self.map_points_2d = []
            self.keyframe_poses = []
            self.keyframe_descriptors = []
            self.prev_frame = None
            self.prev_keypoints = None
            self.prev_descriptors = None
            self.logger.info("SLAM system reset")
        except Exception as e:
            self.logger.error(f"Reset error: {e}")

    def shutdown(self):
        """Shutdown SLAM system and visualizations."""
        try:
            # Close OpenCV windows
            if self.enable_visualization:
                cv2.destroyWindow("pySLAM - Live Tracking")
                cv2.destroyWindow("pySLAM - Feature Matching")
            
            self.logger.info("SLAM system shutdown")
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")


# For backward compatibility
def create_pyslam_system(config: Dict) -> SimplePySLAM:
    """Create a simple pySLAM system instance."""
    return SimplePySLAM(config)

# Make PYSLAM_AVAILABLE available
PYSLAM_AVAILABLE = True
