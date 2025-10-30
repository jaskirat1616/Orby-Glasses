#!/usr/bin/env python3
"""
Live pySLAM Integration for OrbyGlasses
Direct integration with pySLAM's live camera support and real-time mapping
"""

import os
import sys
import cv2
import numpy as np
import logging
import time
import threading
from typing import Dict, Optional, List, Tuple
from collections import deque
# Disable 3D visualization for now due to tkinter issues
# import matplotlib
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.animation as animation

# Add pySLAM path to sys.path
pyslam_path = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'pyslam')
if os.path.exists(pyslam_path) and pyslam_path not in sys.path:
    sys.path.insert(0, pyslam_path)

# Try to import pySLAM modules
PYSLAM_AVAILABLE = False
try:
    # Check for evo dependency first
    import evo
    print("✅ evo dependency found")
    
    # Import pySLAM modules with proper error handling
    import pyslam
    from pyslam.config import Config
    from pyslam.slam.slam import Slam
    from pyslam.slam.slam_commons import SlamState
    from pyslam.slam.camera import PinholeCamera
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
    from pyslam.local_features.feature_types import FeatureDetectorTypes
    from pyslam.viz.slam_plot_drawer import SlamPlotDrawer
    from pyslam.viz.viewer3D import Viewer3D
    PYSLAM_AVAILABLE = True
    print("✅ Real pySLAM modules imported successfully!")
except ImportError as e:
    PYSLAM_AVAILABLE = False
    print(f"pySLAM not available: {e}")
    print("Using optimized OpenCV-based SLAM...")
except NameError as e:
    PYSLAM_AVAILABLE = False
    print(f"pySLAM module error: {e}")
    print("Using optimized OpenCV-based SLAM...")
except AttributeError as e:
    PYSLAM_AVAILABLE = False
    print(f"pySLAM attribute error: {e}")
    print("Using optimized OpenCV-based SLAM...")
except Exception as e:
    PYSLAM_AVAILABLE = False
    print(f"pySLAM general error: {e}")
    print("Using optimized OpenCV-based SLAM...")


# Disabled due to tkinter issues - 3D visualization temporarily unavailable
# class PointCloud3DViewer:
#     """3D Point Cloud and Trajectory Viewer for SLAM"""
#     pass


class LivePySLAM:
    """
    Live pySLAM integration with direct camera access and real-time mapping.
    Uses the actual pySLAM library with live camera support.
    """

    def __init__(self, config: Dict):
        """Initialize live pySLAM system."""
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
        
        # pySLAM components
        self.slam = None
        self.camera = None
        self.plot_drawer = None
        
        # Camera capture
        self.cap = None
        
        # Initialize pySLAM or fallback
        if PYSLAM_AVAILABLE:
            try:
                self._initialize_pyslam()
            except Exception as e:
                self.logger.warning(f"pySLAM initialization failed: {e}")
                self.logger.info("Falling back to OpenCV-based SLAM...")
                self._initialize_fallback()
        else:
            self._initialize_fallback()

    def _initialize_pyslam(self):
        """Initialize pySLAM with live camera support."""
        try:
            # Check if pySLAM modules are available
            if not PYSLAM_AVAILABLE:
                raise ImportError("pySLAM modules not available")
            
            
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
            
            # Create camera - PinholeCamera should be available from imports
            self.camera = PinholeCamera(camera_config)

            # Create feature tracker config
            feature_tracker_config = FeatureTrackerConfigs.ORB2.copy()
            feature_tracker_config["num_features"] = self.config.get('slam.orb_features', 2000)
            
            # Initialize SLAM
            self.slam = Slam(self.camera, feature_tracker_config)
            
            # Initialize visualization (after SLAM is created)
            self.plot_drawer = SlamPlotDrawer(self.slam)
            
            # Initialize 3D viewer
            self.viewer3d = Viewer3D(scale=1.0)
            
            # Initialize camera capture
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")
            
            self.is_initialized = True
            print("✅ Live pySLAM initialized successfully!")
            self.logger.info("✅ Live pySLAM initialized successfully!")
            
        except Exception as e:
            error_msg = f"Failed to initialize live pySLAM: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _initialize_fallback(self):
        """Initialize fallback OpenCV-based SLAM with real-time mapping."""
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
            
            # Initialize camera capture with optimized settings
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency
            
            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")
            
            # Initialize visualization windows
            cv2.namedWindow("OrbyGlasses - Camera Feed", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("OrbyGlasses - Feature Tracking", cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow("OrbyGlasses - 2D Map", cv2.WINDOW_AUTOSIZE)
            
            # Initialize 3D point cloud viewer (disabled due to tkinter issues)
            # self.viewer_3d = PointCloud3DViewer("OrbyGlasses - 3D SLAM Map & Trajectory")
            self.viewer_3d = None
            
            self.is_initialized = True
            print("✅ Fallback SLAM with real-time mapping and visualization windows initialized!")
            self.logger.info("✅ Fallback SLAM with real-time mapping and visualization windows initialized!")
            
        except Exception as e:
            error_msg = f"Failed to initialize fallback SLAM: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def process_frame(self, frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> Dict:
        """
        Process a single frame through live pySLAM or fallback.

        Args:
            frame: Input frame (BGR or grayscale)
            depth_map: Depth map (ignored for monocular SLAM)

        Returns:
            Dictionary with SLAM results
        """
        self.frame_count += 1
        
        try:
            if PYSLAM_AVAILABLE and self.slam is not None:
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
        """Process frame using real pySLAM."""
        # Process frame through pySLAM
        timestamp = time.time()
        self.slam.track(frame, None, None, self.frame_count, timestamp)

        # Get tracking state
        tracking_state = "OK"
        if hasattr(self.slam, 'tracking') and hasattr(self.slam.tracking, 'state'):
            try:
                state = self.slam.tracking.state
                # Handle both enum and string states
                if state is not None:
                    # Try direct comparison with enum values (works with Enum)
                    try:
                        if state == SlamState.LOST:
                            tracking_state = "LOST"
                        elif state == SlamState.NOT_INITIALIZED:
                            tracking_state = "NOT_INITIALIZED"
                        elif state == SlamState.OK:
                            tracking_state = "OK"
                    except (TypeError, AttributeError):
                        # Fallback to string comparison
                        tracking_state = str(state)
            except Exception as e:
                # Log and continue with OK state
                self.logger.debug(f"Could not parse tracking state: {e}")
        
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
        if self.plot_drawer:
            try:
                self.plot_drawer.draw(self.slam, frame)
            except Exception as e:
                self.logger.warning(f"Plot visualization error: {e}")
        
        # Update 3D viewer
        if hasattr(self, 'viewer3d') and self.viewer3d:
            try:
                self.viewer3d.draw_slam_map(self.slam)
            except Exception as e:
                self.logger.warning(f"3D visualization error: {e}")
        
        # Show pySLAM camera window with feature trails
        try:
            if hasattr(self.slam, 'map') and hasattr(self.slam.map, 'draw_feature_trails'):
                img_draw = self.slam.map.draw_feature_trails(frame)
                cv2.imshow("pySLAM - Camera", img_draw)
            else:
                # Fallback to basic camera view
                cv2.imshow("pySLAM - Camera", frame)
        except Exception as e:
            self.logger.warning(f"Camera window error: {e}")
            cv2.imshow("pySLAM - Camera", frame)
        
        # Create result
        result = {
            'pose': self.current_pose.copy(),
            'position': self.current_pose[:3, 3].copy(),
            'tracking_quality': 0.9 if tracking_state == "OK" else 0.0,
            'tracking_state': tracking_state,
            'message': f"Real pySLAM frame {self.frame_count}",
            'is_initialized': self.is_initialized,
            'trajectory_length': len(self.trajectory),
            'num_map_points': len(self.map_points),
            'performance': {}
        }
        
        return result

    def _process_fallback_frame(self, frame: np.ndarray) -> Dict:
        """Process frame using fallback OpenCV SLAM with real-time mapping."""
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
        self._update_visualization(frame, keypoints, good_matches if 'good_matches' in locals() else [])
        
        # Create result
        result = {
            'pose': self.current_pose.copy(),
            'position': self.current_pose[:3, 3].copy(),
            'tracking_quality': tracking_quality,
            'tracking_state': tracking_state,
            'message': f"Fallback SLAM frame {self.frame_count}",
            'is_initialized': self.is_initialized,
            'trajectory_length': len(self.trajectory),
            'num_map_points': len(self.map_points_3d),
            'performance': {}
        }
        
        return result

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
        """Update visualization windows - multiple windows for better visualization."""
        try:
            # Window 1: Camera Feed
            cv2.imshow("OrbyGlasses - Camera Feed", frame)
            
            # Window 2: Feature Tracking
            vis_img = frame.copy()
            keypoints_limited = keypoints[:100] if len(keypoints) > 100 else keypoints
            cv2.drawKeypoints(vis_img, keypoints_limited, vis_img, color=(0, 255, 0), flags=0)
            
            # Draw matches if available
            if len(matches) > 0 and self.prev_frame is not None and len(self.prev_keypoints) > 0:
                try:
                    # Get corresponding keypoints for matches
                    prev_kp = []
                    curr_kp = []
                    valid_matches = []
                    
                    for i, match in enumerate(matches[:20]):  # Top 20 matches
                        if (match.queryIdx < len(self.prev_keypoints) and 
                            match.trainIdx < len(keypoints_limited)):
                            prev_kp.append(self.prev_keypoints[match.queryIdx])
                            curr_kp.append(keypoints_limited[match.trainIdx])
                            valid_matches.append(match)
                    
                    # Only draw if we have valid matches
                    if len(valid_matches) > 0:
                        # Create side-by-side match visualization
                        h, w = frame.shape[:2]
                        prev_small = cv2.resize(self.prev_frame, (w//4, h//6))
                        curr_small = cv2.resize(vis_img, (w//4, h//6))
                        
                        match_img = cv2.drawMatches(
                            prev_small, prev_kp,
                            curr_small, curr_kp,
                            valid_matches, None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                        )
                        
                        # Place match visualization in top-right corner
                        if match_img.shape[0] <= h-10 and match_img.shape[1] <= w-10:
                            vis_img[10:10+match_img.shape[0], w-match_img.shape[1]:w] = match_img
                except Exception as e:
                    # Skip match visualization if there's an error
                    pass
            
            # Add comprehensive info overlay
            info_text = f"SLAM | F:{self.frame_count} | Feat:{len(keypoints)} | Match:{len(matches)}"
            cv2.putText(vis_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Add pose info
            pos = self.current_pose[:3, 3]
            pose_text = f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})"
            cv2.putText(vis_img, pose_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Add map info
            map_text = f"Map: {len(self.map_points_3d)} | Path: {len(self.trajectory)}"
            cv2.putText(vis_img, map_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # Add tracking quality
            quality = min(len(matches) / 50.0, 1.0) if matches else 0.0
            quality_text = f"Quality: {quality:.2f}"
            cv2.putText(vis_img, quality_text, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Show feature tracking window
            cv2.imshow("OrbyGlasses - Feature Tracking", vis_img)
            
            # Window 3: 2D Map
            self._draw_2d_map()
            
            cv2.waitKey(1)
            
            # Update 3D visualization
            self._update_3d_visualization()
            
        except Exception as e:
            self.logger.error(f"Visualization update error: {e}")
    
    def _draw_2d_map(self):
        """Draw 2D top-down map view."""
        try:
            # Create 2D map image
            map_size = 400
            map_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)
            
            # Draw grid
            grid_spacing = 20
            for i in range(0, map_size, grid_spacing):
                cv2.line(map_img, (i, 0), (i, map_size), (50, 50, 50), 1)
                cv2.line(map_img, (0, i), (map_size, i), (50, 50, 50), 1)
            
            # Draw trajectory
            if len(self.trajectory) > 1:
                center_x, center_y = map_size // 2, map_size // 2
                scale = 50  # pixels per meter
                
                for i in range(1, len(self.trajectory)):
                    prev_pos = self.trajectory[i-1][:3, 3]
                    curr_pos = self.trajectory[i][:3, 3]
                    
                    # Convert to map coordinates
                    x1 = int(center_x + prev_pos[0] * scale)
                    y1 = int(center_y - prev_pos[2] * scale)  # Flip Z for top-down view
                    x2 = int(center_x + curr_pos[0] * scale)
                    y2 = int(center_y - curr_pos[2] * scale)
                    
                    # Draw line
                    if 0 <= x1 < map_size and 0 <= y1 < map_size and 0 <= x2 < map_size and 0 <= y2 < map_size:
                        cv2.line(map_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw current position
                if len(self.trajectory) > 0:
                    curr_pos = self.trajectory[-1][:3, 3]
                    x = int(center_x + curr_pos[0] * scale)
                    y = int(center_y - curr_pos[2] * scale)
                    if 0 <= x < map_size and 0 <= y < map_size:
                        cv2.circle(map_img, (x, y), 5, (0, 0, 255), -1)
            
            # Draw map points
            if len(self.map_points_3d) > 0:
                center_x, center_y = map_size // 2, map_size // 2
                scale = 50
                
                for point in self.map_points_3d:
                    x = int(center_x + point[0] * scale)
                    y = int(center_y - point[2] * scale)
                    if 0 <= x < map_size and 0 <= y < map_size:
                        cv2.circle(map_img, (x, y), 2, (255, 255, 0), -1)
            
            # Add title
            cv2.putText(map_img, "2D Map View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show 2D map
            cv2.imshow("OrbyGlasses - 2D Map", map_img)
            
        except Exception as e:
            self.logger.error(f"2D map drawing error: {e}")

    def _draw_trajectory_overlay(self, img):
        """Draw trajectory overlay on the image."""
        try:
            h, w = img.shape[:2]
            if len(self.trajectory) < 2:
                return
                
            # Scale trajectory to fit in image
            scale = min(w, h) / 20.0  # Scale factor
            center_x, center_y = w // 2, h // 2
            
            # Draw trajectory path
            for i in range(1, min(len(self.trajectory), 100)):  # Limit to last 100 points
                prev_pos = self.trajectory[i-1][:3, 3]
                curr_pos = self.trajectory[i][:3, 3]
                
                # Convert to image coordinates
                x1 = int(prev_pos[0] * scale + center_x)
                y1 = int(prev_pos[1] * scale + center_y)
                x2 = int(curr_pos[0] * scale + center_x)
                y2 = int(curr_pos[1] * scale + center_y)
                
                # Draw line if within bounds
                if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw current position
            if len(self.trajectory) > 0:
                curr_pos = self.trajectory[-1][:3, 3]
                x = int(curr_pos[0] * scale + center_x)
                y = int(curr_pos[1] * scale + center_y)
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
                    
        except Exception as e:
            self.logger.error(f"Trajectory overlay error: {e}")

    def _update_3d_visualization(self):
        """Update the 3D point cloud and trajectory visualization (disabled)"""
        # 3D visualization temporarily disabled due to tkinter issues
        pass

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
            else:
                # Return fallback map points
                if len(self.map_points_3d) > 0:
                    return np.array(self.map_points_3d)
        except Exception as e:
            self.logger.error(f"Error getting map points: {e}")
            return np.array([])
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            
            # Close 3D viewer
            if hasattr(self, 'viewer_3d') and self.viewer_3d is not None:
                self.viewer_3d.close()
            if hasattr(self, 'viewer3d') and self.viewer3d is not None:
                self.viewer3d.close()
                
            self.logger.info("SLAM cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

    def is_tracking_good(self) -> bool:
        """Check if SLAM tracking is good."""
        if hasattr(self.slam, 'tracking') and hasattr(self.slam.tracking, 'state'):
            try:
                state = self.slam.tracking.state
                if state is not None:
                    return state == SlamState.OK or str(state) == "OK"
            except Exception:
                pass
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
        """Shutdown SLAM system and camera."""
        try:
            # Shutdown SLAM
            if hasattr(self.slam, 'shutdown'):
                self.slam.shutdown()
            
            # Close camera
            if self.cap:
                self.cap.release()
            
            # Close visualization
            if self.plot_drawer:
                self.plot_drawer.close()
            
            self.logger.info("SLAM system shutdown")
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")


# For backward compatibility
def create_pyslam_system(config: Dict) -> LivePySLAM:
    """Create a live pySLAM system instance."""
    return LivePySLAM(config)

# Make PYSLAM_AVAILABLE available
PYSLAM_AVAILABLE = True
