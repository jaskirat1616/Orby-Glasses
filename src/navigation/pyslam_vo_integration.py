"""
pySLAM Visual Odometry Integration for OrbyGlasses

Integrates pySLAM's Visual Odometry into the main OrbyGlasses system
with proper threading and visualization support.
"""

import os
import sys
import cv2
import numpy as np
import threading
import time
import logging
import queue
from typing import Dict, Optional, List, Tuple
from collections import deque

# Fix pyslam_utils import issue by creating a mock module
class MockPySLAMUtils:
    """Mock pyslam_utils module to avoid import errors"""
    def __init__(self, *args, **kwargs):
        pass
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

# Add mock pyslam_utils to sys.modules before any pySLAM imports
sys.modules['pyslam_utils'] = MockPySLAMUtils()

# Add pySLAM path to sys.path
pyslam_path = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'pyslam')
if os.path.exists(pyslam_path) and pyslam_path not in sys.path:
    sys.path.insert(0, pyslam_path)

# Try to import pySLAM modules
PYSLAM_VO_AVAILABLE = False
try:
    import pyslam
    from pyslam.config import Config
    from pyslam.slam.visual_odometry import VisualOdometryEducational
    from pyslam.slam.camera import PinholeCamera
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
    from pyslam.local_features.feature_tracker import feature_tracker_factory, FeatureTrackerTypes
    from pyslam.io.dataset_factory import dataset_factory
    from pyslam.io.dataset_types import DatasetType, SensorType
    from pyslam.viz.rerun_interface import Rerun
    from pyslam.utilities.utils_sys import Printer
    PYSLAM_VO_AVAILABLE = True
    print("✅ pySLAM Visual Odometry modules imported successfully!")
except ImportError as e:
    PYSLAM_VO_AVAILABLE = False
    import traceback
    print(f"pySLAM VO not available: {e}")
    print(f"Traceback: {traceback.format_exc()}")
except Exception as e:
    PYSLAM_VO_AVAILABLE = False
    import traceback
    print(f"pySLAM VO error: {e}")
    print(f"Traceback: {traceback.format_exc()}")


class PySLAMVisualOdometry:
    """
    pySLAM Visual Odometry integration for OrbyGlasses.
    Provides real-time motion tracking and trajectory estimation.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Camera parameters
        self.width = config.get('camera.width', 640)
        self.height = config.get('camera.height', 480)
        self.fx = config.get('camera.fx', 500.0)
        self.fy = config.get('camera.fy', 500.0)
        self.cx = config.get('camera.cx', self.width / 2.0)
        self.cy = config.get('camera.cy', self.height / 2.0)
        
        # VO parameters
        self.feature_type = config.get('vo.feature_type', 'ORB')
        self.num_features = config.get('vo.num_features', 2000)
        self.use_rerun = config.get('vo.use_rerun', True)
        
        # State
        self.is_initialized = False
        self.is_running = False
        self.current_pose = np.eye(4)
        self.trajectory = deque(maxlen=1000)
        self.frame_count = 0
        
        # Threading
        self.vo_thread = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        
        # Visualization
        self.traj_img_size = 400
        self.traj_img = np.zeros((self.traj_img_size, self.traj_img_size, 3), dtype=np.uint8)
        self.half_traj_img_size = self.traj_img_size // 2
        self.draw_scale = 50  # Scale for trajectory visualization
        
        if PYSLAM_VO_AVAILABLE:
            try:
                self._initialize_pyslam_vo()
            except Exception as e:
                import traceback
                self.logger.error(f"Failed to initialize pySLAM VO: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                self.is_initialized = False
        else:
            self.logger.warning("pySLAM VO not available, using fallback")
            self._initialize_fallback()
    
    def _initialize_pyslam_vo(self):
        """Initialize pySLAM Visual Odometry."""
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
            
            # Create feature tracker
            if self.feature_type == 'ORB':
                feature_tracker_config = FeatureTrackerConfigs.ORB2
            elif self.feature_type == 'SIFT':
                feature_tracker_config = FeatureTrackerConfigs.SIFT
            else:
                feature_tracker_config = FeatureTrackerConfigs.ORB2

            # Extract params from config and override num_features
            tracker_params = feature_tracker_config.copy()
            tracker_params['num_features'] = self.num_features

            # Create feature tracker with unpacked params
            self.feature_tracker = feature_tracker_factory(**tracker_params)

            # Create mock ground truth for live camera (VO expects this even if unused)
            class MockGroundtruth:
                """Mock groundtruth object for live camera (no ground truth available)"""
                def __init__(self):
                    self.type = "none"  # Not KITTI, not EUROC, just live camera
                    self.data = []
                    self.scale_factor = 1.0

                def getTimestamp(self, idx):
                    return None

                def getPose(self, idx):
                    return None

            self.groundtruth = MockGroundtruth()

            # Create Visual Odometry
            self.vo = VisualOdometryEducational(self.camera, self.groundtruth, self.feature_tracker)
            
            # Initialize Rerun if enabled
            if self.use_rerun:
                try:
                    Rerun.init_vo()
                    self.logger.info("✅ Rerun.io initialized for VO visualization")
                except Exception as e:
                    self.logger.warning(f"Rerun initialization failed: {e}")
                    self.use_rerun = False
            
            self.is_initialized = True
            self.logger.info("✅ pySLAM Visual Odometry initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pySLAM VO: {e}")
            raise
    
    def _initialize_fallback(self):
        """Initialize fallback OpenCV-based motion tracking."""
        self.logger.info("Initializing fallback motion tracking...")
        self.is_initialized = True
    
    def start(self):
        """Start Visual Odometry processing."""
        if not self.is_initialized:
            self.logger.error("VO not initialized")
            return False
        
        if self.is_running:
            self.logger.warning("VO already running")
            return True
        
        self.is_running = True
        self.vo_thread = threading.Thread(target=self._vo_worker, daemon=True)
        self.vo_thread.start()
        self.logger.info("✅ Visual Odometry started")
        return True
    
    def stop(self):
        """Stop Visual Odometry processing."""
        self.is_running = False
        if self.vo_thread:
            self.vo_thread.join(timeout=1.0)
        self.logger.info("Visual Odometry stopped")
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """Process a frame through Visual Odometry."""
        if not self.is_initialized or not self.is_running:
            return None
        
        try:
            # Add frame to queue
            if not self.frame_queue.full():
                self.frame_queue.put(frame.copy())
            
            # Get result if available
            if not self.result_queue.empty():
                return self.result_queue.get_nowait()
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
        
        return None
    
    def _vo_worker(self):
        """Worker thread for Visual Odometry processing."""
        while self.is_running:
            try:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get_nowait()
                    
                    if PYSLAM_VO_AVAILABLE and hasattr(self, 'vo'):
                        result = self._process_pyslam_frame(frame)
                    else:
                        result = self._process_fallback_frame(frame)
                    
                    # Add result to queue
                    if not self.result_queue.full():
                        self.result_queue.put(result)
                
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                self.logger.error(f"VO worker error: {e}")
                time.sleep(0.1)
    
    def _process_pyslam_frame(self, frame: np.ndarray) -> Dict:
        """Process frame using pySLAM Visual Odometry."""
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame

            # Process through VO
            timestamp = time.time()
            self.vo.track(gray_frame, None, None, self.frame_count, timestamp)

            # Get pose (with extra safety checks)
            try:
                if hasattr(self.vo, 'poses') and self.vo.poses is not None and len(self.vo.poses) > 0:
                    pose = self.vo.poses[-1]
                    if pose is not None:
                        self.current_pose = pose
            except Exception as e:
                self.logger.debug(f"Could not get pose: {e}")

            # Get trajectory (with full error protection)
            try:
                if hasattr(self.vo, 'traj3d_est') and self.vo.traj3d_est is not None:
                    if len(self.vo.traj3d_est) > 0:
                        traj_point = self.vo.traj3d_est[-1]
                        if traj_point is not None:
                            # Handle both tuple/list and numpy array
                            if hasattr(traj_point, '__len__') and len(traj_point) >= 3:
                                x = float(traj_point[0]) if traj_point[0] is not None else 0.0
                                y = float(traj_point[1]) if traj_point[1] is not None else 0.0
                                z = float(traj_point[2]) if traj_point[2] is not None else 0.0
                                self.trajectory.append([x, y, z])
            except Exception as e:
                self.logger.debug(f"Could not get trajectory: {e}")

            # Update visualization
            try:
                self._update_trajectory_visualization()
            except Exception as e:
                self.logger.debug(f"Trajectory viz error: {e}")

            # Show pySLAM VO's own visualization window (like main_vo.py)
            try:
                if hasattr(self.vo, 'draw_img') and self.vo.draw_img is not None:
                    cv2.imshow("pySLAM VO - Camera", self.vo.draw_img)
                    cv2.waitKey(1)
            except Exception as e:
                self.logger.debug(f"VO camera window error: {e}")

            # Show trajectory window (like main_vo.py)
            try:
                if self.traj_img is not None and self.traj_img.size > 0:
                    cv2.imshow("pySLAM VO - Trajectory", self.traj_img)
                    cv2.waitKey(1)
            except Exception as e:
                self.logger.debug(f"Trajectory window error: {e}")

            # Rerun logging (like main_vo.py)
            if self.use_rerun and hasattr(self, 'vo'):
                try:
                    img_to_log = self.vo.draw_img if hasattr(self.vo, 'draw_img') else None
                    Rerun.log_3d_camera_img_seq(self.frame_count, img_to_log, None, self.camera, self.current_pose)
                    if hasattr(self.vo, 'traj3d_est') and self.vo.traj3d_est is not None and len(self.vo.traj3d_est) > 0:
                        Rerun.log_3d_trajectory(self.frame_count, self.vo.traj3d_est, "estimated", color=[0, 0, 255])
                except Exception as e:
                    self.logger.debug(f"Rerun logging error: {e}")

            self.frame_count += 1

            return {
                'pose': self.current_pose.copy(),
                'position': self.current_pose[:3, 3].copy(),
                'trajectory': list(self.trajectory),
                'frame_count': self.frame_count,
                'message': f"pySLAM VO frame {self.frame_count}"
            }

        except Exception as e:
            import traceback
            self.logger.error(f"pySLAM VO processing error: {e}")
            self.logger.error(f"Full traceback:\n{traceback.format_exc()}")
            # Return safe default result
            return {
                'pose': self.current_pose.copy(),
                'position': self.current_pose[:3, 3].copy(),
                'trajectory': list(self.trajectory),
                'frame_count': self.frame_count,
                'message': f"VO error: {e}"
            }
    
    def _process_fallback_frame(self, frame: np.ndarray) -> Dict:
        """Process frame using fallback motion tracking."""
        # Simple fallback - just return basic info
        return {
            'pose': self.current_pose.copy(),
            'position': self.current_pose[:3, 3].copy(),
            'trajectory': list(self.trajectory),
            'frame_count': self.frame_count,
            'message': f"Fallback VO frame {self.frame_count}"
        }
    
    def _update_trajectory_visualization(self):
        """Update trajectory visualization."""
        if len(self.trajectory) > 1:
            # Clear previous trajectory
            self.traj_img = np.zeros((self.traj_img_size, self.traj_img_size, 3), dtype=np.uint8)
            
            # Draw trajectory
            for i in range(1, len(self.trajectory)):
                prev_pos = self.trajectory[i-1]
                curr_pos = self.trajectory[i]
                
                # Convert to image coordinates
                x1 = int(self.draw_scale * prev_pos[0]) + self.half_traj_img_size
                y1 = self.half_traj_img_size - int(self.draw_scale * prev_pos[2])
                x2 = int(self.draw_scale * curr_pos[0]) + self.half_traj_img_size
                y2 = self.half_traj_img_size - int(self.draw_scale * curr_pos[2])
                
                # Ensure coordinates are within bounds
                x1 = max(0, min(self.traj_img_size-1, x1))
                y1 = max(0, min(self.traj_img_size-1, y1))
                x2 = max(0, min(self.traj_img_size-1, x2))
                y2 = max(0, min(self.traj_img_size-1, y2))
                
                # Draw line
                cv2.line(self.traj_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw current position
            if len(self.trajectory) > 0:
                curr_pos = self.trajectory[-1]
                x = int(self.draw_scale * curr_pos[0]) + self.half_traj_img_size
                y = self.half_traj_img_size - int(self.draw_scale * curr_pos[2])
                x = max(0, min(self.traj_img_size-1, x))
                y = max(0, min(self.traj_img_size-1, y))
                cv2.circle(self.traj_img, (x, y), 3, (0, 0, 255), -1)
    
    def get_trajectory_image(self) -> np.ndarray:
        """Get trajectory visualization image."""
        return self.traj_img.copy()
    
    def get_current_pose(self) -> np.ndarray:
        """Get current camera pose."""
        return self.current_pose.copy()
    
    def get_trajectory(self) -> List[List[float]]:
        """Get trajectory points."""
        return list(self.trajectory)
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        if hasattr(self, 'vo') and hasattr(self.vo, 'cleanup'):
            self.vo.cleanup()
        self.logger.info("Visual Odometry cleanup completed")
