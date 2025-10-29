#!/usr/bin/env python3
"""
pySLAM Subprocess Runner - Runs pySLAM in a separate process to avoid OpenCV conflicts
"""

import os
import sys
import json
import time
import logging
import numpy as np
import cv2
import subprocess
import tempfile
from typing import Dict, Optional

class PySLAMSubprocess:
    """pySLAM subprocess runner that avoids OpenCV conflicts"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Camera parameters
        self.width = config.get('camera.width', 640)
        self.height = config.get('camera.height', 480)
        self.fx = config.get('mapping3d.fx', 500)
        self.fy = config.get('mapping3d.fy', 500)
        self.cx = self.width / 2
        self.cy = self.height / 2

        # State
        self.frame_count = 0
        self.is_initialized = False
        self.current_pose = np.eye(4)
        self.trajectory = []
        self.performance_stats = {}
        
        # Create temporary directory for communication
        self.temp_dir = tempfile.mkdtemp(prefix='pyslam_')
        self.frame_file = os.path.join(self.temp_dir, 'frame.png')
        self.result_file = os.path.join(self.temp_dir, 'result.json')
        
        # Initialize pySLAM subprocess
        self._initialize_subprocess()

    def _initialize_subprocess(self):
        """Initialize pySLAM subprocess"""
        try:
            # Create the subprocess script
            script_content = f'''
import sys
import os
import json
import time
import numpy as np
import cv2

# Add pySLAM path
pyslam_path = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'pyslam')
if os.path.exists(pyslam_path) and pyslam_path not in sys.path:
    sys.path.insert(0, pyslam_path)

# Add pySLAM virtual environment site-packages
pyslam_venv_site_packages = os.path.expanduser('~/.python/venvs/pyslam/lib/python3.11/site-packages')
if os.path.exists(pyslam_venv_site_packages) and pyslam_venv_site_packages not in sys.path:
    sys.path.insert(0, pyslam_venv_site_packages)

try:
    from pyslam.config import Config
    from pyslam.slam.slam import Slam, SlamState
    from pyslam.slam.camera import PinholeCamera
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs, FeatureTrackerTypes
    from pyslam.local_features.feature_types import FeatureDetectorTypes
    
    # Initialize pySLAM
    camera_config = Config()
    camera_config.cam_settings = {{
        'Camera.width': {self.width},
        'Camera.height': {self.height},
        'Camera.fx': {self.fx},
        'Camera.fy': {self.fy},
        'Camera.cx': {self.cx},
        'Camera.cy': {self.cy},
        'Camera.fps': 30,
        'Camera.k1': 0.0,
        'Camera.k2': 0.0,
        'Camera.p1': 0.0,
        'Camera.p2': 0.0,
        'Camera.k3': 0.0
    }}
    
    camera = PinholeCamera(camera_config)
    
    # Feature detector configuration
    feature_type = '{self.config.get('slam.feature_type', 'ORB')}'
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
    slam_config.num_features = {self.config.get('slam.orb_features', 2000)}
    slam_config.enable_loop_closing = {self.config.get('slam.loop_closure', False)}
    slam_config.enable_local_mapping = True

    # Create feature tracker config
    feature_tracker_config = FeatureTrackerConfigs.ORB2.copy()
    feature_tracker_config["num_features"] = {self.config.get('slam.orb_features', 2000)}
    
    # Initialize SLAM
    slam = Slam(camera, feature_tracker_config)
    
    frame_count = 0
    current_pose = np.eye(4)
    trajectory = []
    
    print("pySLAM subprocess initialized", flush=True)
    
    while True:
        try:
            # Read frame file
            if os.path.exists('{self.frame_file}'):
                frame = cv2.imread('{self.frame_file}')
                if frame is not None:
                    # Process frame
                    timestamp = time.time()
                    slam.track(frame, None, None, frame_count, timestamp)
                    
                    # Get tracking state
                    tracking_state = "OK"
                    if hasattr(slam, 'tracking') and hasattr(slam.tracking, 'state'):
                        if slam.tracking.state == SlamState.LOST:
                            tracking_state = "LOST"
                        elif slam.tracking.state == SlamState.NOT_INITIALIZED:
                            tracking_state = "NOT_INITIALIZED"
                    
                    # Get current pose
                    if hasattr(slam, 'tracking') and hasattr(slam.tracking, 'cur_pose'):
                        current_pose = slam.tracking.cur_pose
                    elif hasattr(slam, 'tracking') and hasattr(slam.tracking, 'get_current_pose'):
                        current_pose = slam.tracking.get_current_pose()
                    
                    # Add to trajectory
                    trajectory.append(current_pose.copy())
                    
                    # Get map points
                    map_points = []
                    if hasattr(slam, 'map') and slam.map is not None:
                        try:
                            points = slam.map.get_points()
                            if points is not None and len(points) > 0:
                                for point in points:
                                    if hasattr(point, 'get_pos'):
                                        map_points.append(point.get_pos().tolist())
                                    elif hasattr(point, 'pos'):
                                        map_points.append(point.pos.tolist())
                        except:
                            pass
                    
                    # Create result
                    result = {{
                        'pose': current_pose.tolist(),
                        'position': current_pose[:3, 3].tolist(),
                        'tracking_quality': 0.9 if tracking_state == "OK" else 0.0,
                        'tracking_state': tracking_state,
                        'message': f"pySLAM frame {{frame_count}}",
                        'is_initialized': True,
                        'trajectory_length': len(trajectory),
                        'num_map_points': len(map_points),
                        'performance': {{}}
                    }}
                    
                    # Write result
                    with open('{self.result_file}', 'w') as f:
                        json.dump(result, f)
                    
                    frame_count += 1
                    
                    # Remove frame file
                    os.remove('{self.frame_file}')
            
            time.sleep(0.01)  # Small delay
            
        except Exception as e:
            print(f"Error in pySLAM subprocess: {{e}}", flush=True)
            time.sleep(0.1)
            
except ImportError as e:
    print(f"pySLAM not available: {{e}}", flush=True)
    # Fallback implementation
    import cv2
    import numpy as np
    
    orb = cv2.ORB_create(nfeatures={self.config.get('slam.orb_features', 2000)})
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    camera_matrix = np.array([
        [{self.fx}, 0, {self.cx}],
        [0, {self.fy}, {self.cy}],
        [0, 0, 1]
    ], dtype=np.float32)
    
    prev_frame = None
    prev_keypoints = None
    prev_descriptors = None
    frame_count = 0
    current_pose = np.eye(4)
    trajectory = []
    
    print("pySLAM fallback subprocess initialized", flush=True)
    
    while True:
        try:
            if os.path.exists('{self.frame_file}'):
                frame = cv2.imread('{self.frame_file}')
                if frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    keypoints, descriptors = orb.detectAndCompute(gray, None)
                    
                    if prev_frame is not None and prev_keypoints is not None and prev_descriptors is not None:
                        matches = matcher.match(prev_descriptors, descriptors)
                        matches = sorted(matches, key=lambda x: x.distance)
                        good_matches = matches[:50]
                        
                        if len(good_matches) > 10:
                            src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                            
                            E, mask = cv2.findEssentialMat(src_pts, dst_pts, camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                            
                            if E is not None:
                                _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, camera_matrix)
                                pose_delta = np.eye(4)
                                pose_delta[:3, :3] = R
                                pose_delta[:3, 3] = t.flatten()
                                current_pose = current_pose @ pose_delta
                                trajectory.append(current_pose.copy())
                    
                    prev_frame = gray.copy()
                    prev_keypoints = keypoints
                    prev_descriptors = descriptors
                    
                    result = {{
                        'pose': current_pose.tolist(),
                        'position': current_pose[:3, 3].tolist(),
                        'tracking_quality': 0.8 if len(keypoints) > 50 else 0.3,
                        'tracking_state': "OK" if len(keypoints) > 50 else "LOST",
                        'message': f"Fallback SLAM frame {{frame_count}}",
                        'is_initialized': True,
                        'trajectory_length': len(trajectory),
                        'num_map_points': 0,
                        'performance': {{}}
                    }}
                    
                    with open('{self.result_file}', 'w') as f:
                        json.dump(result, f)
                    
                    frame_count += 1
                    os.remove('{self.frame_file}')
            
            time.sleep(0.01)
            
        except Exception as e:
            print(f"Error in fallback subprocess: {{e}}", flush=True)
            time.sleep(0.1)
'''
            
            # Write the script
            script_file = os.path.join(self.temp_dir, 'pyslam_worker.py')
            with open(script_file, 'w') as f:
                f.write(script_content)
            
            # Start the subprocess
            self.process = subprocess.Popen(
                [sys.executable, script_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for initialization
            time.sleep(2)
            
            self.logger.info("✅ pySLAM subprocess initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pySLAM subprocess: {e}")
            raise RuntimeError(f"Failed to initialize pySLAM subprocess: {e}")

    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame"""
        try:
            # Save frame to file
            cv2.imwrite(self.frame_file, frame)
            
            # Wait for result
            max_wait = 1.0  # 1 second timeout
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                if os.path.exists(self.result_file):
                    # Read result
                    with open(self.result_file, 'r') as f:
                        result = json.load(f)
                    
                    # Convert pose back to numpy array
                    result['pose'] = np.array(result['pose'])
                    result['position'] = np.array(result['position'])
                    
                    # Update local state
                    self.current_pose = result['pose']
                    self.trajectory.append(self.current_pose.copy())
                    self.is_initialized = result['is_initialized']
                    self.frame_count += 1
                    
                    return result
                
                time.sleep(0.01)
            
            # Timeout - return error
            return {
                'pose': self.current_pose.copy(),
                'position': self.current_pose[:3, 3].copy(),
                'tracking_quality': 0.0,
                'tracking_state': "ERROR",
                'message': "pySLAM subprocess timeout",
                'is_initialized': False,
                'trajectory_length': len(self.trajectory),
                'num_map_points': 0,
                'performance': self.performance_stats.copy()
            }
            
        except Exception as e:
            self.logger.error(f"pySLAM subprocess error: {e}")
            return {
                'pose': self.current_pose.copy(),
                'position': self.current_pose[:3, 3].copy(),
                'tracking_quality': 0.0,
                'tracking_state': "ERROR",
                'message': f"pySLAM subprocess error: {e}",
                'is_initialized': False,
                'trajectory_length': len(self.trajectory),
                'num_map_points': 0,
                'performance': self.performance_stats.copy()
            }

    def get_map_points(self) -> np.ndarray:
        """Get map points"""
        return np.array([]).reshape(0, 3)

    def reset(self):
        """Reset SLAM system"""
        self.frame_count = 0
        self.is_initialized = False
        self.current_pose = np.eye(4)
        self.trajectory = []
        self.performance_stats = {}

    def shutdown(self):
        """Shutdown SLAM system"""
        try:
            if hasattr(self, 'process') and self.process:
                self.process.terminate()
                self.process.wait(timeout=5)
            
            # Clean up temp directory
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
            self.logger.info("pySLAM subprocess shutdown")
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")
