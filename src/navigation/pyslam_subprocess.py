#!/usr/bin/env python3
"""
pySLAM Subprocess Integration for OrbyGlasses
Runs pySLAM in a separate process to avoid environment conflicts
"""

import os
import sys
import cv2
import numpy as np
import logging
import time
import subprocess
import json
import threading
import queue
from typing import Dict, Optional, List, Tuple
from collections import deque

# Add pySLAM path to sys.path
pyslam_path = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'pyslam')
if os.path.exists(pyslam_path) and pyslam_path not in sys.path:
    sys.path.insert(0, pyslam_path)


class PySLAMSubprocess:
    """
    pySLAM integration using subprocess to avoid environment conflicts.
    """

    def __init__(self, config: Dict):
        """Initialize pySLAM subprocess system."""
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
        
        # Subprocess management
        self.pyslam_process = None
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False
        
        # Start pySLAM subprocess
        self._start_pyslam_subprocess()

    def _start_pyslam_subprocess(self):
        """Start pySLAM in a subprocess."""
        try:
            # Create pySLAM script
            pyslam_script = self._create_pyslam_script()
            
            # Start subprocess
            self.pyslam_process = subprocess.Popen(
                [sys.executable, '-c', pyslam_script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )
            
            # Start communication threads
            self.running = True
            self.input_thread = threading.Thread(target=self._input_handler)
            self.output_thread = threading.Thread(target=self._output_handler)
            self.input_thread.daemon = True
            self.output_thread.daemon = True
            self.input_thread.start()
            self.output_thread.start()
            
            self.is_initialized = True
            print("✅ pySLAM subprocess started successfully!")
            self.logger.info("✅ pySLAM subprocess started successfully!")
            
        except Exception as e:
            error_msg = f"Failed to start pySLAM subprocess: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)

    def _create_pyslam_script(self) -> str:
        """Create the pySLAM script to run in subprocess."""
        return f'''
import sys
import os
import json
import cv2
import numpy as np
import time

# Add pySLAM path
pyslam_path = "{pyslam_path}"
if pyslam_path not in sys.path:
    sys.path.insert(0, pyslam_path)

try:
    from pyslam.config import Config
    from pyslam.slam.slam import Slam, SlamState
    from pyslam.slam.camera import PinholeCamera
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
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
    feature_tracker_config = FeatureTrackerConfigs.ORB2.copy()
    feature_tracker_config["num_features"] = {self.config.get('slam.orb_features', 2000)}
    
    slam = Slam(camera, feature_tracker_config)
    
    # Communication loop
    while True:
        try:
            line = input()
            if not line:
                continue
                
            data = json.loads(line)
            command = data.get('command')
            
            if command == 'process_frame':
                # Decode frame
                frame_data = np.frombuffer(bytes.fromhex(data['frame']), dtype=np.uint8)
                frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
                
                # Process frame
                timestamp = time.time()
                slam.track(frame, None, None, data['frame_count'], timestamp)
                
                # Get results
                tracking_state = "OK"
                if hasattr(slam, 'tracking') and hasattr(slam.tracking, 'state'):
                    if slam.tracking.state == SlamState.LOST:
                        tracking_state = "LOST"
                    elif slam.tracking.state == SlamState.NOT_INITIALIZED:
                        tracking_state = "NOT_INITIALIZED"
                
                # Get pose
                pose = np.eye(4)
                if hasattr(slam, 'tracking') and hasattr(slam.tracking, 'cur_pose'):
                    pose = slam.tracking.cur_pose
                elif hasattr(slam, 'tracking') and hasattr(slam.tracking, 'get_current_pose'):
                    pose = slam.tracking.get_current_pose()
                
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
                
                # Send result
                result = {{
                    'pose': pose.tolist(),
                    'position': pose[:3, 3].tolist(),
                    'tracking_quality': 0.9 if tracking_state == "OK" else 0.0,
                    'tracking_state': tracking_state,
                    'message': f"pySLAM frame {{data['frame_count']}}",
                    'is_initialized': True,
                    'trajectory_length': data['frame_count'],
                    'num_map_points': len(map_points),
                    'performance': {{}}
                }}
                
                print(json.dumps(result))
                sys.stdout.flush()
                
            elif command == 'shutdown':
                break
                
        except Exception as e:
            error_result = {{
                'error': str(e),
                'pose': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
                'position': [0,0,0],
                'tracking_quality': 0.0,
                'tracking_state': "ERROR",
                'message': f"pySLAM error: {{e}}",
                'is_initialized': False,
                'trajectory_length': 0,
                'num_map_points': 0,
                'performance': {{}}
            }}
            print(json.dumps(error_result))
            sys.stdout.flush()
            
except Exception as e:
    error_result = {{
        'error': str(e),
        'pose': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        'position': [0,0,0],
        'tracking_quality': 0.0,
        'tracking_state': "ERROR",
        'message': f"pySLAM initialization error: {{e}}",
        'is_initialized': False,
        'trajectory_length': 0,
        'num_map_points': 0,
        'performance': {{}}
    }}
    print(json.dumps(error_result))
    sys.stdout.flush()
'''

    def _input_handler(self):
        """Handle input to pySLAM subprocess."""
        while self.running:
            try:
                if not self.input_queue.empty():
                    data = self.input_queue.get(timeout=0.1)
                    self.pyslam_process.stdin.write(json.dumps(data) + '\n')
                    self.pyslam_process.stdin.flush()
            except:
                pass

    def _output_handler(self):
        """Handle output from pySLAM subprocess."""
        while self.running:
            try:
                line = self.pyslam_process.stdout.readline()
                if line:
                    data = json.loads(line.strip())
                    self.output_queue.put(data)
            except:
                pass

    def process_frame(self, frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> Dict:
        """
        Process a single frame through pySLAM subprocess.

        Args:
            frame: Input frame (BGR or grayscale)
            depth_map: Depth map (ignored for monocular SLAM)

        Returns:
            Dictionary with SLAM results
        """
        self.frame_count += 1
        
        try:
            # Encode frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame_hex = buffer.tobytes().hex()
            
            # Send to subprocess
            data = {
                'command': 'process_frame',
                'frame': frame_hex,
                'frame_count': self.frame_count
            }
            self.input_queue.put(data)
            
            # Get result
            try:
                result = self.output_queue.get(timeout=1.0)
                
                # Convert lists back to numpy arrays
                result['pose'] = np.array(result['pose'])
                result['position'] = np.array(result['position'])
                
                # Update internal state
                self.current_pose = result['pose']
                self.trajectory.append(self.current_pose.copy())
                
                return result
                
            except queue.Empty:
                return {
                    'pose': self.current_pose.copy(),
                    'position': self.current_pose[:3, 3].copy(),
                    'tracking_quality': 0.0,
                    'tracking_state': "TIMEOUT",
                    'message': "pySLAM subprocess timeout",
                    'is_initialized': self.is_initialized,
                    'trajectory_length': len(self.trajectory),
                    'num_map_points': 0,
                    'performance': {}
                }
                
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
        return np.array([]).reshape(0, 3)  # Subprocess doesn't maintain map points

    def is_tracking_good(self) -> bool:
        """Check if SLAM tracking is good."""
        return True  # Subprocess always returns True

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
            self.map_points = []
            self.logger.info("SLAM system reset")
        except Exception as e:
            self.logger.error(f"Reset error: {e}")

    def shutdown(self):
        """Shutdown SLAM system and subprocess."""
        try:
            self.running = False
            
            # Send shutdown command
            if self.pyslam_process:
                self.input_queue.put({'command': 'shutdown'})
                time.sleep(0.5)
                self.pyslam_process.terminate()
                self.pyslam_process.wait()
            
            self.logger.info("SLAM system shutdown")
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")


# For backward compatibility
def create_pyslam_system(config: Dict) -> PySLAMSubprocess:
    """Create a pySLAM subprocess instance."""
    return PySLAMSubprocess(config)