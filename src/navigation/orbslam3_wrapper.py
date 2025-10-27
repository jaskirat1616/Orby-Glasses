"""
ORB-SLAM3 Integration for OrbyGlasses
Industry-standard monocular SLAM with loop closure and relocalization.

ORB-SLAM3 is the most accurate open-source SLAM system (CVPR 2020).
- 30-60 FPS real-time performance
- 2-5x more accurate than ORB-SLAM2
- Loop closure prevents drift
- Relocalization for lost tracking recovery
"""

import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from collections import deque
import time

try:
    import orb_slam3
    ORBSLAM3_AVAILABLE = True
except ImportError:
    ORBSLAM3_AVAILABLE = False
    logging.warning("ORB-SLAM3 not installed. Run: pip install python-orb-slam3")


class ORBSLAM3System:
    """
    Wrapper for ORB-SLAM3 monocular SLAM.
    Provides the most accurate monocular SLAM available.
    """

    def __init__(self, config):
        """
        Initialize ORB-SLAM3 system.

        Args:
            config: ConfigManager instance
        """
        if not ORBSLAM3_AVAILABLE:
            raise ImportError("ORB-SLAM3 not installed. Run: ./install_orbslam3.sh")

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Camera parameters
        width = config.get('camera.width', 320)
        height = config.get('camera.height', 240)
        self.fx = config.get('mapping3d.fx', 500)
        self.fy = config.get('mapping3d.fy', 500)
        self.cx = width / 2
        self.cy = height / 2

        # Camera matrix
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Initialize ORB-SLAM3
        # Mode: MONOCULAR (0), STEREO (1), RGBD (2)
        self.slam_mode = 0  # Monocular

        # Create vocabulary and settings files if needed
        self._create_config_files()

        # Initialize ORB-SLAM3
        try:
            if not hasattr(orb_slam3, "System"):
                raise ImportError("The installed orb_slam3 package does NOT provide a System class. You must build the python-orbslam3 bindings from source -- see README.md or https://github.com/uoip/python-orbslam3. Do NOT install from pip. Uninstall with 'pip uninstall python-orb-slam3'.")
            self.slam = orb_slam3.System(
                vocab_file=self.vocab_path,
                settings_file=self.settings_path,
                sensor_type=orb_slam3.Sensor.MONOCULAR
            )
            self.slam.set_use_viewer(False)  # We'll use our own visualization
            self.slam.activate_localization_mode(False)  # Full SLAM mode

            self.logger.info("✅ ORB-SLAM3 initialized successfully")
            self.logger.info(f"Camera: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
        except Exception as e:
            error_msg = (
                f"Failed to initialize ORB-SLAM3: {e}\n"
                f"\n>>> If the error message references missing 'System' class in orb_slam3,\n"
                f"   you MUST build the correct python-orbslam3 from scratch: https://github.com/uoip/python-orbslam3\n"
                f"   (Do NOT install 'python-orb-slam3' from pip!)\n\n"
                f"   See README.md for step-by-step Mac instructions."
            )
            self.logger.error(error_msg)
            raise ImportError(error_msg)

        # State
        self.is_initialized = False
        self.frame_count = 0
        self.current_pose = np.eye(4, dtype=np.float32)
        self.tracking_state = "NOT_INITIALIZED"

        # History
        self.pose_history = deque(maxlen=1000)
        self.position_history = deque(maxlen=1000)

        # Tracking states mapping
        self.STATE_NAMES = {
            -1: "SYSTEM_NOT_READY",
            0: "NO_IMAGES_YET",
            1: "NOT_INITIALIZED",
            2: "OK",
            3: "RECENTLY_LOST",
            4: "LOST",
            5: "OK_KLT"
        }

    def _create_config_files(self):
        """Create necessary config files for ORB-SLAM3."""
        import os
        import yaml

        config_dir = "data/orbslam3"
        os.makedirs(config_dir, exist_ok=True)

        self.vocab_path = os.path.join(config_dir, "ORBvoc.txt")
        self.settings_path = os.path.join(config_dir, "settings.yaml")

        # Download vocabulary if needed
        if not os.path.exists(self.vocab_path):
            self.logger.info("Downloading ORB vocabulary file...")
            import urllib.request
            vocab_url = "https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz"
            try:
                urllib.request.urlretrieve(vocab_url, self.vocab_path + ".tar.gz")
                import tarfile
                with tarfile.open(self.vocab_path + ".tar.gz", "r:gz") as tar:
                    tar.extractall(config_dir)
                self.logger.info("✅ Vocabulary downloaded")
            except Exception as e:
                self.logger.warning(f"Could not download vocabulary: {e}")
                self.logger.info("Using bundled vocabulary...")

        # Create settings file
        if not os.path.exists(self.settings_path):
            settings = {
                "File.version": "1.0",

                # Camera Parameters
                "Camera.type": "PinHole",
                "Camera.fx": float(self.fx),
                "Camera.fy": float(self.fy),
                "Camera.cx": float(self.cx),
                "Camera.cy": float(self.cy),

                # Camera distortion (assuming no distortion)
                "Camera.k1": 0.0,
                "Camera.k2": 0.0,
                "Camera.p1": 0.0,
                "Camera.p2": 0.0,
                "Camera.k3": 0.0,

                # Camera frames per second
                "Camera.fps": float(self.config.get('camera.fps', 30)),

                # Color order (0: BGR, 1: RGB)
                "Camera.RGB": 0,

                # ORB Parameters
                "ORBextractor.nFeatures": self.config.get('slam.orb_features', 1500),
                "ORBextractor.scaleFactor": 1.2,
                "ORBextractor.nLevels": 8,
                "ORBextractor.iniThFAST": self.config.get('slam.fast_threshold', 12),
                "ORBextractor.minThFAST": 7,

                # Viewer Parameters
                "Viewer.KeyFrameSize": 0.05,
                "Viewer.KeyFrameLineWidth": 1.0,
                "Viewer.GraphLineWidth": 0.9,
                "Viewer.PointSize": 2.0,
                "Viewer.CameraSize": 0.08,
                "Viewer.CameraLineWidth": 3.0,
                "Viewer.ViewpointX": 0.0,
                "Viewer.ViewpointY": -0.7,
                "Viewer.ViewpointZ": -1.8,
                "Viewer.ViewpointF": 500.0,
            }

            with open(self.settings_path, 'w') as f:
                yaml.dump(settings, f)

            self.logger.info(f"✅ Settings created: {self.settings_path}")

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> Dict:
        """
        Process frame through ORB-SLAM3.

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
            timestamp = time.time()

        # Process frame in ORB-SLAM3
        try:
            pose = self.slam.process_image_mono(gray, timestamp)
            state = self.slam.get_tracking_state()

            self.tracking_state = self.STATE_NAMES.get(state, "UNKNOWN")

            # Check if tracking is successful
            if state == 2:  # OK
                if pose is not None and pose.size > 0:
                    # ORB-SLAM3 returns 4x4 transformation matrix (Tcw: camera to world)
                    # Convert to world to camera for consistency
                    if pose.shape == (4, 4):
                        # Invert to get world frame
                        self.current_pose = np.linalg.inv(pose).astype(np.float32)
                    else:
                        self.current_pose = pose.astype(np.float32)

                    self.is_initialized = True

                    # Extract position
                    position = self.current_pose[:3, 3].copy()

                    # Update history
                    self.pose_history.append(self.current_pose.copy())
                    self.position_history.append(position.tolist())

                    # Get map points for quality estimation
                    num_map_points = self.slam.get_num_map_points()
                    num_tracked_points = self.slam.get_num_tracked_points()

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

            # Tracking lost or not initialized
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
            self.logger.error(f"ORB-SLAM3 processing error: {e}")
            return self._get_default_result()

    def get_map_points(self) -> np.ndarray:
        """
        Get all map points for visualization.

        Returns:
            Nx3 array of map point positions
        """
        try:
            points = self.slam.get_all_map_points()
            if points is not None and len(points) > 0:
                return np.array(points)
        except:
            pass
        return np.array([]).reshape(0, 3)

    def get_keyframes(self) -> list:
        """
        Get all keyframe poses.

        Returns:
            List of 4x4 pose matrices
        """
        try:
            keyframes = self.slam.get_all_keyframes()
            return keyframes if keyframes else []
        except:
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
            self.logger.info("ORB-SLAM3 reset")
        except Exception as e:
            self.logger.error(f"Failed to reset: {e}")

    def shutdown(self):
        """Shutdown SLAM system."""
        try:
            self.slam.shutdown()
            self.logger.info("ORB-SLAM3 shutdown")
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
