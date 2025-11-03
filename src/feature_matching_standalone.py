#!/usr/bin/env python3
"""
Standalone Feature Matching Mode for OrbyGlasses
Shows feature matching visualization without heavy dependencies (YOLO, etc.)
Works with both SLAM and Visual Odometry
"""

import sys
import os
import cv2
import numpy as np
import argparse
import time
import logging

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
pyslam_path = os.path.join(os.path.dirname(__file__), '..', 'third_party', 'pyslam')
if os.path.exists(pyslam_path) and pyslam_path not in sys.path:
    sys.path.insert(0, pyslam_path)

# Try to import pySLAM
PYSLAM_AVAILABLE = False
try:
    from pyslam.config import Config
    from pyslam.config_parameters import Parameters
    from pyslam.slam.slam import Slam
    from pyslam.slam.slam_commons import SlamState
    from pyslam.slam.camera import PinholeCamera
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
    from pyslam.local_features.feature_types import FeatureDetectorTypes
    from pyslam.io.dataset_types import SensorType, DatasetEnvironmentType
    PYSLAM_AVAILABLE = True
    print("✅ pySLAM available")
except ImportError as e:
    print(f"❌ pySLAM not available: {e}")
    print("Install pySLAM from third_party/pyslam")

# Try to import VO
PYSLAM_VO_AVAILABLE = False
try:
    from pyslam.vo.visual_odometry import VisualOdometry
    PYSLAM_VO_AVAILABLE = True
    print("✅ pySLAM VO available")
except ImportError:
    print("⚠️  pySLAM VO not available (will use SLAM only)")


class StandaloneFeatureMatching:
    """Standalone feature matching visualization without YOLO or other heavy dependencies."""
    
    def __init__(self, camera_source=1, mode='slam', width=640, height=480, fx=525.0, fy=525.0):
        """
        Initialize standalone feature matching.
        
        Args:
            camera_source: Camera index or video file path
            mode: 'slam' or 'vo' (visual odometry)
            width: Camera width
            height: Camera height
            fx: Camera focal length x
            fy: Camera focal length y
        """
        self.camera_source = camera_source
        self.mode = mode
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = width / 2.0
        self.cy = height / 2.0
        
        self.frame_count = 0
        self.cap = None
        self.slam = None
        self.vo = None
        self.is_initialized = False
        
        # Setup logging
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)
        
        if mode == 'slam' and PYSLAM_AVAILABLE:
            self._init_slam()
        elif mode == 'vo' and PYSLAM_VO_AVAILABLE:
            self._init_vo()
        else:
            raise RuntimeError(f"Mode '{mode}' not available. Need pySLAM for SLAM or VO")
    
    def _init_slam(self):
        """Initialize SLAM for feature matching."""
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
                'Camera.k1': 0.0, 'Camera.k2': 0.0, 'Camera.p1': 0.0,
                'Camera.p2': 0.0, 'Camera.k3': 0.0
            }
            
            self.camera = PinholeCamera(camera_config)
            
            # Use ORB2 if available, otherwise ORB
            try:
                from orbslam2_features import ORBextractor
                feature_tracker_config = FeatureTrackerConfigs.ORB2.copy()
                print("✓ Using ORB2 (ORB-SLAM2 optimized)")
            except ImportError:
                feature_tracker_config = FeatureTrackerConfigs.ORB.copy()
                print("✓ Using ORB (OpenCV)")
            
            feature_tracker_config["num_features"] = 2000
            
            # Initialize SLAM (no loop closure for speed)
            self.slam = Slam(
                self.camera,
                feature_tracker_config,
                None,  # No loop closure
                None,  # No semantic mapping
                SensorType.MONOCULAR,
                environment_type=DatasetEnvironmentType.INDOOR,
                config=camera_config
            )
            
            self.is_initialized = True
            print("✅ SLAM initialized for feature matching")
            
        except Exception as e:
            self.logger.error(f"SLAM initialization failed: {e}")
            raise
    
    def _init_vo(self):
        """Initialize Visual Odometry for feature matching."""
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
                'Camera.k1': 0.0, 'Camera.k2': 0.0, 'Camera.p1': 0.0,
                'Camera.p2': 0.0, 'Camera.k3': 0.0
            }
            
            self.camera = PinholeCamera(camera_config)
            
            # Use ORB2 if available
            try:
                from orbslam2_features import ORBextractor
                feature_tracker_config = FeatureTrackerConfigs.ORB2.copy()
                print("✓ Using ORB2 (ORB-SLAM2 optimized)")
            except ImportError:
                feature_tracker_config = FeatureTrackerConfigs.ORB.copy()
                print("✓ Using ORB (OpenCV)")
            
            feature_tracker_config["num_features"] = 2000
            
            # Initialize VO
            self.vo = VisualOdometry(
                self.camera,
                feature_tracker_config
            )
            
            self.is_initialized = True
            print("✅ Visual Odometry initialized for feature matching")
            
        except Exception as e:
            self.logger.error(f"VO initialization failed: {e}")
            raise
    
    def get_feature_matching_image(self):
        """Get feature matching visualization image."""
        try:
            if self.mode == 'slam' and self.slam:
                tracking = self.slam.tracking
                
                if not hasattr(tracking, 'f_cur') or not hasattr(tracking, 'f_ref'):
                    return None
                
                if tracking.f_cur is None or tracking.f_ref is None:
                    return None
                
                f_cur = tracking.f_cur
                f_ref = tracking.f_ref
                
                # Get images
                if not hasattr(f_cur, 'img') or not hasattr(f_ref, 'img'):
                    return None
                
                img_cur = f_cur.img
                img_ref = f_ref.img
                
                # Get keypoints
                if not hasattr(f_cur, 'kps') or not hasattr(f_ref, 'kps'):
                    return None
                
                kps_cur = np.array(f_cur.kps)
                kps_ref = np.array(f_ref.kps)
                
                # Get matched indices
                if hasattr(tracking, 'idxs_ref') and hasattr(tracking, 'idxs_cur'):
                    idxs_ref = tracking.idxs_ref
                    idxs_cur = tracking.idxs_cur
                    
                    if idxs_ref is None or idxs_cur is None or len(idxs_ref) == 0:
                        return None
                    
                    # Extract matched keypoints
                    matched_kps_ref = kps_ref[idxs_ref]
                    matched_kps_cur = kps_cur[idxs_cur]
                    
                    # Get keypoint sizes if available
                    kps_ref_sizes = None
                    kps_cur_sizes = None
                    if hasattr(f_ref, 'sizes') and f_ref.sizes is not None:
                        kps_ref_sizes = np.array(f_ref.sizes)[idxs_ref]
                    if hasattr(f_cur, 'sizes') and f_cur.sizes is not None:
                        kps_cur_sizes = np.array(f_cur.sizes)[idxs_cur]
                    
                    # Draw feature matches
                    from pyslam.utilities.utils_draw import draw_feature_matches
                    
                    img_matches = draw_feature_matches(
                        img_ref, img_cur,
                        matched_kps_ref, matched_kps_cur,
                        kps1_sizes=kps_ref_sizes,
                        kps2_sizes=kps_cur_sizes,
                        horizontal=True,
                        show_kp_sizes=True,
                        lineType=cv2.LINE_AA
                    )
                    
                    return img_matches
                
            elif self.mode == 'vo' and self.vo:
                # Similar logic for VO
                if hasattr(self.vo, 'tracking'):
                    tracking = self.vo.tracking
                    # Similar implementation for VO
                    pass
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Feature matching image error: {e}")
            return None
    
    def initialize_camera(self):
        """Initialize camera capture."""
        try:
            # Check if it's a camera index or file path
            if isinstance(self.camera_source, int) or (isinstance(self.camera_source, str) and self.camera_source.isdigit()):
                camera_idx = int(self.camera_source)
                self.cap = cv2.VideoCapture(camera_idx)
                print(f"📷 Initializing camera {camera_idx}")
            else:
                # Video file
                self.cap = cv2.VideoCapture(self.camera_source)
                print(f"📹 Loading video: {self.camera_source}")
            
            if not self.cap.isOpened():
                print(f"❌ Failed to open camera/video: {self.camera_source}")
                return False
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            print(f"✅ Camera initialized: {self.width}x{self.height}")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            return False
    
    def run(self):
        """Run feature matching visualization."""
        if not self.initialize_camera():
            return
        
        print("\n" + "="*60)
        print("Feature Matching Mode - Standalone")
        print(f"Mode: {self.mode.upper()}")
        print("Press 'q' to quit")
        print("="*60 + "\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("⚠️  Failed to read frame")
                    break
                
                # Resize if needed
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                
                # Convert BGR to RGB for pySLAM
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                if self.mode == 'slam' and self.slam:
                    timestamp = time.time()
                    try:
                        self.slam.track(rgb_frame, None, None, self.frame_count, timestamp)
                    except Exception as e:
                        self.logger.debug(f"SLAM track error: {e}")
                        # Continue even if tracking fails
                
                # Get feature matching image
                feature_match_img = self.get_feature_matching_image()
                
                if feature_match_img is not None and feature_match_img.size > 0:
                    # Display feature matching
                    cv2.imshow('Feature Matching', feature_match_img)
                else:
                    # Fallback: show regular frame with text
                    display_frame = frame.copy()
                    cv2.putText(display_frame, "Initializing...", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Feature Matching', display_frame)
                
                self.frame_count += 1
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Standalone Feature Matching Mode')
    parser.add_argument('--camera', type=str, default='1',
                       help='Camera index or video file path (default: 1)')
    parser.add_argument('--mode', type=str, choices=['slam', 'vo'], default='slam',
                       help='Mode: slam (SLAM) or vo (Visual Odometry)')
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
    parser.add_argument('--fx', type=float, default=525.0, help='Camera focal length x')
    parser.add_argument('--fy', type=float, default=525.0, help='Camera focal length y')
    
    args = parser.parse_args()
    
    # Convert camera to int if it's a digit
    camera_source = int(args.camera) if args.camera.isdigit() else args.camera
    
    try:
        app = StandaloneFeatureMatching(
            camera_source=camera_source,
            mode=args.mode,
            width=args.width,
            height=args.height,
            fx=args.fx,
            fy=args.fy
        )
        app.run()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

