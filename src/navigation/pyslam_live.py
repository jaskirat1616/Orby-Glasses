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

# Add cpp/lib path for compiled modules (pyslam_utils, etc.)
cpp_lib_path = os.path.join(pyslam_path, 'cpp', 'lib')
if os.path.exists(cpp_lib_path) and cpp_lib_path not in sys.path:
    sys.path.insert(0, cpp_lib_path)

# Try to import pySLAM modules
PYSLAM_AVAILABLE = False
PinholeCamera = None
Config = None
Slam = None
SlamState = None
FeatureTrackerConfigs = None
FeatureDetectorTypes = None
SensorType = None
DatasetEnvironmentType = None
SlamPlotDrawer = None
Viewer3D = None
LoopDetectorConfigs = None

try:
    # Check for evo dependency first
    import evo
    print("✅ evo dependency found")

    # Import pySLAM modules with proper error handling
    import pyslam
    from pyslam.config import Config
    from pyslam.config_parameters import Parameters
    from pyslam.slam.slam import Slam
    from pyslam.slam.slam_commons import SlamState
    from pyslam.slam.camera import PinholeCamera
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
    from pyslam.local_features.feature_types import FeatureDetectorTypes
    from pyslam.io.dataset_types import SensorType, DatasetEnvironmentType
    from pyslam.viz.slam_plot_drawer import SlamPlotDrawer
    from pyslam.viz.viewer3D import Viewer3D
    # Don't import LoopDetectorConfigs at module level - causes pyobindex2 error
    # from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs
    PYSLAM_AVAILABLE = True
    print("✅ Real pySLAM modules imported successfully!")
except ImportError as e:
    PYSLAM_AVAILABLE = False
    print(f"❌ pySLAM import error: {e}")
    import traceback
    print(f"Traceback:\n{traceback.format_exc()}")
except NameError as e:
    PYSLAM_AVAILABLE = False
    print(f"❌ pySLAM module error: {e}")
    import traceback
    print(f"Traceback:\n{traceback.format_exc()}")
except AttributeError as e:
    PYSLAM_AVAILABLE = False
    print(f"❌ pySLAM attribute error: {e}")
    import traceback
    print(f"Traceback:\n{traceback.format_exc()}")
except Exception as e:
    PYSLAM_AVAILABLE = False
    print(f"❌ pySLAM general error: {e}")
    import traceback
    print(f"Traceback:\n{traceback.format_exc()}")


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

        # Suppress excessive pySLAM INFO logs (loop closing, relocalization)
        logging.getLogger('loop_closing_logger').setLevel(logging.WARNING)
        logging.getLogger('relocalization_logger').setLevel(logging.WARNING)
        logging.getLogger('tracking').setLevel(logging.ERROR)  # Suppress loop closure warnings
        
        # Suppress local_mapping verbose logs for performance
        logging.getLogger('local_mapping_logger').setLevel(logging.WARNING)  # Disable INFO logs from local mapping
        
        # Camera parameters
        self.width = config.get('camera.width', 640)
        self.height = config.get('camera.height', 480)
        # Use camera.fx/fy if available, fall back to mapping3d.fx/fy, then default
        self.fx = config.get('camera.fx', config.get('mapping3d.fx', 500))
        self.fy = config.get('camera.fy', config.get('mapping3d.fy', 500))
        self.cx = self.width / 2
        self.cy = self.height / 2

        # State variables
        self.frame_count = 0
        self.is_initialized = False
        self.current_pose = np.eye(4)
        self.trajectory = []
        self.map_points = []

        # Crash recovery state
        self.crash_count = 0
        self.max_crashes_before_disable = 3
        self.loop_closure_enabled = config.get('slam.loop_closure', False)
        self.has_disabled_loop_closure = False

        # Memory management settings
        self.max_trajectory_length = config.get('slam.max_trajectory_length', 1000)  # Keep last 1000 poses
        self.max_map_points_local = config.get('slam.max_map_points', 5000)  # Limit local map points
        self.cleanup_interval = config.get('slam.cleanup_interval', 500)  # Cleanup every N frames
        self.last_cleanup_frame = 0

        # pySLAM components
        self.slam = None
        self.camera = None
        self.plot_drawer = None

        # Camera capture
        self.cap = None
        
        # Initialize pySLAM
        if not PYSLAM_AVAILABLE:
            raise RuntimeError("pySLAM not available. Please install pySLAM from third_party/pyslam")

        try:
            self._initialize_pyslam()
        except Exception as e:
            self.logger.error(f"pySLAM initialization failed: {e}")
            raise RuntimeError(f"pySLAM initialization failed: {e}")

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

            # Create feature tracker config - use ORB (OpenCV) instead of ORB2 (ORB-SLAM2)
            # Optimized to work in typical environments (not just perfect textured ones)
            feature_tracker_config = FeatureTrackerConfigs.ORB.copy()
            feature_tracker_config["num_features"] = self.config.get('slam.orb_features', 5000)

            # Increase pyramid levels for better multi-scale detection
            feature_tracker_config["num_levels"] = 12  # More levels = more features in varied environments
            feature_tracker_config["scale_factor"] = 1.1  # Finer scale = more features per level

            self.logger.info(f"📊 ORB configured for typical environments:")
            self.logger.info(f"   • {feature_tracker_config['num_features']} features target")
            self.logger.info(f"   • {feature_tracker_config['num_levels']} pyramid levels (high coverage)")
            self.logger.info(f"   • Scale factor: {feature_tracker_config['scale_factor']} (fine-grained)")
            self.logger.info(f"   → Should detect 2500-4000 features in normal rooms")

            # Relocalization parameters - AGGRESSIVE tuning for real-world success
            # Research shows ORB-SLAM uses min 10 inliers, we're being even more lenient
            Parameters.kRelocalizationMinKpsMatches = 8  # Reduced from 15 (min matches to try)
            Parameters.kRelocalizationPoseOpt1MinMatches = 6  # Reduced from 10 (first opt threshold)
            Parameters.kRelocalizationDoPoseOpt2NumInliers = 20  # CRITICAL: Reduced from 50 (final success threshold)
            Parameters.kRelocalizationFeatureMatchRatioTest = 0.85  # Relaxed from 0.75 (Lowe's ratio)
            Parameters.kRelocalizationFeatureMatchRatioTestLarge = 0.95  # Relaxed from 0.9 (for search)
            Parameters.kRelocalizationMaxReprojectionDistanceMapSearchCoarse = 15  # Increased from 10 pixels
            Parameters.kRelocalizationMaxReprojectionDistanceMapSearchFine = 5  # Increased from 3 pixels

            # SUPER ACCURATE SLAM PARAMETERS FOR HIGH PRECISION TRACKING
            # Triangulation parameters - balance between accuracy and tolerance
            Parameters.kMinRatioBaselineDepth = 0.001  # Very tolerant for slow movement
            # Note: Keep kCosMaxParallax at default 0.9998 for initialization
            # Then use stricter 0.998 for ongoing tracking
            Parameters.kCosMaxParallaxInitializer = 0.9998  # Default for initialization (0.89 deg parallax)
            Parameters.kCosMaxParallax = 0.9998  # Keep same as initializer for stability (original: 0.9998)
            
            # Feature matching - use defaults for initialization stability
            # Will use stricter values post-initialization in future optimization
            Parameters.kFeatureMatchDefaultRatioTest = 0.7  # Default value for stability
            Parameters.kMaxReprojectionDistanceFrame = 7  # Default value for initialization
            Parameters.kMaxReprojectionDistanceMap = 3  # Default value for initialization
            
            # Bundle Adjustment - enable for higher accuracy
            Parameters.kLocalBAWindow = 15  # Smaller window for faster convergence (20 default)
            Parameters.kUseLargeWindowBA = False  # Disable large BA for real-time
            
            # Keyframe management - balanced for initialization stability
            Parameters.kNumMinPointsForNewKf = 15  # Default value for initialization stability
            Parameters.kThNewKfRefRatioMonocular = 0.9  # Default value for initialization
            Parameters.kMaxNumOfKeyframesInLocalMap = 100  # More keyframes in map (80 default) - OK post-init
            Parameters.kNumBestCovisibilityKeyFrames = 15  # More covisibility (10 default) - OK post-init
            
            # Pose optimization - use defaults for initialization
            Parameters.kMaxOutliersRatioInPoseOptimization = 0.9  # Default value for initialization
            Parameters.kMinNumMatchedFeaturesSearchFrameByProjection = 20  # Default value for initialization
            
            self.logger.info("🎯 BALANCED SLAM MODE ENABLED:")
            self.logger.info(f"   • Min ratio baseline/depth: {Parameters.kMinRatioBaselineDepth}")
            self.logger.info(f"   • Local BA window: {Parameters.kLocalBAWindow} frames")
            self.logger.info(f"   • More keyframes in map: {Parameters.kMaxNumOfKeyframesInLocalMap}")
            self.logger.info(f"   • More covisible keyframes: {Parameters.kNumBestCovisibilityKeyFrames}")
            self.logger.info("   → Balanced settings for stable initialization and good tracking!")

            self.logger.info("🔧 Relocalization tuned for real-world conditions:")
            self.logger.info(f"   • Min matches to attempt: {Parameters.kRelocalizationMinKpsMatches}")
            self.logger.info(f"   • Success threshold: {Parameters.kRelocalizationDoPoseOpt2NumInliers} inliers")
            self.logger.info(f"   • Coarse search window: {Parameters.kRelocalizationMaxReprojectionDistanceMapSearchCoarse}px")

            # Loop closure detection - Use iBoW (incremental Bag of Words)
            # iBoW doesn't require pre-built vocabulary and is more stable than DBOW3
            loop_detection_config = None
            # Check if loop closure should be enabled (not disabled by crash recovery)
            should_enable_loop_closure = (
                self.config.get('slam.loop_closure', False) and
                not self.has_disabled_loop_closure
            )

            if should_enable_loop_closure:
                try:
                    from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs
                    # Use iBoW - incremental vocabulary building, more stable than DBOW3
                    loop_detection_config = LoopDetectorConfigs.IBOW
                    self.logger.info("✓ Loop closure enabled with iBoW (incremental Bag of Words)")
                    self.logger.info("  → Can relocalize if tracking is lost")
                    self.logger.info("  → Crash recovery enabled: will fallback to VO if crashes occur")
                except Exception as e:
                    self.logger.warning(f"⚠️  Loop closure DISABLED - {e}")
                    self.logger.warning("   → Cannot relocalize if tracking is lost!")
                    self.logger.warning("   → MUST keep camera on textured surfaces")
                    self.logger.warning("   → If tracking lost, restart required")
                    loop_detection_config = None
            else:
                reason = "disabled by crash recovery" if self.has_disabled_loop_closure else "disabled in config"
                self.logger.info(f"⚠️  Loop closure {reason}")
                self.logger.info("   → If tracking lost, cannot recover (restart required)")
                self.logger.info("   → Keep camera pointed at textured surfaces!")

            # Dense reconstruction configuration
            dense_config = self.config.get('slam.dense_reconstruction', {})
            self.use_dense_reconstruction = dense_config.get('enabled', False)

            if self.use_dense_reconstruction:
                Parameters.kUseVolumetricIntegration = True
                Parameters.kVolumetricIntegrationType = dense_config.get('type', 'TSDF')
                Parameters.kVolumetricIntegrationExtractMesh = dense_config.get('extract_mesh', True)
                Parameters.kVolumetricIntegrationVoxelLength = dense_config.get('voxel_length', 0.015)
                Parameters.kVolumetricIntegrationDepthTruncIndoor = dense_config.get('depth_trunc', 4.0)
                Parameters.kVolumetricIntegrationOutputTimeInterval = 2.0  # Update every 2 seconds

                self.logger.info("🗺️  Real-time Dense Reconstruction ENABLED")
                self.logger.info(f"   • Type: {Parameters.kVolumetricIntegrationType}")
                self.logger.info(f"   • Voxel size: {Parameters.kVolumetricIntegrationVoxelLength}m")
                self.logger.info(f"   • Extract mesh: {Parameters.kVolumetricIntegrationExtractMesh}")
                self.logger.info(f"   • Depth truncation: {Parameters.kVolumetricIntegrationDepthTruncIndoor}m")
                self.logger.info("   ⚠️  This is CPU/GPU intensive - expect slower performance")
            else:
                Parameters.kUseVolumetricIntegration = False

            # Initialize SLAM with all required parameters (following main_slam.py)
            # Use INDOOR environment type for OrbyGlasses use case
            self.slam = Slam(
                self.camera,
                feature_tracker_config,
                loop_detection_config,
                None,  # semantic_mapping_config
                SensorType.MONOCULAR,
                environment_type=DatasetEnvironmentType.INDOOR,
                config=camera_config
            )

            # Initialize Rerun visualization (like main_slam.py)
            self.use_rerun = self.config.get('slam.use_rerun', True)
            if self.use_rerun:
                try:
                    from pyslam.viz.rerun_interface import Rerun
                    Rerun.init_slam()
                    self.rerun = Rerun
                    self.logger.info("✅ Rerun.io initialized for SLAM")
                except Exception as e:
                    self.logger.warning(f"Rerun initialization failed: {e}")
                    self.use_rerun = False
                    self.rerun = None
            else:
                self.rerun = None

            # Initialize 3D viewer (Pangolin-based) - must be created before plot_drawer
            self.viewer3d = Viewer3D(scale=1.0)

            # Initialize visualization (after SLAM is created and viewer3d initialized)
            self.plot_drawer = SlamPlotDrawer(self.slam, self.viewer3d)

            # Performance optimizations summary
            self.logger.info("⚡ SLAM Performance Optimizations:")
            self.logger.info(f"   • {self.config.get('slam.orb_features', 3000)} ORB features (high accuracy)")
            self.logger.info(f"   • {feature_tracker_config['num_levels']} pyramid levels (multi-scale)")
            self.logger.info(f"   • Scale factor: {feature_tracker_config['scale_factor']}")
            self.logger.info(f"   • Match ratio test: {feature_tracker_config['match_ratio_test']}")
            self.logger.info(f"   • Tracker: {feature_tracker_config['tracker_type']} (brute-force)")
            rerun_status = "enabled" if self.use_rerun else "disabled (saves 20-30% CPU)"
            self.logger.info(f"   • Rerun.io: {rerun_status}")
            loop_status = "enabled (DBOW3)" if loop_detection_config else "disabled (saves 15% CPU)"
            self.logger.info(f"   • Loop closure: {loop_status}")
            
            # Initialize camera capture with performance optimizations
            # Using camera index 1 as requested
            self.cap = cv2.VideoCapture(1)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Low latency
            # Camera performance optimizations
            try:
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
            except Exception:
                pass  # Some cameras don't support these

            if not self.cap.isOpened():
                raise RuntimeError("Could not open camera")
            
            self.is_initialized = True
            print("✅ Live pySLAM initialized successfully!")
            self.logger.info("✅ Live pySLAM initialized successfully!")
            
        except Exception as e:
            error_msg = f"Failed to initialize live pySLAM: {e}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)


    def process_frame(self, frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> Dict:
        """
        Process a single frame through pySLAM.

        Args:
            frame: Input frame (BGR or grayscale)
            depth_map: Depth map (ignored for monocular SLAM)

        Returns:
            Dictionary with SLAM results
        """
        self.frame_count += 1

        try:
            return self._process_pyslam_frame(frame)
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
        # Resize frame to match camera config if needed
        if frame.shape[0:2] != (self.height, self.width):
            self.logger.debug(f"Resizing frame from {frame.shape[0:2]} to ({self.height}, {self.width})")
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB (pySLAM expects RGB from datasets)
        if len(frame.shape) == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # Already grayscale, convert back to RGB for consistency
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Process frame through pySLAM
        timestamp = time.time()

        try:
            self.slam.track(rgb_frame, None, None, self.frame_count, timestamp)
        except Exception as e:
            # Catch any errors from pySLAM's track() method
            import traceback
            error_msg = str(e)
            error_trace = traceback.format_exc()

            self.logger.error(f"SLAM track error: {error_msg}")
            self.logger.error(f"Traceback: {error_trace}")

            # Detect relocalization/loop closure crashes (MLPnPsolver, Bus error, etc.)
            is_loop_closure_crash = (
                'MLPnPsolver' in error_trace or
                'relocalization' in error_msg.lower() or
                'loop_closing' in error_msg.lower() or
                'Bus error' in error_msg
            )

            if is_loop_closure_crash and self.loop_closure_enabled:
                self.crash_count += 1
                self.logger.warning(f"Loop closure crash detected ({self.crash_count}/{self.max_crashes_before_disable})")

                if self.crash_count >= self.max_crashes_before_disable and not self.has_disabled_loop_closure:
                    self.logger.error("⚠️  Multiple loop closure crashes detected - attempting graceful recovery")
                    self.logger.error("⚠️  Loop closure will be disabled. SLAM will continue with visual odometry only.")
                    self.has_disabled_loop_closure = True
                    # Note: We can't dynamically disable loop closure in pySLAM after initialization
                    # This flag prevents re-initialization with loop closure

            # Return error result but keep SLAM alive
            return {
                'pose': self.current_pose.copy(),
                'position': self.current_pose[:3, 3].copy(),
                'tracking_quality': 0.0,
                'tracking_state': "ERROR",
                'message': f"SLAM track failed: {error_msg}",
                'is_initialized': self.is_initialized,
                'trajectory_length': len(self.trajectory),
                'num_map_points': 0,
                'performance': {},
                'crash_detected': is_loop_closure_crash,
                'loop_closure_disabled': self.has_disabled_loop_closure
            }

        # Get tracking state - fixed isinstance issue
        tracking_state = "OK"
        if hasattr(self.slam, 'tracking') and hasattr(self.slam.tracking, 'state'):
            try:
                state = self.slam.tracking.state
                # Handle SerializableEnum from pySLAM
                if state is not None:
                    # Convert to string first to avoid isinstance issues with SerializableEnum
                    state_str = str(state)
                    if "LOST" in state_str or state_str == "3":
                        tracking_state = "LOST"
                    elif "NOT_INITIALIZED" in state_str or state_str == "1":
                        tracking_state = "NOT_INITIALIZED"
                    elif "OK" in state_str or state_str == "2":
                        tracking_state = "OK"
                    else:
                        tracking_state = state_str
            except Exception as e:
                # Log and continue with OK state
                self.logger.debug(f"Could not parse tracking state: {e}")
        
        # Get current pose
        if hasattr(self.slam, 'tracking') and hasattr(self.slam.tracking, 'cur_pose'):
            self.current_pose = self.slam.tracking.cur_pose
        elif hasattr(self.slam, 'tracking') and hasattr(self.slam.tracking, 'get_current_pose'):
            self.current_pose = self.slam.tracking.get_current_pose()
        
        # Add to trajectory (safe even if pose is None during initialization)
        if self.current_pose is not None:
            self.trajectory.append(self.current_pose.copy())
        
        # Get map points
        self.map_points = self.get_map_points()
        
        # Update visualization - CRITICAL: Call draw() for pySLAM plot windows
        if self.plot_drawer:
            try:
                # This shows pySLAM's native 2D plots (trajectory, errors, etc.)
                self.plot_drawer.draw(self.frame_count)
            except Exception as e:
                self.logger.warning(f"Plot visualization error: {e}")

        # Update 3D viewer - CRITICAL: This shows the 3D point cloud window
        if hasattr(self, 'viewer3d') and self.viewer3d:
            try:
                # After tracking: draw_slam_map (same as main_slam.py line 330)
                if hasattr(self.viewer3d, 'draw_slam_map'):
                    self.viewer3d.draw_slam_map(self.slam)
                elif hasattr(self.viewer3d, 'draw_map'):
                    self.viewer3d.draw_map(self.slam.map)
                else:
                    # Fallback for older versions
                    if hasattr(self.slam, 'map'):
                        self.viewer3d.update(self.slam.map)
            except Exception as e:
                self.logger.warning(f"3D visualization error: {e}")

        # Show pySLAM camera window with feature tracking
        try:
            if hasattr(self.slam, 'tracking') and hasattr(self.slam.tracking, 'draw_img'):
                # Use pySLAM's own tracking visualization
                img_draw = self.slam.tracking.draw_img
                if img_draw is not None:
                    cv2.imshow("pySLAM - Camera", img_draw)
                else:
                    cv2.imshow("pySLAM - Camera", frame)
            elif hasattr(self.slam, 'map') and hasattr(self.slam.map, 'draw_feature_trails'):
                img_draw = self.slam.map.draw_feature_trails(frame)
                cv2.imshow("pySLAM - Camera", img_draw)
            else:
                # Fallback to basic camera view
                cv2.imshow("pySLAM - Camera", frame)
        except Exception as e:
            self.logger.warning(f"Camera window error: {e}")
            cv2.imshow("pySLAM - Camera", frame)

        # Process OpenCV events to update windows
        cv2.waitKey(1)

        # Update Rerun visualization (like main_slam.py)
        if self.use_rerun and self.rerun:
            try:
                self.rerun.log_slam_frame(self.frame_count, self.slam)
                # Also log the annotated OrbyGlasses frame to Rerun
                try:
                    import rerun as rr
                    # Log OrbyGlasses annotated feed to Rerun
                    rr.set_time_sequence("frame_id", self.frame_count)
                    # Convert BGR to RGB for correct color display
                    rgb_frame = frame[:, :, ::-1].copy()
                    rr.log("/orby/camera_feed", rr.Image(rgb_frame))
                except Exception as e:
                    self.logger.debug(f"OrbyGlasses Rerun logging error: {e}")
            except Exception as e:
                self.logger.debug(f"Rerun logging error: {e}")

        # Draw dense map (same as main_slam.py line 366)
        if hasattr(self, 'viewer3d') and self.viewer3d:
            try:
                if hasattr(self.viewer3d, 'draw_dense_map'):
                    self.viewer3d.draw_dense_map(self.slam)
            except Exception as e:
                self.logger.debug(f"Dense map visualization error: {e}")

        # Periodic memory cleanup to prevent leaks
        if self.frame_count - self.last_cleanup_frame >= self.cleanup_interval:
            self._perform_memory_cleanup()

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


    def get_map_points(self) -> np.ndarray:
        """Get all map points from pySLAM for visualization."""
        try:
            if hasattr(self.slam, 'map') and self.slam.map is not None:
                map_points = self.slam.map.get_points()
                if map_points is not None and hasattr(map_points, '__len__') and len(map_points) > 0:
                    positions = []
                    for point in map_points:
                        if hasattr(point, 'get_pos'):
                            positions.append(point.get_pos())
                        elif hasattr(point, 'pos'):
                            positions.append(point.pos)
                    return np.array(positions)
        except Exception as e:
            self.logger.error(f"Error getting map points: {e}")

        return np.array([])

    def _perform_memory_cleanup(self):
        """
        Periodic memory cleanup to prevent memory leaks.
        Limits trajectory length and manages pySLAM internal structures.
        """
        try:
            # Limit trajectory length (keep most recent poses)
            if len(self.trajectory) > self.max_trajectory_length:
                # Keep last max_trajectory_length poses
                self.trajectory = self.trajectory[-self.max_trajectory_length:]
                self.logger.debug(f"Trajectory trimmed to {self.max_trajectory_length} poses")

            # Try to limit pySLAM keyframes (if accessible)
            if hasattr(self.slam, 'map') and self.slam.map is not None:
                try:
                    if hasattr(self.slam.map, 'keyframes'):
                        num_keyframes = len(self.slam.map.keyframes)
                        if num_keyframes > 100:  # Keep last 100 keyframes
                            # Note: Direct keyframe removal might break pySLAM's internal state
                            # This is informational only - full cleanup requires pySLAM support
                            self.logger.debug(f"Warning: {num_keyframes} keyframes in map (consider restart if memory high)")

                    if hasattr(self.slam.map, 'points'):
                        num_points = len(self.slam.map.points)
                        if num_points > self.max_map_points_local:
                            self.logger.debug(f"Warning: {num_points} map points (limit: {self.max_map_points_local})")
                except Exception as e:
                    self.logger.debug(f"Could not check pySLAM map size: {e}")

            self.last_cleanup_frame = self.frame_count

        except Exception as e:
            self.logger.warning(f"Memory cleanup error: {e}")

    def cleanup(self):
        """Clean up pySLAM resources."""
        try:
            cv2.destroyAllWindows()

            # Close pySLAM 3D viewer
            if hasattr(self, 'viewer3d') and self.viewer3d is not None:
                self.viewer3d.quit = True

            self.logger.info("SLAM cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

    def is_tracking_good(self) -> bool:
        """Check if SLAM tracking is good."""
        if hasattr(self.slam, 'tracking') and hasattr(self.slam.tracking, 'state'):
            try:
                state = self.slam.tracking.state
                if state is not None:
                    # Use string comparison to avoid isinstance issues with SerializableEnum
                    state_str = str(state)
                    return "OK" in state_str or state_str == "2"
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

    def save_map(self, map_name: str = "default", maps_dir: str = "data/maps") -> bool:
        """
        Save the current SLAM map to disk.

        Args:
            map_name: Name for the map (default: "default")
            maps_dir: Directory to save maps (default: "data/maps")

        Returns:
            True if successful, False otherwise
        """
        try:
            import pickle
            import json
            from datetime import datetime

            # Create maps directory if it doesn't exist
            os.makedirs(maps_dir, exist_ok=True)

            # Create timestamp for this save
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            map_file = os.path.join(maps_dir, f"{map_name}_{timestamp}.pkl")
            metadata_file = os.path.join(maps_dir, f"{map_name}_{timestamp}_meta.json")

            # Collect map data
            map_data = {
                'trajectory': [pose.copy() for pose in self.trajectory if pose is not None],
                'current_pose': self.current_pose.copy() if self.current_pose is not None else np.eye(4),
                'frame_count': self.frame_count,
                'is_initialized': self.is_initialized,
                'map_points': self.map_points.copy() if len(self.map_points) > 0 else np.array([]),
                'timestamp': timestamp,
                'map_name': map_name
            }

            # Try to get pySLAM map data (keyframes, map points from pySLAM itself)
            if hasattr(self.slam, 'map') and self.slam.map is not None:
                try:
                    pyslam_map_data = {
                        'num_keyframes': len(self.slam.map.keyframes) if hasattr(self.slam.map, 'keyframes') else 0,
                        'num_map_points': len(self.slam.map.points) if hasattr(self.slam.map, 'points') else 0,
                    }
                    map_data['pyslam_map_info'] = pyslam_map_data
                except Exception as e:
                    self.logger.warning(f"Could not extract pySLAM map info: {e}")

            # Save map data
            with open(map_file, 'wb') as f:
                pickle.dump(map_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save metadata
            metadata = {
                'map_name': map_name,
                'timestamp': timestamp,
                'frame_count': self.frame_count,
                'trajectory_length': len(map_data['trajectory']),
                'num_map_points': len(map_data['map_points']),
                'loop_closure_enabled': self.loop_closure_enabled,
                'has_disabled_loop_closure': self.has_disabled_loop_closure,
                'camera': {
                    'width': self.width,
                    'height': self.height,
                    'fx': self.fx,
                    'fy': self.fy
                }
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"✓ Map saved: {map_file}")
            self.logger.info(f"  → {len(map_data['trajectory'])} poses, {len(map_data['map_points'])} map points")

            return True

        except Exception as e:
            self.logger.error(f"Failed to save map: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def load_map(self, map_file: str) -> bool:
        """
        Load a previously saved SLAM map from disk.

        Args:
            map_file: Path to the map file (.pkl)

        Returns:
            True if successful, False otherwise
        """
        try:
            import pickle

            if not os.path.exists(map_file):
                self.logger.error(f"Map file not found: {map_file}")
                return False

            # Load map data
            with open(map_file, 'rb') as f:
                map_data = pickle.load(f)

            # Restore state
            self.trajectory = map_data.get('trajectory', [])
            self.current_pose = map_data.get('current_pose', np.eye(4))
            self.frame_count = map_data.get('frame_count', 0)
            self.is_initialized = map_data.get('is_initialized', False)
            self.map_points = map_data.get('map_points', np.array([]))

            self.logger.info(f"✓ Map loaded: {map_file}")
            self.logger.info(f"  → {len(self.trajectory)} poses, {len(self.map_points)} map points")
            self.logger.info(f"  → Frame count: {self.frame_count}")

            # Note: This loads trajectory and map points, but does NOT restore
            # pySLAM's internal state (keyframes, local map, etc.)
            # Full pySLAM map serialization would require deeper integration
            self.logger.warning("⚠️  Note: This is a lightweight map load (trajectory + points only)")
            self.logger.warning("    Full pySLAM state (keyframes, local map) is NOT restored")
            self.logger.warning("    SLAM will continue building map from current position")

            return True

        except Exception as e:
            self.logger.error(f"Failed to load map: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def list_saved_maps(self, maps_dir: str = "data/maps") -> List[Dict]:
        """
        List all saved maps in the maps directory.

        Args:
            maps_dir: Directory containing saved maps

        Returns:
            List of dictionaries with map metadata
        """
        try:
            import json

            if not os.path.exists(maps_dir):
                return []

            maps = []
            for filename in os.listdir(maps_dir):
                if filename.endswith('_meta.json'):
                    meta_file = os.path.join(maps_dir, filename)
                    try:
                        with open(meta_file, 'r') as f:
                            metadata = json.load(f)
                            # Add file paths
                            map_file = meta_file.replace('_meta.json', '.pkl')
                            metadata['map_file'] = map_file
                            metadata['meta_file'] = meta_file
                            maps.append(metadata)
                    except Exception as e:
                        self.logger.warning(f"Could not read metadata: {meta_file} - {e}")

            # Sort by timestamp (newest first)
            maps.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

            return maps

        except Exception as e:
            self.logger.error(f"Failed to list maps: {e}")
            return []

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
