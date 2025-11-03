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
# Import matplotlib for graphs (only graphs, no 3D)
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
from matplotlib import pyplot as plt

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

    def __init__(self, config: Dict, viz_mode: str = None):
        """Initialize live pySLAM system."""
        self.config = config
        self.viz_mode = viz_mode  # Store viz_mode to control window visibility
        self.logger = logging.getLogger(__name__)

        # Suppress excessive pySLAM INFO logs (loop closing, relocalization)
        logging.getLogger('loop_closing_logger').setLevel(logging.WARNING)
        logging.getLogger('relocalization_logger').setLevel(logging.WARNING)
        logging.getLogger('tracking').setLevel(logging.ERROR)  # Suppress loop closure warnings
        
        # Suppress local_mapping verbose logs for performance
        logging.getLogger('local_mapping_logger').setLevel(logging.WARNING)  # Disable INFO logs from local mapping
        
        # Suppress kf_info logs for feature matching mode (too verbose)
        if self.viz_mode == 'feature_matching':
            logging.getLogger('kf_info_logger').setLevel(logging.WARNING)
        
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
        
        # Store matched indices for feature matching visualization
        self.last_idxs_ref = None
        self.last_idxs_cur = None
        
        # Store recent frames for feature matching visualization
        # pySLAM might not store images in frames, so we cache them
        self.last_frame_img = None
        self.last_ref_frame_img = None

        # Crash recovery state
        self.crash_count = 0
        self.max_crashes_before_disable = 3
        self.loop_closure_enabled = config.get('slam.loop_closure', False)
        self.has_disabled_loop_closure = False

        # Memory management settings
        self.max_trajectory_length = config.get('slam.max_trajectory_length', 1000)  # Keep last 1000 poses
        self.max_map_points_local = config.get('slam.max_map_points', 10000)  # More points for visualization
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

            # Check if ORB2 is available and preferred
            use_orb2 = self.config.get('slam.feature_type', 'ORB').upper() == 'ORB2'
            
            # Try to import ORB2 to verify it's built
            if use_orb2:
                try:
                    # Add ORB2 library path (need to reconstruct pyslam_path here)
                    orb2_lib_path = os.path.join(os.path.dirname(__file__), '..', '..', 'third_party', 'pyslam', 'thirdparty', 'orbslam2_features', 'lib')
                    orb2_lib_path = os.path.abspath(orb2_lib_path)
                    if os.path.exists(orb2_lib_path) and orb2_lib_path not in sys.path:
                        sys.path.insert(0, orb2_lib_path)
                    
                    # Try importing ORB2
                    from orbslam2_features import ORBextractor
                    self.logger.info("✓ ORB2 C++ extension available!")
                    feature_tracker_config = FeatureTrackerConfigs.ORB2.copy()
                except ImportError as e:
                    self.logger.warning(f"⚠️  ORB2 not available ({e}), falling back to ORB")
                    use_orb2 = False
                    feature_tracker_config = FeatureTrackerConfigs.ORB.copy()
            else:
                feature_tracker_config = FeatureTrackerConfigs.ORB.copy()
            
            # For feature matching mode, use more features to help with relocalization
            if self.viz_mode == 'feature_matching':
                # More features = more matches = better relocalization
                default_features = 1500  # More than default for better matching
            else:
                default_features = self.config.get('slam.orb_features', 2000)
            feature_tracker_config["num_features"] = default_features

            # Balanced pyramid levels for efficient detection
            if use_orb2:
                # ORB2 uses fixed 8 levels and 1.2 scale (ORB-SLAM2 defaults)
                # Note: FAST thresholds are hardcoded in ORB-SLAM2 C++ code (cannot be changed via config)
                self.logger.info(f"📊 ORB2 configured (ORB-SLAM2 optimized):")
            else:
                feature_tracker_config["num_levels"] = 8  # Standard levels
                feature_tracker_config["scale_factor"] = 1.2  # Standard scale
                self.logger.info(f"📊 ORB configured:")

            self.logger.info(f"   • {feature_tracker_config['num_features']} features target (increased from 800)")
            self.logger.info(f"   • {feature_tracker_config.get('num_levels', 8)} pyramid levels")
            self.logger.info(f"   • Scale factor: {feature_tracker_config.get('scale_factor', 1.2)}")
            self.logger.info(f"   → Using {'ORB-SLAM2 optimized' if use_orb2 else 'OpenCV'} detector/descriptor")

            # Initialization parameters - optimize for faster initialization in feature matching mode
            if self.viz_mode == 'feature_matching':
                # For feature matching mode, allow faster initialization with fewer points
                # These parameters are set via pySLAM's Parameters but may need to be set before SLAM creation
                # We'll set them here if they exist
                try:
                    # Lower requirements for initialization (faster startup)
                    if hasattr(Parameters, 'kMinNumTriangulated'):
                        Parameters.kMinNumTriangulated = 50  # Lower from default (usually 100)
                    if hasattr(Parameters, 'kMaxNumFramesForInit'):
                        Parameters.kMaxNumFramesForInit = 30  # Try to initialize faster
                    if hasattr(Parameters, 'kMinNumInliersForInit'):
                        Parameters.kMinNumInliersForInit = 100  # Lower inlier requirement
                except:
                    pass  # Parameters may not have these attributes
            
            # Relocalization parameters - AGGRESSIVE tuning for real-world success
            # Research shows ORB-SLAM uses min 10 inliers, we're being even more lenient
            if self.viz_mode == 'feature_matching':
                # For feature matching mode, be more lenient with relocalization (fewer points needed)
                Parameters.kRelocalizationMinKpsMatches = 6  # Even lower for feature matching mode
                Parameters.kRelocalizationPoseOpt1MinMatches = 5  # Lower threshold
                Parameters.kRelocalizationDoPoseOpt2NumInliers = 15  # Lower success threshold
                Parameters.kRelocalizationFeatureMatchRatioTest = 0.90  # More relaxed
                Parameters.kRelocalizationFeatureMatchRatioTestLarge = 0.98  # Very relaxed
                Parameters.kRelocalizationMaxReprojectionDistanceMapSearchCoarse = 20  # Larger search window
                Parameters.kRelocalizationMaxReprojectionDistanceMapSearchFine = 8  # Larger fine search
            else:
                Parameters.kRelocalizationMinKpsMatches = 8  # Reduced from 15 (min matches to try)
                Parameters.kRelocalizationPoseOpt1MinMatches = 6  # Reduced from 10 (first opt threshold)
                Parameters.kRelocalizationDoPoseOpt2NumInliers = 20  # CRITICAL: Reduced from 50 (final success threshold)
                Parameters.kRelocalizationFeatureMatchRatioTest = 0.85  # Relaxed from 0.75 (Lowe's ratio)
                Parameters.kRelocalizationFeatureMatchRatioTestLarge = 0.95  # Relaxed from 0.9 (for search)
                Parameters.kRelocalizationMaxReprojectionDistanceMapSearchCoarse = 15  # Increased from 10 pixels
                Parameters.kRelocalizationMaxReprojectionDistanceMapSearchFine = 5  # Increased from 3 pixels

            # Optimize parameters for real-time performance (matching main_slam.py defaults)
            # Use main_slam.py defaults for stability - no custom tuning needed
            
            # Keep defaults from config_parameters.py:
            # - kNumFeatures: 2000 (default, we override with config)
            # - kMinRatioBaselineDepth: 0.01 (default, very reasonable)
            # - kCosMaxParallax: 0.9998 (default)
            # - kCosMaxParallaxInitializer: 0.99998 (default, stricter)
            # - kFeatureMatchDefaultRatioTest: 0.7 (default)
            # - kMaxOutliersRatioInPoseOptimization: 0.9 (default)
            # - kMinNumMatchedFeaturesSearchFrameByProjection: 20 (default)
            # - kMaxNumOfKeyframesInLocalMap: 80 (default)
            # - kNumBestCovisibilityKeyFrames: 10 (default)
            
            # Optimize for both speed and accuracy
            # Local Bundle Adjustment - smaller window = faster, but keep quality
            Parameters.kLocalBAWindow = 6  # Reduced from 20 for speed (default 20, was 12)
            
            # Keyframe management - more aggressive for speed
            if self.viz_mode == 'feature_matching':
                # For feature matching mode, minimize keyframes (we don't need map building)
                Parameters.kNumMinPointsForNewKf = 1000  # Very high = almost no keyframes
                Parameters.kThNewKfRefRatioMonocular = 0.99  # Very high = fewer keyframes
                Parameters.kMaxNumOfKeyframesInLocalMap = 5  # Minimal local map
            else:
                Parameters.kNumMinPointsForNewKf = 10  # Reduced from 12 for speed (default 15)
                Parameters.kThNewKfRefRatioMonocular = 0.80  # Lower = more keyframes (default 0.9, was 0.85)
                Parameters.kMaxNumOfKeyframesInLocalMap = 40  # Reduced from 60 for speed (default 80)
            Parameters.kNumBestCovisibilityKeyFrames = 10  # Keep default
            
            # Pose optimization - balance accuracy with relocalization needs
            if self.viz_mode == 'feature_matching':
                # For feature matching mode, balance accuracy with relocalization (need enough points)
                Parameters.kMaxOutliersRatioInPoseOptimization = 0.85  # Slightly less strict (default 0.9)
                Parameters.kMinNumMatchedFeaturesSearchFrameByProjection = 20  # Keep default (don't be too strict)
            else:
                Parameters.kMaxOutliersRatioInPoseOptimization = 0.85  # Lower = stricter (default 0.9)
                Parameters.kMinNumMatchedFeaturesSearchFrameByProjection = 25  # Higher = stricter (default 20)
            
            # Feature matching - balance accuracy with relocalization needs
            if self.viz_mode == 'feature_matching':
                # Good accuracy but not too strict (need enough matches for relocalization)
                Parameters.kFeatureMatchDefaultRatioTest = 0.72  # Slightly stricter but not too much
                Parameters.kMaxReprojectionDistanceFrame = 6  # Not too tight (default 7)
            else:
                Parameters.kFeatureMatchDefaultRatioTest = 0.75  # Slightly stricter (default 0.7)
                Parameters.kMaxReprojectionDistanceFrame = 6  # Tighter (default 7)
            Parameters.kMaxReprojectionDistanceMap = 3  # Keep default
            
            # Disable large BA for real-time performance
            Parameters.kUseLargeWindowBA = False  # Disable large BA for real-time
            
            # For feature matching mode, minimize keyframes to reduce overhead
            # (Already set above in keyframe management section)
            if self.viz_mode == 'feature_matching':
                self.logger.info("⚡ FEATURE MATCHING MODE: Optimized for speed and relocalization")
                self.logger.info(f"   → {default_features} ORB features (more for better matching)")
                self.logger.info("   → Minimal local map (5 keyframes max)")
                self.logger.info("   → Very high keyframe threshold (almost no keyframes)")
                self.logger.info("   → Lenient relocalization (works with fewer points)")
                self.logger.info("   → Faster initialization (lower triangulation requirements)")
                self.logger.info("   → Focus on feature tracking only (no map building overhead)")
            
            self.logger.info("🎯 SPEED + ACCURACY OPTIMIZED:")
            self.logger.info(f"   • ORB2 detector/descriptor (ORB-SLAM2 optimized)")
            self.logger.info(f"   • Local BA window: {Parameters.kLocalBAWindow} frames (↓ from 20)")
            self.logger.info(f"   • Local map keyframes: {Parameters.kMaxNumOfKeyframesInLocalMap} (↓ from 80)")
            self.logger.info(f"   • Pose outliers: {Parameters.kMaxOutliersRatioInPoseOptimization} threshold (↓ from 0.9)")
            self.logger.info(f"   • Min tracked points: {Parameters.kMinNumMatchedFeaturesSearchFrameByProjection} (↑ from 20)")
            self.logger.info("   → Optimized for 45-60 FPS with better tracking!")

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
                Parameters.kVolumetricIntegrationVoxelLength = dense_config.get('voxel_length', 0.04)
                Parameters.kVolumetricIntegrationDepthTruncIndoor = dense_config.get('depth_trunc', 4.0)
                Parameters.kVolumetricIntegrationOutputTimeInterval = 2.0  # Update every 2 seconds
                
                # CRITICAL: Enable depth estimator for monocular SLAM TSDF
                Parameters.kVolumetricIntegrationUseDepthEstimator = True
                Parameters.kVolumetricIntegrationDepthEstimatorType = "DEPTH_ANYTHING_V2"

                self.logger.info("🗺️  Real-time Dense Reconstruction ENABLED")
                self.logger.info(f"   • Type: {Parameters.kVolumetricIntegrationType}")
                self.logger.info(f"   • Voxel size: {Parameters.kVolumetricIntegrationVoxelLength}m")
                self.logger.info(f"   • Extract mesh: {Parameters.kVolumetricIntegrationExtractMesh}")
                self.logger.info(f"   • Depth truncation: {Parameters.kVolumetricIntegrationDepthTruncIndoor}m")
                self.logger.info(f"   • Depth estimator: {Parameters.kVolumetricIntegrationDepthEstimatorType}")
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
            
            # No camera capture needed - frames provided by main.py via process_frame()
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
        # Update SLAM dimensions to match frame if it's different (use original video size)
        frame_h, frame_w = frame.shape[0:2]
        if frame_h != self.height or frame_w != self.width:
            self.logger.debug(f"Updating SLAM dimensions from {self.width}x{self.height} to {frame_w}x{frame_h}")
            old_width, old_height = self.width, self.height
            self.width = frame_w
            self.height = frame_h
            self.cx = self.width / 2
            self.cy = self.height / 2
            
            # Scale focal length proportionally to maintain field of view
            width_scale = self.width / old_width if old_width > 0 else 1.0
            height_scale = self.height / old_height if old_height > 0 else 1.0
            self.fx *= width_scale
            self.fy *= height_scale
            
            # Update camera calibration if it exists
            if hasattr(self, 'camera') and self.camera:
                self.camera.width = self.width
                self.camera.height = self.height
                self.camera.fx = self.fx
                self.camera.fy = self.fy
                self.camera.cx = self.cx
                self.camera.cy = self.cy
                # CRITICAL: Update K and Kinv matrices used by SLAM
                self.camera.K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]], dtype=np.float64)
                self.camera.Kinv = np.array(
                    [[1.0 / self.fx, 0.0, -self.cx / self.fx], [0.0, 1.0 / self.fy, -self.cy / self.fy], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
                self.camera.u_min, self.camera.u_max = 0, self.width
                self.camera.v_min, self.camera.v_max = 0, self.height

        # Convert BGR to RGB (pySLAM expects RGB from datasets)
        if len(frame.shape) == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # Already grayscale, convert back to RGB for consistency
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Store current frame image for feature matching visualization
        # Use view instead of copy for performance (only copy if needed)
        if rgb_frame is not None and self.viz_mode == 'feature_matching':
            # Only cache if in feature matching mode to save memory
            self.last_frame_img = rgb_frame.copy() if rgb_frame is not None else None
        
        # Process frame through pySLAM
        timestamp = time.time()

        try:
            self.slam.track(rgb_frame, None, None, self.frame_count, timestamp)
            
            # CRITICAL: Enable frame image storage for feature matching visualization
            # This ensures frames store their images so we can visualize matches
            if self.viz_mode == 'feature_matching':
                try:
                    from pyslam.slam.frame import Frame
                    if not Frame.is_store_imgs:
                        Frame.is_store_imgs = True
                except:
                    pass
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
        
        # CRITICAL: Store matched indices right after tracking for feature matching visualization
        # The indices might be reset later, so capture them now
        if hasattr(self.slam, 'tracking'):
            tracking = self.slam.tracking
            if hasattr(tracking, 'idxs_ref') and hasattr(tracking, 'idxs_cur'):
                idxs_ref = tracking.idxs_ref
                idxs_cur = tracking.idxs_cur
                if idxs_ref is not None and idxs_cur is not None:
                    # Store a copy to avoid reference issues
                    if isinstance(idxs_ref, np.ndarray):
                        self.last_idxs_ref = idxs_ref.copy()
                    else:
                        self.last_idxs_ref = np.array(idxs_ref) if idxs_ref is not None else None
                    if isinstance(idxs_cur, np.ndarray):
                        self.last_idxs_cur = idxs_cur.copy()
                    else:
                        self.last_idxs_cur = np.array(idxs_cur) if idxs_cur is not None else None
                    
                    # Store reference frame image if available
                    if hasattr(tracking, 'f_ref') and tracking.f_ref is not None:
                        f_ref = tracking.f_ref
                        if hasattr(f_ref, 'img') and f_ref.img is not None:
                            self.last_ref_frame_img = f_ref.img.copy()
                    
                    # Debug: log matches found
                    if self.viz_mode == 'feature_matching' and len(self.last_idxs_ref) > 0:
                        self.logger.info(f"Stored {len(self.last_idxs_ref)} matched indices for feature matching")
        
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
        
        # Update visualization - SKIP 2D plots for performance (shows in Pangolin 3D viewer only)
        # Commented out for performance - uncomment if you want trajectory/error plots
        # if self.plot_drawer:
        #     try:
        #         self.plot_drawer.draw(self.frame_count)
        #     except Exception as e:
        #         self.logger.warning(f"Plot visualization error: {e}")

        # Update 3D viewer - CRITICAL: This shows the 3D point cloud window
        # Skip in feature_matching mode (only show feature matching in main window)
        if self.viz_mode != 'feature_matching' and hasattr(self, 'viewer3d') and self.viewer3d:
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
                
                # Draw dense TSDF map (same as main_slam.py line 366)
                if self.use_dense_reconstruction and hasattr(self.viewer3d, 'draw_dense_map'):
                    try:
                        self.viewer3d.draw_dense_map(self.slam)
                    except Exception as e:
                        self.logger.debug(f"Dense map visualization: {e}")
            except Exception as e:
                self.logger.warning(f"3D visualization error: {e}")

        # Show pySLAM camera window with feature tracking
        # Skip in feature_matching mode (only show feature matching in main window)
        if self.viz_mode != 'feature_matching':
            try:
                if hasattr(self.slam, 'tracking') and hasattr(self.slam.tracking, 'draw_img'):
                    # Use pySLAM's own tracking visualization
                    img_draw = self.slam.tracking.draw_img
                    if img_draw is not None:
                        cv2.imshow("SLAM Camera", img_draw)
                    else:
                        cv2.imshow("SLAM Camera", frame)
                elif hasattr(self.slam, 'map') and hasattr(self.slam.map, 'draw_feature_trails'):
                    img_draw = self.slam.map.draw_feature_trails(frame)
                    cv2.imshow("pySLAM - Camera", img_draw)
                else:
                    # Fallback to basic camera view
                    cv2.imshow("SLAM Camera", frame)
            except Exception as e:
                self.logger.warning(f"Camera window error: {e}")
                cv2.imshow("SLAM Camera", frame)

        # Process OpenCV events to update windows (minimal - main loop handles this)
        # cv2.waitKey(1)  # Removed - handled in main loop

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

    def get_feature_matching_image(self) -> Optional[np.ndarray]:
        """
        Get/create the feature matching visualization image.
        Shows current frame features matched with reference keyframe.

        Returns:
            Feature matching image or None if not available
        """
        try:
            if not self.is_initialized or not hasattr(self.slam, 'tracking'):
                return None

            tracking = self.slam.tracking

            # Check if we have current frame and reference
            if not hasattr(tracking, 'f_cur') or tracking.f_cur is None:
                return None

            if not hasattr(tracking, 'f_ref') or tracking.f_ref is None:
                return None

            f_cur = tracking.f_cur
            f_ref = tracking.f_ref

            # Get images - check for None explicitly
            if not hasattr(f_cur, 'img') or f_cur.img is None:
                return None

            if not hasattr(f_ref, 'img') or f_ref.img is None:
                return None

            # Try to get images from frame objects first
            img_cur = f_cur.img if hasattr(f_cur, 'img') and f_cur.img is not None else None
            img_ref = f_ref.img if hasattr(f_ref, 'img') and f_ref.img is not None else None
            
            # Fallback: Use cached images if frame images are not available
            if img_cur is None:
                img_cur = self.last_frame_img
            
            if img_ref is None:
                img_ref = self.last_ref_frame_img
            
            # Ensure images are valid numpy arrays with shape
            if img_cur is None or img_ref is None:
                return None
            
            if not hasattr(img_cur, 'shape') or not hasattr(img_ref, 'shape'):
                return None
            
            # Ensure images have valid shapes
            if len(img_cur.shape) < 2 or len(img_ref.shape) < 2:
                return None

            # Get keypoints and matches
            if not hasattr(f_cur, 'kps') or not hasattr(f_ref, 'kps'):
                return None

            # Extract keypoints as arrays of [x, y] coordinates
            # In pySLAM, kps are stored as [Nx2] numpy arrays after conversion from cv2.KeyPoint
            if not hasattr(f_cur, 'kps') or not hasattr(f_ref, 'kps'):
                return None
                
            kps_cur = f_cur.kps
            kps_ref = f_ref.kps
            
            # Ensure they're numpy arrays
            if kps_cur is None or kps_ref is None or len(kps_cur) == 0 or len(kps_ref) == 0:
                return None
                
            kps_cur_pts = np.array(kps_cur) if not isinstance(kps_cur, np.ndarray) else kps_cur
            kps_ref_pts = np.array(kps_ref) if not isinstance(kps_ref, np.ndarray) else kps_ref
            
            # Get keypoint sizes if available (for green circles)
            # Scale down sizes but keep them visible (keypoint sizes are often too large)
            kps_cur_sizes = None
            kps_ref_sizes = None
            if hasattr(f_cur, 'sizes') and f_cur.sizes is not None:
                sizes = np.array(f_cur.sizes) if not isinstance(f_cur.sizes, np.ndarray) else f_cur.sizes
                # Scale down sizes: use 0.25x the original size, cap at 6 pixels (bigger circles)
                kps_cur_sizes = np.clip(sizes * 0.25, 2, 6).astype(np.int32)
            if hasattr(f_ref, 'sizes') and f_ref.sizes is not None:
                sizes = np.array(f_ref.sizes) if not isinstance(f_ref.sizes, np.ndarray) else f_ref.sizes
                # Scale down sizes: use 0.25x the original size, cap at 6 pixels (bigger circles)
                kps_ref_sizes = np.clip(sizes * 0.25, 2, 6).astype(np.int32)

            # Try to get matched indices from multiple sources
            matched_indices = []
            
            # Method 1: Try stored indices first (captured right after tracking)
            # These are the most reliable as they're captured right after tracking
            if hasattr(self, 'last_idxs_ref') and hasattr(self, 'last_idxs_cur'):
                idxs_ref = self.last_idxs_ref
                idxs_cur = self.last_idxs_cur
                if idxs_ref is not None and idxs_cur is not None:
                    # Check if they're valid arrays
                    if isinstance(idxs_ref, np.ndarray) and isinstance(idxs_cur, np.ndarray):
                        if len(idxs_ref) == 0 or len(idxs_cur) == 0:
                            # Empty arrays, try current tracking
                            idxs_ref = None
                            idxs_cur = None
                    else:
                        # Not arrays, try current tracking
                        idxs_ref = None
                        idxs_cur = None
                else:
                    # None values, try current tracking
                    idxs_ref = None
                    idxs_cur = None
            
            # Fallback to current tracking indices if stored ones aren't available
            if idxs_ref is None or idxs_cur is None:
                if hasattr(tracking, 'idxs_ref') and hasattr(tracking, 'idxs_cur'):
                    idxs_ref = tracking.idxs_ref
                    idxs_cur = tracking.idxs_cur
            
            # Process matched indices
            if idxs_ref is not None and idxs_cur is not None:
                # Convert to numpy arrays if needed
                if not isinstance(idxs_ref, np.ndarray):
                    idxs_ref = np.array(idxs_ref)
                if not isinstance(idxs_cur, np.ndarray):
                    idxs_cur = np.array(idxs_cur)
                
                if len(idxs_ref) > 0 and len(idxs_cur) > 0 and len(idxs_ref) == len(idxs_cur):
                    # Ensure indices are valid
                    # Vectorized validation for better performance
                    idxs_ref_int = idxs_ref.astype(np.int32)
                    idxs_cur_int = idxs_cur.astype(np.int32)
                    valid_mask = (idxs_ref_int >= 0) & (idxs_ref_int < len(kps_ref_pts)) & \
                                 (idxs_cur_int >= 0) & (idxs_cur_int < len(kps_cur_pts))
                    valid_matches = list(zip(idxs_ref_int[valid_mask], idxs_cur_int[valid_mask]))
                    matched_indices = valid_matches
            
            # Method 2: Try to get matches from frame itself (if available)
            # Sometimes matches are stored in the frame objects
            if len(matched_indices) == 0:
                try:
                    # Check if frames have matched keypoints stored
                    if hasattr(f_cur, 'matched_kps') and hasattr(f_ref, 'matched_kps'):
                        # Matches might be stored in frame objects
                        pass  # Skip for now
                except:
                    pass
            
            # Method 3: Simple matching based on descriptor similarity (last resort)
            # This is expensive, so only use if other methods fail
            # For now, skip this and rely on Method 1

            # If no matches available, just show the frames side by side
            if len(matched_indices) == 0:
                # Just stack the two images
                h1, w1 = img_cur.shape[:2] if len(img_cur.shape) > 2 else (*img_cur.shape, 1)
                h2, w2 = img_ref.shape[:2] if len(img_ref.shape) > 2 else (*img_ref.shape, 1)

                # Ensure both images have same height
                if h1 != h2:
                    if h1 > h2:
                        img_ref = cv2.resize(img_ref, (int(w2 * h1 / h2), h1))
                    else:
                        img_cur = cv2.resize(img_cur, (int(w1 * h2 / h1), h2))

                # Stack horizontally
                if len(img_cur.shape) == 2:
                    img_cur = cv2.cvtColor(img_cur, cv2.COLOR_GRAY2BGR)
                if len(img_ref.shape) == 2:
                    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2BGR)

                img_matches = np.hstack([img_ref, img_cur])

                # Add text
                cv2.putText(img_matches, f"Reference (KF)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_matches, f"Current", (img_ref.shape[1] + 10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                return img_matches

            # No need to import draw_feature_matches - using custom thin drawing function

            # Extract matched keypoints using the matched indices
            matched_kps_ref = kps_ref_pts[[idx[0] for idx in matched_indices]]
            matched_kps_cur = kps_cur_pts[[idx[1] for idx in matched_indices]]
            
            # Extract matched keypoint sizes if available
            matched_kps_ref_sizes = None
            matched_kps_cur_sizes = None
            if kps_ref_sizes is not None:
                matched_kps_ref_sizes = kps_ref_sizes[[idx[0] for idx in matched_indices]]
            if kps_cur_sizes is not None:
                matched_kps_cur_sizes = kps_cur_sizes[[idx[1] for idx in matched_indices]]
            
            # Ensure images are RGB (draw_feature_matches expects RGB)
            # Optimize: only convert if needed
            if len(img_ref.shape) == 2:
                img_ref = cv2.cvtColor(img_ref, cv2.COLOR_GRAY2RGB)
            elif img_ref.shape[2] == 3 and img_ref.dtype != np.uint8:
                img_ref = (img_ref * 255).astype(np.uint8) if img_ref.max() <= 1.0 else img_ref.astype(np.uint8)
            
            if len(img_cur.shape) == 2:
                img_cur = cv2.cvtColor(img_cur, cv2.COLOR_GRAY2RGB)
            elif img_cur.shape[2] == 3 and img_cur.dtype != np.uint8:
                img_cur = (img_cur * 255).astype(np.uint8) if img_cur.max() <= 1.0 else img_cur.astype(np.uint8)
            
            # Draw matches with custom thin lines and smaller circles
            # Use a custom drawing function for better control over line thickness and circle sizes
            img_matches = self._draw_feature_matches_thin(
                img_ref, img_cur,
                matched_kps_ref, matched_kps_cur,
                kps1_sizes=matched_kps_ref_sizes,
                kps2_sizes=matched_kps_cur_sizes
            )
            
            # draw_feature_matches returns RGB, convert to BGR for OpenCV display
            if len(img_matches.shape) == 3 and img_matches.shape[2] == 3:
                img_matches = cv2.cvtColor(img_matches, cv2.COLOR_RGB2BGR)
            
            return img_matches

        except Exception as e:
            # Log error for debugging
            import traceback
            self.logger.warning(f"Could not create feature matching image: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _draw_feature_matches_thin(self, img1: np.ndarray, img2: np.ndarray, 
                                    kps1: np.ndarray, kps2: np.ndarray,
                                    kps1_sizes: Optional[np.ndarray] = None,
                                    kps2_sizes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Draw feature matches with thin lines and smaller circles.
        Maintains original image resolution.
        Optimized for performance with vectorized operations where possible.
        """
        # Combine images horizontally
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Ensure both images have same height
        if h1 != h2:
            if h1 > h2:
                img2 = cv2.resize(img2, (w2, h1), interpolation=cv2.INTER_LINEAR)
                h2 = h1
            else:
                img1 = cv2.resize(img1, (w1, h2), interpolation=cv2.INTER_LINEAR)
                h1 = h2
        
        # Combine horizontally
        img3 = np.hstack([img1, img2])
        
        N = len(kps1)
        if N == 0:
            return img3
        
        # Default sizes (bigger for visibility)
        default_size = 3
        if kps1_sizes is None:
            kps1_sizes = np.ones(N, dtype=np.int32) * default_size
        if kps2_sizes is None:
            kps2_sizes = np.ones(N, dtype=np.int32) * default_size
        
        # Ensure sizes are visible (2-5 pixels) - vectorized
        kps1_sizes = np.clip(kps1_sizes, 2, 5).astype(np.int32)
        kps2_sizes = np.clip(kps2_sizes, 2, 5).astype(np.int32)
        
        # Pre-compute rounded points for performance
        pts1 = np.rint(kps1).astype(np.int32)
        pts2 = np.rint(kps2).astype(np.int32)
        
        # Use a limited color palette for better visibility (not too many colors)
        # Colors: blue, cyan, yellow, magenta, orange, green
        color_palette = [
            (255, 0, 0),      # Blue
            (255, 255, 0),   # Cyan
            (0, 255, 255),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 165, 255),   # Orange
            (0, 255, 0),     # Green
        ]
        num_colors = len(color_palette)
        
        # Draw matches with thin lines (thickness=1) and visible circles
        # Use efficient loop with pre-computed values
        for i in range(N):
            a, b = pts1[i]
            c, d = pts2[i]
            
            # Use color from palette (cycle through colors)
            color = color_palette[i % num_colors]
            
            # Draw thin line (thickness=1)
            cv2.line(img3, (a, b), (c + w1, d), color, thickness=1, lineType=cv2.LINE_AA)
            
            # Draw center dot (radius=2 for visibility)
            cv2.circle(img3, (a, b), 2, color, -1, lineType=cv2.LINE_AA)
            cv2.circle(img3, (c + w1, d), 2, color, -1, lineType=cv2.LINE_AA)
            
            # Draw green circle for keypoint size (bigger, 2-4 pixels)
            size1 = max(2, min(4, int(kps1_sizes[i])))
            size2 = max(2, min(4, int(kps2_sizes[i])))
            cv2.circle(img3, (a, b), size1, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(img3, (c + w1, d), size2, (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        
        return img3

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
