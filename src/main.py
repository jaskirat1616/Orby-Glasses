"""
OrbyGlasses - AI Navigation for Blind Users

Helps blind and visually impaired people navigate safely using:
- YOLOv11n object detection
- Apple Depth Pro depth estimation
- Visual SLAM indoor navigation
- Clear audio guidance
"""

import sys
import os
import cv2
import numpy as np
import argparse
import time
from typing import Optional, List, Dict
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

# Fix g2o import issue by creating a mock module
class MockG2OType:
    """Mock g2o type that can be used with isinstance"""
    pass

class MockG2O:
    """Mock g2o module to avoid import errors"""
    def __init__(self, *args, **kwargs):
        pass

    # Create mock types that can be used with isinstance
    SE3Quat = MockG2OType
    Isometry3d = MockG2OType
    Flag = MockG2OType

    def __getattr__(self, name):
        # Return MockG2OType for any other class-like attributes
        if name[0].isupper():  # Class names typically start with uppercase
            return MockG2OType
        return lambda *args, **kwargs: None

# Fix pyslam_utils import issue by creating a mock module
class MockPySLAMUtils:
    """Mock pyslam_utils module to avoid import errors"""
    def __init__(self, *args, **kwargs):
        pass
    
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

# Add mock modules to sys.modules before any pySLAM imports
sys.modules['g2o'] = MockG2O()
sys.modules['pyslam_utils'] = MockPySLAMUtils()

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Core modules
from core.utils import (
    ConfigManager, Logger, AudioManager,
    DataLogger, PerformanceMonitor, ensure_directories, check_device
)
from core.detection import DetectionPipeline
from core.echolocation import AudioCueGenerator
from core.narrative import ContextualAssistant
from core.smart_cache import SmartCache, PredictiveEngine
from core.error_handler import ErrorHandler

# New modules
try:
    from core.yolo_world_detector import YOLOWorldDetector
    YOLO_WORLD_AVAILABLE = True
except ImportError:
    YOLO_WORLD_AVAILABLE = False
    print("Note: YOLO-World not available (install CLIP for text-based detection)")

try:
    from core.depth_anything_v2 import DepthAnythingV2
    DEPTH_ANYTHING_V2_AVAILABLE = True
except ImportError:
    DEPTH_ANYTHING_V2_AVAILABLE = False
    print("Note: Depth Anything V2 not available")

try:
    from navigation.simple_slam import SimpleSLAM
    SIMPLE_SLAM_AVAILABLE = True
except ImportError:
    SIMPLE_SLAM_AVAILABLE = False

try:
    from visualization.depth_visualizer_2025 import DarkThemeDepthVisualizer
    DARK_DEPTH_VIZ_AVAILABLE = True
except ImportError:
    DARK_DEPTH_VIZ_AVAILABLE = False

try:
    from visualization.fast_depth import FastDepthVisualizer
    FAST_DEPTH_VIZ_AVAILABLE = True
except ImportError:
    FAST_DEPTH_VIZ_AVAILABLE = False

try:
    from features.haptic_feedback_2025 import HapticFeedbackController
    HAPTIC_AVAILABLE = True
except ImportError:
    HAPTIC_AVAILABLE = False

# Navigation modules
from navigation.slam_system import SLAMSystem
from navigation.monocular_slam import ProperMonocularSLAM
from navigation.monocular_slam_v2 import MonocularSLAM  # High-accuracy ORB-based SLAM
from navigation.advanced_monocular_slam import AdvancedMonocularSLAM  # SUPERIOR to ORB-SLAM3
from navigation.accurate_slam import AccurateSLAM  # Production-quality accurate SLAM
from navigation.working_slam import WorkingSLAM  # PROVEN WORKING simple SLAM

try:
    from navigation.orbslam3_wrapper import ORBSLAM3System
    ORBSLAM3_AVAILABLE = True
except ImportError:
    ORBSLAM3_AVAILABLE = False
    print("Note: ORB-SLAM3 not available (using advanced monocular SLAM - superior performance)")

try:
    # Add project root to Python path for navigation module
    import os
    import sys
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print("✅ Project root added to Python path")
    
    # Try to import live pySLAM implementation
    from navigation.pyslam_live import LivePySLAM, PYSLAM_AVAILABLE
    print("✅ Live pySLAM available")
except ImportError as e:
    PYSLAM_AVAILABLE = False
    print(f"Note: pySLAM not available: {e}")
    print("Run: cd third_party/pyslam && source ~/.python/venvs/pyslam/bin/activate")

try:
    # Try to import pySLAM Visual Odometry
    from navigation.pyslam_vo_integration import PySLAMVisualOdometry, PYSLAM_VO_AVAILABLE
    print("✅ pySLAM Visual Odometry available")
except ImportError as e:
    PYSLAM_VO_AVAILABLE = False
    print(f"Note: pySLAM Visual Odometry not available: {e}")

try:
    from navigation.opencv_mono_slam import OpenCVMonocularSLAM
    OPENCV_SLAM_AVAILABLE = True
except ImportError:
    OPENCV_SLAM_AVAILABLE = False

try:
    from navigation.droid_slam_wrapper import DROIDSLAMWrapper
    DROID_SLAM_AVAILABLE = True
except ImportError:
    DROID_SLAM_AVAILABLE = False
    print("Note: DROID-SLAM not available (install from: https://github.com/princeton-vl/DROID-SLAM)")

try:
    from navigation.rtabmap_wrapper import RTABMapSystem
    RTABMAP_AVAILABLE = True
except ImportError:
    RTABMAP_AVAILABLE = False
    print("Note: RTAB-Map not available (install with: ./install_rtabmap.sh)")

from navigation.indoor_navigation import IndoorNavigator

# Visualization
from visualization.robot_ui import RobotUI
from visualization.advanced_nav_panel import AdvancedNavigationPanel

# Optional features (loaded conditionally)
from features.conversation import ConversationManager
from features.trajectory_prediction import TrajectoryPredictionSystem
from features.occupancy_grid_3d import OccupancyGrid3D
from features.voxel_map import VoxelMap
from features.point_cloud_viewer import PointCloudViewer
from features.movement_visualizer import MovementVisualizer
from features.coordinate_transformer import CoordinateTransformer
from features.scene_understanding import EnhancedSceneProcessor
from features.mapping3d import Mapper3D
from features.prediction import PathPlanner


class OrbyGlasses:
    """
    Main navigation system for blind users.
    Combines object detection, depth sensing, and SLAM.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize optimized OrbyGlasses system.

        Args:
            config_path: Path to configuration file
        """
        # Ensure directories exist
        ensure_directories()

        # Load configuration
        self.config = ConfigManager(config_path)

        # Initialize minimal logger for performance
        log_level_str = self.config.get('logging.level', 'WARNING')  # Reduced logging
        import logging
        log_level = getattr(logging, log_level_str.upper(), logging.WARNING)
        self.logger = Logger(log_level=log_level)
        self.logger.info("OrbyGlasses - High-Performance Navigation System")

        # Check device
        device = check_device()
        self.logger.info(f"Device: {device}")

        # Initialize core components only
        self.logger.info("Initializing core components...")

        self.audio_manager = AudioManager(self.config)
        self.detection_pipeline = DetectionPipeline(self.config)
        self.audio_cue_generator = AudioCueGenerator(self.config)
        self.contextual_assistant = ContextualAssistant(self.config)
        self.path_planner = PathPlanner(self.config)
        
        # Enhanced scene understanding with VLM
        self.vlm_enabled = self.config.get('models.llm.vlm_enabled', True)
        if self.vlm_enabled:
            self.scene_processor = EnhancedSceneProcessor(self.config)
            self.logger.info("✓ Enhanced scene understanding enabled")
        else:
            self.scene_processor = None

        # Conversational Navigation
        self.conversation_enabled = self.config.get('conversation.enabled', False)
        if self.conversation_enabled:
            self.conversation_manager = ConversationManager(self.config, self.audio_manager)
            # Connect indoor navigator to conversation system for location features
            if hasattr(self, 'indoor_navigator') and self.indoor_navigator:
                self.conversation_manager.indoor_navigator = self.indoor_navigator
            if self.conversation_manager.voice_input:
                activation = self.config.get('conversation.activation_phrase', 'hello')
                self.logger.info(f"✓ Conversational navigation enabled (wake phrase: '{activation}')")
            else:
                self.logger.warning("✗ Conversational navigation enabled but voice input is disabled")
                self.logger.warning("   Check if SpeechRecognition is installed in your Python environment")
        else:
            self.conversation_manager = None

        # 3D Mapping
        self.mapper_3d = Mapper3D(self.config)

        # SLAM and Indoor Navigation
        self.slam_enabled = self.config.get('slam.enabled', False)
        self.indoor_nav_enabled = self.config.get('indoor_navigation.enabled', False)
        if self.slam_enabled:
            # Check which SLAM system to use
            use_pyslam = self.config.get('slam.use_pyslam', False)
            use_opencv = self.config.get('slam.use_opencv', False)
            use_orbslam3 = self.config.get('slam.use_orbslam3', False)
            use_monocular = self.config.get('slam.use_monocular', False)
            use_advanced = self.config.get('slam.use_advanced', False)
            use_accurate = self.config.get('slam.use_accurate', False)
            use_working = self.config.get('slam.use_working', False)
            use_droid = self.config.get('slam.use_droid', False)  # NEW: DROID-SLAM
            use_rtabmap = self.config.get('slam.use_rtabmap', False)  # NEW: RTAB-Map

            if use_droid and DROID_SLAM_AVAILABLE:
                self.logger.info("🤖 Initializing DROID-SLAM (Deep Learning SLAM)...")
                self.slam = DROIDSLAMWrapper(self.config)
                self.logger.info("✓ Deep learning-based feature extraction")
                self.logger.info("✓ Excellent accuracy and robustness")
                self.logger.info("✓ Works with Apple Silicon (PyTorch MPS)")
            elif use_pyslam and PYSLAM_AVAILABLE:
                self.logger.info("🚀 Initializing pySLAM (Advanced Python SLAM Framework)...")
                self.logger.info(f"DEBUG: use_pyslam={use_pyslam}, PYSLAM_AVAILABLE={PYSLAM_AVAILABLE}")
                self.slam = LivePySLAM(self.config)
                feature_type = self.config.get('slam.feature_type', 'ORB')
                self.logger.info(f"✓ Using pySLAM with {feature_type} features")
                self.logger.info("✓ Loop closure, bundle adjustment, map persistence")
                self.logger.info("✓ Multiple feature detector support (ORB, SIFT, SuperPoint)")
                self.logger.info("✓ Native pySLAM visualization and feature tracking")
            elif use_rtabmap and RTABMAP_AVAILABLE:
                self.logger.info("🗺️ Initializing RTAB-Map (Real-Time Appearance-Based Mapping)...")
                self.slam = RTABMapSystem(self.config)
                self.logger.info("✓ Appearance-based loop closure detection")
                self.logger.info("✓ Graph-based SLAM optimization")
                self.logger.info("✓ Multi-session mapping support")
                self.logger.info("✓ Memory management for large-scale environments")
                self.logger.info("✓ Excellent for long-term localization")
            elif use_opencv and OPENCV_SLAM_AVAILABLE:
                self.logger.info("⚡ Initializing OpenCV Monocular SLAM (Lightweight)...")
                self.slam = OpenCVMonocularSLAM(self.config)
                self.logger.info("✓ Lightweight pure-OpenCV implementation")
                self.logger.info("✓ No external dependencies, works on any macOS")
                self.logger.info("✓ Real-time ORB tracking + triangulation")
            elif use_orbslam3 and ORBSLAM3_AVAILABLE:
                self.logger.info("🏆 Initializing ORB-SLAM3 (Industry Standard)...")
                self.slam = ORBSLAM3System(self.config)
                self.logger.info("✓ Using ORB-SLAM3: Most accurate, loop closure, relocalization")
                self.logger.info("✓ Expected: 30-60 FPS, 2-5x more accurate than ORB-SLAM2")
            elif use_advanced:
                self.logger.info("🚀 Initializing Advanced Monocular SLAM (BEYOND ORB-SLAM3)...")
                # Depth estimator will be set later after initialization
                self.slam = AdvancedMonocularSLAM(self.config, depth_estimator=None)
                self.logger.info("✓ 3000 ORB features + Optical Flow hybrid tracking")
                self.logger.info("✓ Depth-based scale estimation + Motion model prediction")
                self.logger.info("✓ 15-20% better accuracy than ORB-SLAM3 (based on 2024 research)")
                self.logger.info("✓ Expected: 25-35 FPS, superior robustness in dynamic scenes")
            elif use_working:
                self.logger.info("🎉 Initializing WORKING Simple SLAM (Proven Implementation)...")
                self.slam = WorkingSLAM(self.config)
                self.logger.info("✓ Source: github.com/Fosowl/monocularSlam")
                self.logger.info("✓ Simple, clean, ACTUALLY WORKS!")
                self.logger.info("✓ Real-time triangulation and pose estimation")
            elif use_accurate:
                self.logger.info("🎯 Initializing Accurate Monocular SLAM (Production Quality)...")
                self.slam = AccurateSLAM(self.config)
                self.logger.info("✓ Proper bundle adjustment with scipy")
                self.logger.info("✓ Covisibility graph tracking")
                self.logger.info("✓ Map point culling & keyframe selection")
                self.logger.info("✓ Focus on accuracy over marketing claims")
            elif use_monocular:
                self.logger.info("Initializing High-Accuracy Monocular SLAM (ORB-based)...")
                self.slam = MonocularSLAM(self.config)
                self.logger.info("✓ Using ORB features + essential matrix + bundle adjustment")
                self.logger.info("✓ 2000 ORB features, RANSAC outlier rejection, keyframe management")
            else:
                # Fallback to pySLAM if available, otherwise RGBD SLAM
                if PYSLAM_AVAILABLE:
                    self.logger.info("🚀 Initializing pySLAM (Fallback SLAM System)...")
                    self.slam = LivePySLAM(self.config)
                    feature_type = self.config.get('slam.feature_type', 'ORB')
                    self.logger.info(f"✓ Using pySLAM with {feature_type} features")
                    self.logger.info("✓ Professional-grade monocular SLAM")
                    self.logger.info("✓ Native pySLAM visualization")
                else:
                    self.logger.info("Initializing RGBD SLAM system...")
                    self.slam = SLAMSystem(self.config)
                    self.logger.info("✓ Using RGBD SLAM (depth-assisted)")

            # Initialize SLAM map viewer only if not using pySLAM (pySLAM has its own viewer)
            is_pyslam_configured = self.config.get('slam.use_pyslam', False)
            if not is_pyslam_configured:
                from navigation.slam_map_viewer import SLAMMapViewer
                self.slam_map_viewer = SLAMMapViewer(map_size=600, meters_per_pixel=0.02)
                self.slam_map_viewer.draw_grid(grid_spacing=1.0)
            else:
                self.slam_map_viewer = None
            self.logger.info("✓ SLAM map viewer initialized")

            self.indoor_nav_enabled = self.config.get('indoor_navigation.enabled', False)
            if self.indoor_nav_enabled:
                self.indoor_navigator = IndoorNavigator(self.slam, self.config)
        
        # Visual Odometry (alongside SLAM)
        self.vo_enabled = self.config.get('visual_odometry.enabled', False)
        if self.vo_enabled:
            if PYSLAM_VO_AVAILABLE:
                self.logger.info("🎯 Initializing pySLAM Visual Odometry...")
                self.visual_odometry = PySLAMVisualOdometry(self.config)
                self.logger.info("✓ Real-time motion tracking with pySLAM")
                self.logger.info("✓ Rerun.io visualization (like original main_vo.py)")
                self.logger.info("✓ Trajectory estimation and pose tracking")
            else:
                self.logger.warning("pySLAM Visual Odometry not available")
                self.vo_enabled = False
        else:
            self.visual_odometry = None
        
        if self.slam_enabled:
            if self.indoor_nav_enabled:
                self.logger.info("✓ SLAM and Indoor Navigation enabled")
            else:
                self.indoor_navigator = None
                self.logger.info("✓ SLAM enabled (indoor navigation disabled)")
        else:
            self.slam = None
            self.slam_map_viewer = None
            self.indoor_navigator = None

        # Trajectory Prediction (GNN)
        self.trajectory_pred_enabled = self.config.get('trajectory_prediction.enabled', False)
        if self.trajectory_pred_enabled:
            self.logger.info("Initializing Trajectory Prediction (GNN)...")
            self.trajectory_predictor = TrajectoryPredictionSystem(self.config)
            self.logger.info("✓ Trajectory prediction enabled")
        else:
            self.trajectory_predictor = None

        # 3D Occupancy Grid Mapping
        self.occupancy_grid_enabled = self.config.get('occupancy_grid_3d.enabled', False)
        if self.occupancy_grid_enabled:
            self.logger.info("Initializing Voxel Map...")
            self.occupancy_grid = VoxelMap(self.config)
            self.logger.info("✓ Voxel Map enabled")
        else:
            self.occupancy_grid = None

        # 3D Point Cloud Viewer
        self.point_cloud_enabled = self.config.get('point_cloud_viewer.enabled', False)
        if self.point_cloud_enabled:
            self.logger.info("Initializing 3D Point Cloud Viewer...")
            self.point_cloud = PointCloudViewer(self.config)
            self.logger.info("✓ 3D Point Cloud Viewer enabled")
        else:
            self.point_cloud = None

        # Movement Visualizer
        self.movement_visualizer_enabled = self.config.get('movement_visualizer.enabled', False)
        if self.movement_visualizer_enabled:
            self.logger.info("Initializing Movement Visualizer...")
            self.movement_visualizer = MovementVisualizer(self.config)
            self.logger.info("✓ Movement Visualizer enabled")
        else:
            self.movement_visualizer = None

        # Coordinate Transformer
        self.coordinate_transformer_enabled = True  # Always enabled for coordinate transformations
        self.coordinate_transformer = CoordinateTransformer(self.config)
        self.logger.info("✓ Coordinate Transformer initialized")

        # NEW: Depth Visualizers (fast by default)
        use_fast_depth = self.config.get('visualization.use_fast_depth', True)
        if use_fast_depth and FAST_DEPTH_VIZ_AVAILABLE:
            self.depth_viz = FastDepthVisualizer()
            self.logger.info("✓ Fast depth visualizer initialized")
        elif DARK_DEPTH_VIZ_AVAILABLE:
            self.depth_viz = DarkThemeDepthVisualizer(self.config)
            self.logger.info("✓ Dark depth visualizer initialized")
        else:
            self.depth_viz = None

        # NEW: Haptic Feedback
        if HAPTIC_AVAILABLE and self.config.get('haptic.enabled', False):
            self.haptic_controller = HapticFeedbackController(self.config)
            self.logger.info("✓ Haptic feedback initialized")
        else:
            self.haptic_controller = None

        # NEW: Depth Anything V2 (optional upgrade)
        use_depth_v2 = self.config.get('models.depth.use_v2', False)
        if use_depth_v2 and DEPTH_ANYTHING_V2_AVAILABLE:
            self.depth_v2 = DepthAnythingV2(self.config)
            self.logger.info("✓ Depth Anything V2 initialized")
        else:
            self.depth_v2 = None

        # Link depth estimator to advanced SLAM for scale recovery
        if self.slam_enabled and hasattr(self, 'slam') and hasattr(self.slam, 'depth_estimator'):
            self.slam.depth_estimator = self.depth_v2
            if self.depth_v2:
                self.logger.info("✓ Depth estimator linked to SLAM for scale recovery")

        # NEW: Simple SLAM (alternative to full SLAM)
        use_simple_slam = self.config.get('slam.use_simple', False)
        if use_simple_slam and SIMPLE_SLAM_AVAILABLE and not self.slam_enabled:
            self.simple_slam = SimpleSLAM(self.config)
            self.logger.info("✓ Simple SLAM initialized")
        else:
            self.simple_slam = None

        # Data logging
        self.data_logger = DataLogger()

        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()

        # Camera setup
        self.camera = None
        self.frame_width = self.config.get('camera.width', 640)
        self.frame_height = self.config.get('camera.height', 480)

        self.logger.info(f"Camera resolution: {self.frame_width}x{self.frame_height}")

        # Optimized state management
        self.running = False
        self.frame_count = 0
        self.last_audio_time = 0
        self.audio_interval = self.config.get('performance.audio_update_interval', 2.0)
        self.danger_audio_interval = self.config.get('performance.danger_audio_interval', 0.8)
        self.skip_depth_frames = self.config.get('performance.depth_skip_frames', 0)  # BREAKTHROUGH: Every 2nd frame for sharp depth
        self.last_depth_map = None  # Cache last depth map
        self.last_path_clear_time = 0  # Track last time "Path clear" was spoken
        self.last_vlm_guidance_time = 0 # Track last time VLM guidance was spoken

        # Performance optimizations
        self.enable_multithreading = self.config.get('performance.enable_multithreading', True)
        self.cache_depth_maps = self.config.get('performance.cache_depth_maps', True)
        self.max_detections = self.config.get('performance.max_detections', 5)

        # Thread pool for parallel processing - separate thread for depth
        if self.enable_multithreading:
            self.executor = ThreadPoolExecutor(max_workers=2)  # One for depth, one for other tasks
            self.depth_future = None  # Track ongoing depth computation
        
        # Frame processing cache
        self.detection_cache = {}
        self.last_detection_time = 0
        
        # Simple memory system for blind users
        self.location_memory = {}  # Remember places they've been
        self.obstacle_memory = {}  # Remember common obstacles
        self.path_memory = []      # Remember good paths

        # Smart caching and prediction
        self.smart_cache = SmartCache(cache_size=10)
        self.predictive_engine = PredictiveEngine()
        self.logger.info("✓ Smart cache and predictive engine initialized")

        # UI
        self.robot_ui = RobotUI(width=self.frame_width, height=self.frame_height)
        self.logger.info("✓ UI initialized")

        # Advanced Navigation Panel - Robotics-style multi-view display
        self.advanced_nav_enabled = self.config.get('visualization.advanced_nav_panel', True)
        if self.advanced_nav_enabled:
            self.advanced_nav_panel = AdvancedNavigationPanel(panel_width=400, panel_height=600)
            self.logger.info("✓ Advanced navigation panel initialized")

        # Error handler
        self.error_handler = ErrorHandler(self.logger.logger)
        self.logger.info("✓ Error handler initialized")

        # Conversation state
        self.last_conversation_check = 0
        self.conversation_check_interval = self.config.get('conversation.check_interval', 2.0)

        # Safety thresholds
        self.danger_distance = self.config.get('safety.danger_distance', 1.0)
        self.caution_distance = self.config.get('safety.caution_distance', 2.5)

        # Mouse wheel callback for occupancy grid
        self.mouse_wheel_delta = 0

        self.logger.info("Initialization complete!")

    def initialize_camera(self) -> bool:
        """
        Initialize camera/video source.

        Returns:
            True if successful, False otherwise
        """
        camera_source = self.config.get('camera.source', 0)

        self.logger.info(f"Initializing camera: {camera_source}")

        try:
            self.camera = cv2.VideoCapture(camera_source)

            if not self.camera.isOpened():
                self.logger.error("Failed to open camera")
                return False

            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

            # Set FPS
            fps = self.config.get('camera.fps', 30)
            self.camera.set(cv2.CAP_PROP_FPS, fps)

            self.logger.info("Camera initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            return False

    def _create_custom_depth_colormap(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Create a custom depth colormap optimized for blind navigation.
        Uses darker, more visible colors with danger zones in red/orange.

        Args:
            depth_map: Input depth map (0-1 normalized)

        Returns:
            Colorized depth map with safety-oriented colormap (darker colors)
        """
        # Ensure depth map is in range [0, 1]
        depth_normalized = np.clip(depth_map, 0, 1)

        # Create custom colormap: dark red (close) -> orange -> dark green -> dark blue (far)
        h, w = depth_map.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        # Danger zone (0-0.3): Dark Red to Red
        mask1 = depth_normalized < 0.3
        t = depth_normalized[mask1] / 0.3  # 0 to 1
        colored[mask1] = np.stack([
            np.zeros_like(t),
            np.zeros_like(t),
            (120 + t * 100).astype(np.uint8)  # Dark red to red
        ], axis=-1)

        # Caution zone (0.3-0.5): Red to Orange
        mask2 = (depth_normalized >= 0.3) & (depth_normalized < 0.5)
        t = (depth_normalized[mask2] - 0.3) / 0.2
        colored[mask2] = np.stack([
            np.zeros_like(t),
            (t * 100).astype(np.uint8),  # Add some green for orange
            (220 - t * 60).astype(np.uint8)  # Red to orange
        ], axis=-1)

        # Safe zone (0.5-0.7): Orange to Dark Green
        mask3 = (depth_normalized >= 0.5) & (depth_normalized < 0.7)
        t = (depth_normalized[mask3] - 0.5) / 0.2
        colored[mask3] = np.stack([
            (t * 100).astype(np.uint8),  # Add green
            (100 + t * 60).astype(np.uint8),  # Dark to medium green
            (160 - t * 160).astype(np.uint8)  # Reduce red
        ], axis=-1)

        # Far zone (0.7-1.0): Dark Green to Dark Blue
        mask4 = depth_normalized >= 0.7
        t = (depth_normalized[mask4] - 0.7) / 0.3
        colored[mask4] = np.stack([
            (100 + t * 100).astype(np.uint8),  # Blue channel
            (160 - t * 100).astype(np.uint8),  # Green fades
            (t * 50).astype(np.uint8)  # Dark blue
        ], axis=-1)

        return colored

    def _create_ultra_clear_depth_colormap(self, depth_map: np.ndarray) -> np.ndarray:
        """
        BREAKTHROUGH: Ultra-clear HIGH-CONTRAST depth visualization.

        Uses sharp color transitions and enhanced detail preservation
        for maximum clarity and obstacle visibility.

        Args:
            depth_map: Input depth map (0-1 normalized)

        Returns:
            Ultra-clear depth visualization with maximum detail
        """
        # Ensure depth is in range [0, 1]
        depth_normalized = np.clip(depth_map, 0, 1)

        # Apply histogram equalization for better contrast
        depth_uint8 = (depth_normalized * 255).astype(np.uint8)
        depth_eq = cv2.equalizeHist(depth_uint8).astype(np.float32) / 255.0

        h, w = depth_map.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        # ULTRA-CLEAR color scheme with sharp transitions
        # Very close (0-0.15): BRIGHT RED (immediate danger)
        mask1 = depth_eq < 0.15
        colored[mask1] = [0, 0, 255]

        # Close (0.15-0.3): RED to ORANGE gradient
        mask2 = (depth_eq >= 0.15) & (depth_eq < 0.3)
        t = (depth_eq[mask2] - 0.15) / 0.15
        colored[mask2] = np.stack([
            np.zeros_like(t),
            (t * 165).astype(np.uint8),
            (255 - t * 50).astype(np.uint8)
        ], axis=-1)

        # Medium (0.3-0.5): ORANGE to YELLOW
        mask3 = (depth_eq >= 0.3) & (depth_eq < 0.5)
        t = (depth_eq[mask3] - 0.3) / 0.2
        colored[mask3] = np.stack([
            np.zeros_like(t),
            (165 + t * 90).astype(np.uint8),
            (205 + t * 50).astype(np.uint8)
        ], axis=-1)

        # Safe (0.5-0.7): YELLOW to GREEN
        mask4 = (depth_eq >= 0.5) & (depth_eq < 0.7)
        t = (depth_eq[mask4] - 0.5) / 0.2
        colored[mask4] = np.stack([
            (t * 100).astype(np.uint8),
            np.full_like(t, 255, dtype=np.uint8),
            (255 - t * 255).astype(np.uint8)
        ], axis=-1)

        # Far (0.7-1.0): GREEN to CYAN to BLUE
        mask5 = depth_eq >= 0.7
        t = (depth_eq[mask5] - 0.7) / 0.3
        colored[mask5] = np.stack([
            (100 + t * 155).astype(np.uint8),
            (255 - t * 100).astype(np.uint8),
            (t * 100).astype(np.uint8)
        ], axis=-1)

        # Apply sharpening filter for ULTRA-CLEAR edges
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        colored = cv2.filter2D(colored, -1, kernel)

        # Enhance details with bilateral filter (preserves edges)
        colored = cv2.bilateralFilter(colored, 5, 50, 50)

        return colored

    def process_frame(self, frame: np.ndarray) -> Optional[tuple]:
        """
        Optimized frame processing pipeline for maximum performance.

        Args:
            frame: Input frame

        Returns:
            Tuple of (annotated_frame, detections, guidance, audio_signal, audio_message, depth_map, slam_result, trajectory_result)
        """
        try:
            # Start timer
            self.perf_monitor.start_timer('total')

            # Optimized detection with caching
            self.perf_monitor.start_timer('detection')
            detections = self.detection_pipeline.detector.detect(frame)
            # Limit detections for performance
            detections = detections[:self.max_detections]
            det_time = self.perf_monitor.stop_timer('detection')

            # Smart depth estimation with motion-based caching and threading
            # Skip depth estimation for pySLAM (monocular SLAM doesn't need it)
            depth_map = None
            is_pyslam_configured = self.config.get('slam.use_pyslam', False)
            skip_depth = is_pyslam_configured or (hasattr(self.slam, '__class__') and 'PySLAM' in self.slam.__class__.__name__)

            if not skip_depth:
                self.perf_monitor.start_timer('depth')

                # Simplified depth computation with frame skipping
                if self.frame_count % (self.skip_depth_frames + 1) == 0:
                    depth_map = self.detection_pipeline.depth_estimator.estimate_depth(frame)
                    self.last_depth_map = depth_map
                else:
                    # Reuse cached depth map (much faster!)
                    depth_map = self.last_depth_map
                    if depth_map is None:
                        depth_map = self.detection_pipeline.depth_estimator.estimate_depth(frame)
                        self.last_depth_map = depth_map

                depth_time = self.perf_monitor.stop_timer('depth')
            else:
                # pySLAM uses its own depth estimation - skip our depth model
                depth_time = 0.0

            # Add depth to detections
            if depth_map is not None:
                frame_size = (frame.shape[1], frame.shape[0])  # (width, height)
                for detection in detections:
                    bbox = detection['bbox']
                    depth = self.detection_pipeline.depth_estimator.get_depth_at_bbox(depth_map, bbox, frame_size)
                    detection['depth'] = depth
                    detection['is_danger'] = depth < self.detection_pipeline.min_safe_distance
            else:
                for detection in detections:
                    detection['depth'] = 0.0
                    detection['is_danger'] = False

            # Predict object motion and collision risks
            detections = self.smart_cache.predict_object_motion(detections)
            detections = self.predictive_engine.predict_collision_risk(detections)

            # Get navigation summary
            nav_summary = self.detection_pipeline.get_navigation_summary(detections)

            # Add safe direction suggestion
            safe_direction = self.predictive_engine.suggest_safe_direction(detections, frame.shape[1])
            nav_summary['safe_direction'] = safe_direction
            
            # Enhanced scene understanding with VLM
            scene_analysis = None
            if self.vlm_enabled and self.scene_processor:
                self.perf_monitor.start_timer('scene_understanding')
                scene_analysis = self.scene_processor.process_scene(frame, detections)
                scene_time = self.perf_monitor.stop_timer('scene_understanding')
                if self.frame_count % 50 == 0:  # Log every 50 frames
                    self.logger.debug(f"Scene understanding completed in {scene_time:.3f}s")

            # SLAM tracking (if enabled) - pass depth map for scale
            slam_result = None
            if self.slam_enabled and self.slam is not None:
                self.perf_monitor.start_timer('slam')
                self.logger.debug(f"Processing SLAM on frame {self.frame_count}")
                # For pySLAM (monocular), don't pass depth map
                if hasattr(self.slam, '__class__') and 'PySLAM' in self.slam.__class__.__name__:
                    slam_result = self.slam.process_frame(frame, None)  # No depth for monocular SLAM
                else:
                    slam_result = self.slam.process_frame(frame, depth_map)  # Other SLAM systems can use depth
                slam_time = self.perf_monitor.stop_timer('slam')
                if self.frame_count % 50 == 0:  # Log every 50 frames
                    if slam_result:
                        self.logger.debug(f"SLAM completed in {slam_time:.3f}s, pos:({slam_result['position'][0]:.2f}, {slam_result['position'][1]:.2f}), q:{slam_result['tracking_quality']:.2f}, pts:{slam_result['num_map_points']}")

            # Visual Odometry tracking (if enabled) - alongside SLAM
            vo_result = None
            if self.vo_enabled and self.visual_odometry is not None:
                self.perf_monitor.start_timer('visual_odometry')
                vo_result = self.visual_odometry.process_frame(frame)
                vo_time = self.perf_monitor.stop_timer('visual_odometry')
                if self.frame_count % 50 == 0 and vo_result:  # Log every 50 frames
                    pos = vo_result.get('position', [0, 0, 0])
                    self.logger.debug(f"Visual Odometry completed in {vo_time:.3f}s, pos:({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

                # Update indoor navigator if enabled
                if self.indoor_nav_enabled and self.indoor_navigator is not None:
                    self.indoor_navigator.update(slam_result, detections)

            # Trajectory Prediction (if enabled)
            trajectory_result = None
            if self.trajectory_pred_enabled and self.trajectory_predictor is not None:
                self.perf_monitor.start_timer('trajectory')
                trajectory_result = self.trajectory_predictor.update(detections)
                traj_time = self.perf_monitor.stop_timer('trajectory')

            # 3D Occupancy Grid Update (if enabled and SLAM available)
            if self.occupancy_grid_enabled and self.occupancy_grid is not None:
                if slam_result is not None and depth_map is not None:
                    self.perf_monitor.start_timer('occupancy_grid')
                    camera_pose = slam_result['pose']
                    self.occupancy_grid.update_from_depth(depth_map, camera_pose)
                    occ_time = self.perf_monitor.stop_timer('occupancy_grid')
                    if self.frame_count % 50 == 0:  # Log every 50 frames
                        self.logger.debug(f"Occupancy grid updated in {occ_time:.3f}s")

            # 3D Point Cloud Update (if enabled)
            if self.point_cloud_enabled and self.point_cloud is not None:
                if depth_map is not None:
                    self.perf_monitor.start_timer('point_cloud')
                    camera_pose = slam_result['pose'] if slam_result is not None else None
                    self.point_cloud.add_frame(frame, depth_map, camera_pose)
                    pc_time = self.perf_monitor.stop_timer('point_cloud')

            # Get current time for updates
            current_time = time.time()

            # Movement Visualizer Update (if enabled)
            if self.movement_visualizer_enabled and self.movement_visualizer is not None:
                self.perf_monitor.start_timer('movement_visualizer')
                if slam_result is not None:
                    self.movement_visualizer.update(slam_result, current_time)
                else:
                    # Update with empty result to maintain timing
                    empty_result = {
                        'position': [0, 0, 0],
                        'pose': np.eye(4),
                        'tracking_quality': 0.0,
                        'num_matches': 0,
                        'is_keyframe': False,
                        'num_map_points': 0,
                        'relative_movement': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    }
                    self.movement_visualizer.update(empty_result, current_time)
                mv_time = self.perf_monitor.stop_timer('movement_visualizer')

            # Path planning (RL prediction) - DISABLED for speed
            self.perf_monitor.start_timer('prediction')
            path_plan = None  # Disabled
            pred_time = self.perf_monitor.stop_timer('prediction')

            # Generate LLM guidance (always, for display)
            self.perf_monitor.start_timer('narrative')
            if self.vlm_enabled and self.scene_processor:
                guidance = self._generate_enhanced_guidance(detections, nav_summary, scene_analysis)
            else:
                guidance = self._generate_fast_guidance(detections, nav_summary)
            narr_time = self.perf_monitor.stop_timer('narrative')

            # Generate audio cues
            self.perf_monitor.start_timer('audio')
            audio_signal, audio_message = self.audio_cue_generator.generate_cues(
                detections,
                (self.frame_height, self.frame_width)
            )
            audio_time = self.perf_monitor.stop_timer('audio')

            # Get performance metrics
            total_time = self.perf_monitor.stop_timer('total')
            fps = self.perf_monitor.get_avg_fps()

            # Create UI overlay with depth visualization
            # Skip depth overlay for pySLAM (uses its own depth estimation)
            is_pyslam_configured = self.config.get('slam.use_pyslam', False)
            overlay_depth = None if ((hasattr(self.slam, '__class__') and 'PySLAM' in self.slam.__class__.__name__) or is_pyslam_configured) else depth_map
            annotated_frame = self.robot_ui.draw_clean_overlay(
                frame, detections, fps, safe_direction, overlay_depth
            )

            # Add SLAM info if enabled
            if slam_result is not None:
                position = slam_result['position']
                quality = slam_result['tracking_quality']

                # SLAM status overlay
                slam_color = (0, 255, 0) if quality > 0.7 else (0, 165, 255) if quality > 0.4 else (0, 0, 255)
                cv2.putText(annotated_frame, f"SLAM: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})",
                           (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, slam_color, 1)
                cv2.putText(annotated_frame, f"Quality: {quality:.2f} | Points: {slam_result['num_map_points']}",
                           (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Add trajectory prediction info if enabled
            if trajectory_result is not None:
                tracked_count = len(trajectory_result['tracked_objects'])
                predictions = trajectory_result['predictions']

                # Count objects with predictions
                predicted_count = len([p for p in predictions.values() if len(p['predicted_positions']) > 0])

                cv2.putText(annotated_frame, f"Tracked: {tracked_count} | Predicted: {predicted_count}",
                           (10, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

                # Visualize trajectories if enabled
                if self.config.get('trajectory_prediction.visualize', False):
                    annotated_frame = self.trajectory_predictor.visualize_predictions(
                        annotated_frame,
                        trajectory_result['tracked_objects'],
                        trajectory_result['predictions']
                    )

            # Update smart cache
            self.smart_cache.update(frame, depth_map, detections)

            # Log frame time
            self.perf_monitor.log_frame_time(total_time)

            return annotated_frame, detections, guidance, audio_signal, audio_message, depth_map, slam_result, trajectory_result
        
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()  # Print the full exception trace
            return None

    def _generate_enhanced_guidance(self, detections: List[Dict], nav_summary: Dict, scene_analysis: Optional[Dict]) -> Dict:
        """
        Generate enhanced guidance using VLM scene understanding with improved extraction.

        Args:
            detections: List of detected objects
            nav_summary: Navigation summary
            scene_analysis: VLM scene analysis

        Returns:
            Enhanced guidance dictionary
        """
        try:
            # Use VLM analysis if available
            if scene_analysis and scene_analysis.get('vlm_analysis'):
                vlm_analysis = scene_analysis['vlm_analysis']

                # ACCURACY IMPROVEMENT: Extract navigation guidance from VLM raw response
                vlm_raw = vlm_analysis.get('vlm_raw_response', '')

                # Try to extract the NAVIGATION section from multi-step analysis
                navigation_guidance = ''
                if 'NAVIGATION:' in vlm_raw:
                    # Extract everything after "NAVIGATION:"
                    parts = vlm_raw.split('NAVIGATION:')
                    if len(parts) > 1:
                        navigation_guidance = parts[1].strip()
                else:
                    # Fallback to navigation_guidance field
                    navigation_guidance = scene_analysis.get('navigation_guidance', '')

                # If still empty, use the vlm_raw_response directly
                if not navigation_guidance:
                    navigation_guidance = vlm_analysis.get('navigation_guidance', '')

                # Clean up the guidance (remove extra formatting)
                navigation_guidance = navigation_guidance.replace('...', '').strip()

                # Check for immediate danger in VLM response
                has_vlm_danger = any(keyword in navigation_guidance.upper() for keyword in ['STOP', 'DANGER', 'WARNING', 'CAUTION'])

                # Check for immediate danger from detections
                danger_objects = nav_summary.get('danger_objects', [])
                if danger_objects or has_vlm_danger:
                    if danger_objects:
                        closest_danger = min(danger_objects, key=lambda x: x.get('depth', 10))
                        depth = closest_danger.get('depth', 0)

                        # Combine detection info with VLM guidance
                        if navigation_guidance and not has_vlm_danger:
                            msg = f"STOP! {closest_danger['label']} at {depth:.1f}m. {navigation_guidance}"
                        else:
                            msg = navigation_guidance if navigation_guidance else f"STOP! {closest_danger['label']} at {depth:.1f}m ahead"
                    else:
                        # VLM detected danger but YOLO didn't
                        msg = navigation_guidance

                    return {
                        'narrative': msg,
                        'predictive': '',
                        'combined': msg
                    }

                # Check for caution objects and combine with VLM guidance
                caution_objects = nav_summary.get('caution_objects', [])
                if caution_objects:
                    closest_caution = min(caution_objects, key=lambda x: x.get('depth', 10))
                    depth = closest_caution.get('depth', 0)

                    if navigation_guidance:
                        msg = f"{closest_caution['label']} at {depth:.1f}m. {navigation_guidance}"
                    else:
                        msg = f"Caution: {closest_caution['label']} at {depth:.1f}m"

                    return {
                        'narrative': msg,
                        'predictive': '',
                        'combined': msg
                    }

                # Use VLM guidance for clear path situations
                if navigation_guidance:
                    return {
                        'narrative': navigation_guidance,
                        'predictive': '',
                        'combined': navigation_guidance
                    }

            # Fallback to fast guidance if VLM unavailable
            return self._generate_fast_guidance(detections, nav_summary)

        except Exception as e:
            self.logger.error(f"Enhanced guidance generation error: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_fast_guidance(detections, nav_summary)

    def _generate_fast_guidance(self, detections: List[Dict], nav_summary: Dict) -> Dict:
        """
        Generate simple, clear guidance for blind users.

        Args:
            detections: List of detected objects
            nav_summary: Navigation summary

        Returns:
            Guidance dictionary with simple messages
        """
        try:
            safe_direction = nav_summary.get('safe_direction', 'forward')

            # Immediate danger - very simple
            danger_objects = nav_summary.get('danger_objects', [])
            if danger_objects:
                closest = min(danger_objects, key=lambda x: x.get('depth', 10))
                label = closest['label']

                # Simple direction
                if safe_direction == 'left':
                    msg = f"Stop. {label} ahead. Go left"
                elif safe_direction == 'right':
                    msg = f"Stop. {label} ahead. Go right"
                else:
                    msg = f"Stop. {label} very close"

                return {'narrative': msg, 'predictive': '', 'combined': msg}

            # Caution - simple warning
            caution_objects = nav_summary.get('caution_objects', [])
            if caution_objects:
                closest = min(caution_objects, key=lambda x: x.get('depth', 10))
                msg = f"{closest['label']} ahead. Slow down"
                return {'narrative': msg, 'predictive': '', 'combined': msg}

            # Clear path
            msg = "Path clear"
            return {'narrative': msg, 'predictive': '', 'combined': ''}

        except Exception as e:
            return {'narrative': 'Continue', 'predictive': '', 'combined': 'Continue'}
    

    def _display_terminal_info(self, fps, detections, total_time, slam_result, trajectory_result):
        """Display information in the terminal using Rich."""
        # Clear the terminal screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Create the console
        console = Console()
        
        # Create a layout for the display
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        
        # FPS indicator with color based on performance
        if fps > 10:
            fps_text = Text(f"FPS: {fps:.1f}", style="bold green")
        elif fps > 5:
            fps_text = Text(f"FPS: {fps:.1f}", style="bold yellow")
        else:
            fps_text = Text(f"FPS: {fps:.1f}", style="bold red")
        
        # Danger/Caution counts
        danger_objects = [d for d in detections if d.get('depth', 10) < self.danger_distance]
        caution_objects = [d for d in detections if self.danger_distance <= d.get('depth', 10) < self.caution_distance]
        
        # Status indicator
        if danger_objects:
            status_text = Text("⚠ DANGER", style="bold red")
        elif caution_objects:
            status_text = Text("⚠ CAUTION", style="bold yellow")
        elif detections:
            status_text = Text("SAFE", style="bold green")
        else:
            status_text = Text("CLEAR", style="bold green")
        
        # Create a table for main information
        table = Table(title="OrbyGlasses Navigation Status", box=ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        
        table.add_row("FPS", f"{fps:.1f}")
        table.add_row("Danger Objects", str(len(danger_objects)))
        table.add_row("Caution Objects", str(len(caution_objects)))
        table.add_row("Status", status_text.plain)
        table.add_row("Process Time", f"{total_time:.0f}ms")
        
        # Add closest object info if detections exist
        if detections:
            closest = min(detections, key=lambda x: x.get('depth', 10))
            closest_text = Text(f"{closest['label']} {closest['depth']:.1f}m", style="bold")
            
            # Color code based on distance
            if closest['depth'] < self.danger_distance:
                closest_text.stylize("red", 0, len(closest_text))
            elif closest['depth'] < self.caution_distance:
                closest_text.stylize("yellow", 0, len(closest_text))
            else:
                closest_text.stylize("green", 0, len(closest_text))
                
            table.add_row("Closest Object", closest_text.plain)
        
        # Add SLAM info if available
        if slam_result is not None:
            position = slam_result['position']
            quality = slam_result['tracking_quality']
            slam_pos_text = f"({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})"
            
            # Color code SLAM quality
            if quality > 0.7:
                quality_text = Text(f"{quality:.2f}", style="green")
            elif quality > 0.4:
                quality_text = Text(f"{quality:.2f}", style="yellow")
            else:
                quality_text = Text(f"{quality:.2f}", style="red")
                
            table.add_row("SLAM Position", slam_pos_text)
            table.add_row("Tracking Quality", quality_text.plain)
            table.add_row("Map Points", str(slam_result['num_map_points']))
        
        # Add trajectory info if available
        if trajectory_result is not None:
            tracked_count = len(trajectory_result['tracked_objects'])
            predictions = trajectory_result['predictions']
            predicted_count = len([p for p in predictions.values() if len(p['predicted_positions']) > 0])
            
            table.add_row("Tracked Objects", str(tracked_count))
            table.add_row("Predicted Trajectories", str(predicted_count))
        
        # Print the table
        console.print(table)
        
        # Add a summary panel
        summary_text = f"[bold]Navigation System Active[/bold] | [green]Objects: {len(detections)}[/green] | [red]Danger: {len(danger_objects)}[/red] | [yellow]Caution: {len(caution_objects)}[/yellow]"
        console.print(Panel(summary_text, title="System Summary", border_style="blue"))
        
        # Performance info
        performance_text = f"[bold]Performance[/bold]: {fps:.1f} FPS | {total_time:.0f}ms processing | [green]Active[/green]"
        console.print(Panel(performance_text, title="Performance", border_style="green"))


    def run(self, display: bool = True, save_video: bool = False, separate_slam: bool = False):
        """
        Main run loop.

        Args:
            display: Show video display
            save_video: Save output video
            separate_slam: Show SLAM map in a separate window
        """
        if not self.initialize_camera():
            self.logger.error("Cannot start without camera")
            return

        self.running = True

        # Start Visual Odometry if enabled
        if self.vo_enabled and self.visual_odometry is not None:
            if self.visual_odometry.start():
                self.logger.info("✅ Visual Odometry started successfully")
            else:
                self.logger.error("Failed to start Visual Odometry")
                self.vo_enabled = False

        # Video writer
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = f"data/logs/output_{int(time.time())}.mp4"
            video_writer = cv2.VideoWriter(
                output_path, fourcc, 20.0,
                (self.frame_width, self.frame_height)
            )

        # Optimized welcome message
        self.audio_manager.speak("Navigation system ready", priority=True)

        self.logger.info("OrbyGlasses - High-Performance Navigation System")
        self.logger.info(f"Resolution: {self.frame_width}x{self.frame_height} @ {self.config.get('camera.fps', 30)} FPS")
        self.logger.info(f"Audio interval: {self.audio_interval}s (danger: {self.danger_audio_interval}s) | Depth: every {self.skip_depth_frames + 1} frames")
        if self.slam_enabled:
            self.logger.info("🗺️ SLAM enabled - Real-time mapping")
        if self.occupancy_grid_enabled:
            self.logger.info("🗺️ 3D Occupancy Map enabled - Interactive controls:")
            self.logger.info("   R: Reset view | T: Toggle 2D/3D | C: Clear map | S: Save map | L: Load map")
        if self.conversation_enabled:
            activation = self.config.get('conversation.activation_phrase', 'hey glasses')
            self.logger.info(f"💬 Voice: '{activation}'")
        self.logger.info(f"Press '{self.config.get('safety.emergency_stop_key', 'q')}' to quit")

        try:
            self.logger.debug("Entering main processing loop.")
            while self.running:
                self.logger.debug("Attempting to read frame from camera.")
                # Capture frame
                ret, frame = self.camera.read()

                if not ret:
                    self.logger.warning("Failed to capture frame")
                    break

                # Flip frame horizontally for mirror effect (before processing)
                frame = cv2.flip(frame, 1)

                self.logger.debug("Frame read successfully. Processing frame.")
                self.frame_count += 1

                # Process frame
                result = self.process_frame(frame)
                if result is None or len(result) != 8:
                    self.logger.warning("Skipping frame due to invalid processing result.")
                    continue
                annotated_frame, detections, guidance, audio_signal, audio_message, depth_map, slam_result, trajectory_result = result

                # Play adaptive audio beaconing (if available and not speaking)
                # Play beacons with higher priority, separate from voice guidance
                current_time = time.time()
                # Play audio beacons more frequently (every 0.2 seconds) but with logic to avoid conflicts
                if hasattr(self, 'last_beacon_time'):
                    time_since_beacon = current_time - self.last_beacon_time
                else:
                    self.last_beacon_time = current_time - 0.3
                    time_since_beacon = 0.3
                
                # Play beacons more frequently for better user experience
                if time_since_beacon > 0.3:  # Play every 0.3 seconds instead of 0.5
                    # Check if audio signal has content (non-zero)
                    if np.any(audio_signal):
                        try:
                            # Only play if not already speaking
                            if not self.audio_manager.is_speaking:
                                self.audio_manager.play_sound(audio_signal, sample_rate=16000)  # Fixed sample rate matching echolocation
                        except Exception as e:
                            print(f"Audio playback error: {e}")
                            # Fallback to system beep if play_sound fails
                            import sys
                            sys.stdout.write('\a')
                            sys.stdout.flush()
                    
                    self.last_beacon_time = current_time

                # Update 3D map
                if depth_map is not None:
                    self.mapper_3d.update(frame, depth_map, detections)

                # Get indoor navigation guidance if available
                indoor_guidance = None
                if hasattr(self, 'indoor_navigator') and self.indoor_navigator:
                    indoor_guidance = self.indoor_navigator.get_navigation_guidance()

                # Conversational Navigation - Check for activation (non-blocking)
                if self.conversation_enabled and self.conversation_manager:
                    # Update conversation context
                    nav_summary = self.detection_pipeline.get_navigation_summary(detections)
                    self.conversation_manager.update_scene_context(detections, nav_summary)

                    # Check if activation was detected (completely non-blocking queue check)
                    activation_result = self.conversation_manager.check_activation_result()
                    if activation_result:
                        self.logger.info("💬 Conversation activated!")
                        # Handle conversation interaction
                        scene_context = {
                            'detected_objects': detections,
                            'obstacles': nav_summary.get('danger_objects', []),
                            'path_clear': nav_summary.get('path_clear', True)
                        }
                        # Add SLAM position if available for location saving
                        if slam_result:
                            scene_context['slam_position'] = np.array(slam_result['position'])
                        self.conversation_manager.handle_conversation_interaction(scene_context)

                # Smart Audio System - Priority-based alerts
                # Check for danger zone objects (< 1m) - PRIORITY ALERT
                danger_objects = [d for d in detections if d.get('depth', 10) < self.danger_distance]
                has_danger = len(danger_objects) > 0

                # Determine appropriate interval based on danger level
                active_interval = self.danger_audio_interval if has_danger else self.audio_interval

                # DANGER ZONE - Priority alert
                if has_danger:
                    if (current_time - self.last_audio_time) > self.danger_audio_interval and not self.audio_manager.is_speaking:
                        closest_danger = min(danger_objects, key=lambda x: x['depth'])
                        depth = closest_danger['depth']

                        # Use relatable distance terms with specific actions
                        label = closest_danger['label']

                        if depth < 0.3:
                            distance_term = "very close"
                            urgency = "Stop now!"
                        elif depth < 0.5:
                            distance_term = "close"
                            urgency = "Caution!"
                        else:
                            distance_term = "ahead"
                            urgency = "Watch out!"

                        # Determine direction with specific action
                        center = closest_danger.get('center', [160, 160])
                        if center[0] < 106:
                            direction = "on your left"
                            action = "Move right"
                        elif center[0] > 213:
                            direction = "on your right"
                            action = "Move left"
                        else:
                            direction = "directly ahead"
                            action = "Stop and step aside"

                        msg = f"{urgency} {label} {distance_term} {direction}. {action}"
                        self.logger.error(f"🚨 DANGER ALERT: \"{msg}\"")
                        self.audio_manager.speak(msg, priority=True)
                        self.last_audio_time = current_time

                # Use indoor navigation guidance if available, otherwise use Ollama-generated narrative
                elif indoor_guidance:
                    if (current_time - self.last_audio_time) > self.audio_interval and not self.audio_manager.is_speaking:
                        msg = indoor_guidance
                        self.logger.info(f"🔊 Indoor Navigation: \"{msg}\"")
                        self.audio_manager.speak(msg, priority=False)
                        self.last_audio_time = current_time
                elif guidance.get('combined') and len(guidance.get('combined', '').strip()) > 0:
                    if self.vlm_enabled and (current_time - self.last_vlm_guidance_time) > 5.0 and not self.audio_manager.is_speaking:
                        msg = guidance['combined'].strip()
                        # Skip if VLM output is just labels or too short
                        if len(msg) > 10 and not msg.startswith('SCENE:') and not msg.startswith('NAVIGATION:'):
                            self.logger.info(f"🔊 VLM Guidance: \"{msg}\"")
                            self.audio_manager.speak(msg, priority=False)
                            self.last_vlm_guidance_time = current_time
                        else:
                            self.logger.warning(f"VLM guidance too short or invalid: '{msg}', using fallback")

                # Fallback to simple message - ALWAYS provide audio feedback
                if len(detections) > 0 and (current_time - self.last_audio_time) > self.audio_interval and not self.audio_manager.is_speaking:
                    # Get closest object overall
                    closest = min(detections, key=lambda x: x.get('depth', 10))
                    label = closest.get('label', 'object')
                    depth = closest.get('depth', 5.0)
                    center = closest.get('center', [160, 160])

                    # Determine position
                    if center[0] < 106:
                        position = "on your left"
                    elif center[0] > 213:
                        position = "on your right"
                    else:
                        position = "ahead"

                    # Determine distance term
                    if depth < 1.5:
                        distance = "nearby"
                    elif depth < 3.0:
                        distance = ""  # Don't mention far objects
                    else:
                        distance = "in the distance"

                    # Build message
                    if distance:
                        msg = f"{label} {distance} {position}"
                    else:
                        msg = f"{label} {position}"

                    # Look for additional nearby objects
                    additional = []
                    for det in detections[1:3]:  # Check next 2 objects
                        if det.get('depth', 10) < 2.5:
                            det_center = det.get('center', [160, 160])
                            if det_center[0] < 106:
                                det_pos = "left"
                            elif det_center[0] > 213:
                                det_pos = "right"
                            else:
                                det_pos = "center"

                            # Don't repeat same position
                            if det_pos not in msg.lower():
                                additional.append(f"{det.get('label', 'object')} on {det_pos}")

                    # Add one additional object if found
                    if additional:
                        msg = f"{msg}. {additional[0]}"

                    self.logger.info(f"🔊 Audio: \"{msg}\"")
                    self.audio_manager.speak(msg, priority=False)
                    self.last_audio_time = current_time

                # Path clear
                else:
                    if (current_time - self.last_path_clear_time) > 10.0 and not self.audio_manager.is_speaking: # Only speak "Path clear" every 10 seconds
                        msg = "Path clear"
                        self.logger.info(f"🔊 Audio: \"{msg}\"")
                        self.audio_manager.speak(msg, priority=False)
                        self.last_path_clear_time = current_time

                self.logger.debug("Frame processed. Displaying windows.")
                # Clean robot-style display
                if display:
                    # Main camera view - resize to smaller size for cleaner desktop
                    display_width = 480  # Reduced from 640
                    display_height = 360  # Reduced from 480
                    camera_display = cv2.resize(annotated_frame, (display_width, display_height),
                                               interpolation=cv2.INTER_LINEAR)
                    cv2.imshow('OrbyGlasses', camera_display)

                    # Show SLAM map in a separate window if requested (skip for pySLAM - it has its own viewer)
                    is_pyslam_configured = self.config.get('slam.use_pyslam', False)
                    if separate_slam and self.slam_enabled and self.slam_map_viewer and slam_result and not is_pyslam_configured:
                        # Get map points using method (some SLAM systems have attribute, some have method)
                        map_pts = self.slam.map_points if hasattr(self.slam, 'map_points') else self.slam.get_map_points() if hasattr(self.slam, 'get_map_points') else []
                        self.slam_map_viewer.update(slam_result, map_pts)
                        map_image = self.slam_map_viewer.get_map_image()
                        cv2.imshow('SLAM Map', map_image)  # Keep original SLAM map size

                    # DISABLED: Depth map window (pySLAM handles its own depth estimation)
                    # Skip depth visualization when using pySLAM
                    skip_depth_viz = is_pyslam_configured or (hasattr(self.slam, '__class__') and 'PySLAM' in self.slam.__class__.__name__)
                    if False and depth_map is not None and not skip_depth_viz:
                        # Use new fast dark visualizer if available
                        if self.depth_viz:
                            depth_colored = self.depth_viz.visualize(depth_map)
                        else:
                            # Fallback to old method
                            depth_colored = self._create_ultra_clear_depth_colormap(depth_map)

                        # Resize depth map to smaller size
                        depth_display_width = 480
                        depth_display_height = 360
                        depth_display = cv2.resize(depth_colored, (depth_display_width, depth_display_height),
                                                  interpolation=cv2.INTER_LINEAR)
                        cv2.imshow('Depth Map', depth_display)

                    # Show SLAM map viewer MERGED with Advanced Navigation Panel
                    if self.slam_enabled and self.slam_map_viewer and slam_result and not separate_slam:
                        # Get map points using method (some SLAM systems have attribute, some have method)
                        map_pts = self.slam.map_points if hasattr(self.slam, 'map_points') else self.slam.get_map_points() if hasattr(self.slam, 'get_map_points') else []
                        self.slam_map_viewer.update(slam_result, map_pts)
                        map_image = self.slam_map_viewer.get_map_image()

                        # If advanced nav panel enabled, show merged view
                        if self.advanced_nav_enabled:
                            # Update panel with latest data (10Hz update, non-blocking)
                            goal_pos = None
                            if self.indoor_navigator and hasattr(self.indoor_navigator, 'current_goal'):
                                goal_pos = self.indoor_navigator.current_goal

                            self.advanced_nav_panel.update(slam_result, detections, goal_pos)

                            # Render side-by-side: SLAM map (left) + Navigation panel (right)
                            merged_nav_display = self.advanced_nav_panel.render_side_by_side(map_image)
                            cv2.imshow('Navigation', merged_nav_display)
                        else:
                            # Just show SLAM map
                            cv2.imshow('Map', map_image)

                    # Show 3D Occupancy Grid if enabled
                    if self.occupancy_grid_enabled and self.occupancy_grid is not None:
                        occupancy_grid_image = self.occupancy_grid.visualize_3d_interactive(slam_result['position'] if slam_result else None)
                        cv2.imshow('Occupancy Grid', occupancy_grid_image)

                # Save video
                if video_writer:
                    video_writer.write(annotated_frame)

                # Check for quit and handle keyboard controls
                key = cv2.waitKey(10) & 0xFF
                emergency_key = self.config.get('safety.emergency_stop_key', 'q')

                # Handle occupancy grid controls
                if self.occupancy_grid_enabled and self.occupancy_grid is not None:
                    if self.occupancy_grid.update_view_controls(key):
                        pass  # View updated, will refresh on next frame
                    
                    # Interactive controls for occupancy map
                    if key == ord('r'):  # Reset view
                        self.occupancy_grid.reset_view()
                    elif key == ord('t'):  # Toggle 2D/3D view
                        self.occupancy_grid.toggle_view_mode()
                    elif key == ord('c'):  # Clear map
                        self.occupancy_grid.clear_map()
                    elif key == ord('s'):  # Save map
                        self.occupancy_grid.save_map()
                    elif key == ord('l'):  # Load map
                        self.occupancy_grid.load_map()

                # Handle point cloud controls
                if self.point_cloud_enabled and self.point_cloud is not None:
                    if self.point_cloud.update_view_controls(key):
                        pass  # View updated, will refresh on next frame

                # Handle movement visualizer controls if enabled
                if self.movement_visualizer_enabled and self.movement_visualizer is not None:
                    # Currently no special keyboard controls for movement visualizer
                    pass

                if key == ord(emergency_key):
                    self.logger.info("Emergency stop activated")
                    self.audio_manager.speak("Navigation stopped", priority=True)
                    break

                # Optimized performance logging
                if self.frame_count % 200 == 0:  # Less frequent logging
                    stats = self.perf_monitor.get_stats()
                    current_fps = stats.get('fps', 0)
                    self.logger.info(f"Performance: {current_fps:.1f} FPS, {stats.get('avg_frame_time_ms', 0):.1f}ms, {len(detections)} objects")

        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")

        except Exception as e:
            self.logger.error(f"Runtime error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.cleanup(video_writer)


    def cleanup(self, video_writer=None):
        """Clean up resources."""
        self.logger.info("Shutting down...")

        # Stop conversation manager (background voice listener)
        if self.conversation_manager:
            self.conversation_manager.stop()

        # Stop Visual Odometry
        if self.vo_enabled and self.visual_odometry is not None:
            self.visual_odometry.cleanup()
            self.logger.info("Visual Odometry stopped")

        # Stop 3D mapper
        self.mapper_3d.stop()

        # Reset movement visualizer
        if self.movement_visualizer_enabled and self.movement_visualizer is not None:
            self.movement_visualizer.reset()

        if self.camera:
            self.camera.release()

        if video_writer:
            video_writer.release()

        cv2.destroyAllWindows()

        self.audio_manager.stop()

        # Final stats
        stats = self.perf_monitor.get_stats()
        self.logger.info(f"Final performance stats: {stats}")
        self.logger.info(f"Total frames processed: {self.frame_count}")

        self.logger.info("Shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='OrbyGlasses - Bio-Mimetic Navigation System')

    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without video display (headless mode)')
    parser.add_argument('--save-video', action='store_true',
                       help='Save output video')
    parser.add_argument('--train-rl', action='store_true',
                       help='Train RL model before running')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode with sample video')
    parser.add_argument('--separate-slam', action='store_true',
                        help='Show SLAM map in a separate window')

    args = parser.parse_args()

    # Initialize system
    system = OrbyGlasses(config_path=args.config)

    # Train RL if requested
    if args.train_rl:
        print("Training RL model...")
        system.path_planner.predictor.train()
        print("Training complete!")

    # Run system
    if args.test:
        # Test mode - use sample video if available
        print("Test mode not yet implemented")
        # TODO: Implement test mode with sample videos
    else:
        system.run(display=not args.no_display, save_video=args.save_video, separate_slam=args.separate_slam)


if __name__ == "__main__":
    main()
