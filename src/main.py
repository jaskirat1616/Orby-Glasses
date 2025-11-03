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

# Rich terminal UI
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.table import Table
    from rich.text import Text
    from rich.panel import Panel
    from rich import box
    ROUNDED = box.ROUNDED
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Rich not available - terminal display disabled")

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Core modules - lazy import for feature_matching mode
from core.utils import (
    ConfigManager, Logger, AudioManager,
    DataLogger, PerformanceMonitor, ensure_directories, check_device
)

# Lazy import DetectionPipeline to avoid YOLO dependency issues in feature_matching mode
DetectionPipeline = None
try:
    from core.detection import DetectionPipeline
except ImportError as e:
    print(f"⚠️ DetectionPipeline not available: {e}")
    print("   This is OK for feature_matching mode")
from core.echolocation import AudioCueGenerator
from core.narrative import ContextualAssistant
from core.smart_cache import SmartCache, PredictiveEngine
from core.error_handler import ErrorHandler
from core.stair_detection import StairCurbDetector

# New modules
try:
    from core.depth_anything_v2 import DepthAnythingV2
    DEPTH_ANYTHING_V2_AVAILABLE = True
except ImportError:
    DEPTH_ANYTHING_V2_AVAILABLE = False
    print("Note: Depth Anything V2 not available")

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

# Navigation modules - Simplified to use native pySLAM
try:
    from navigation.pyslam_live import LivePySLAM, PYSLAM_AVAILABLE
    print("✅ Live pySLAM available")
except ImportError as e:
    PYSLAM_AVAILABLE = False
    print(f"Note: pySLAM not available: {e}")
    print("Run: cd third_party/pyslam && source ~/.python/venvs/pyslam/bin/activate")

try:
    from navigation.pyslam_vo_integration import PySLAMVisualOdometry, PYSLAM_VO_AVAILABLE
    print("✅ pySLAM Visual Odometry available")
except ImportError as e:
    PYSLAM_VO_AVAILABLE = False
    print(f"Note: pySLAM Visual Odometry not available: {e}")

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

    def __init__(self, config_path: str = "config/config.yaml", video_source: Optional[str] = None, mode: Optional[str] = None):
        """
        Initialize optimized OrbyGlasses system.

        Args:
            config_path: Path to configuration file
            video_source: Optional video file path or camera index to override config
        """
        # Ensure directories exist
        ensure_directories()

        # Load configuration
        self.config = ConfigManager(config_path)

        # Set visualization mode (already set in early init if feature_matching)
        if not hasattr(self, 'viz_mode'):
            if mode:
                self.viz_mode = mode
            else:
                self.viz_mode = self.config.get('visualization_mode.mode', 'features') if self.config else 'features'

        # Adjust SLAM/VO settings based on mode (access internal dict)
        if self.viz_mode == 'features':
            # Features mode: Use SLAM but only show feature matching window
            # SLAM is needed to get feature matching data
            if 'slam' not in self.config.config:
                self.config.config['slam'] = {}
            if 'visual_odometry' not in self.config.config:
                self.config.config['visual_odometry'] = {}
            self.config.config['slam']['enabled'] = True
            self.config.config['visual_odometry']['enabled'] = False
            # Note: We'll control which windows to show via show_features flag
        elif self.viz_mode == 'full_slam':
            # Full SLAM mode: Enable SLAM and show ALL windows
            if 'slam' not in self.config.config:
                self.config.config['slam'] = {}
            if 'visual_odometry' not in self.config.config:
                self.config.config['visual_odometry'] = {}
            self.config.config['slam']['enabled'] = True
            self.config.config['visual_odometry']['enabled'] = False
        elif self.viz_mode == 'vo':
            # VO mode: Visual odometry only (no SLAM)
            if 'slam' not in self.config.config:
                self.config.config['slam'] = {}
            if 'visual_odometry' not in self.config.config:
                self.config.config['visual_odometry'] = {}
            self.config.config['slam']['enabled'] = False
            self.config.config['visual_odometry']['enabled'] = True
        elif self.viz_mode == 'basic':
            # Basic mode: Use whatever config says (default SLAM)
            pass  # Don't override config
        elif self.viz_mode == 'feature_matching':
            # Feature matching mode: Use SLAM, overlay feature matching on main window only
            # SLAM is needed to get feature matching data
            if 'slam' not in self.config.config:
                self.config.config['slam'] = {}
            if 'visual_odometry' not in self.config.config:
                self.config.config['visual_odometry'] = {}
            self.config.config['slam']['enabled'] = True
            self.config.config['visual_odometry']['enabled'] = False
            # Only show main window with feature matching overlay

        # Initialize minimal logger for performance
        log_level_str = self.config.get('logging.level', 'WARNING')  # Reduced logging
        import logging
        log_level = getattr(logging, log_level_str.upper(), logging.WARNING)
        self.logger = Logger(log_level=log_level)
        self.logger.info("OrbyGlasses - High-Performance Navigation System")

        # Check device
        device = check_device()
        self.logger.info(f"Device: {device}")
        self.logger.info(f"Mode: {self.viz_mode.upper()}")

        # Initialize core components only
        self.logger.info("Initializing core components...")

        self.audio_manager = AudioManager(self.config)
        # Only initialize DetectionPipeline if available and not in feature_matching mode
        if DetectionPipeline is not None and self.viz_mode != 'feature_matching':
            self.detection_pipeline = DetectionPipeline(self.config)
        else:
            self.detection_pipeline = None
            if self.viz_mode == 'feature_matching':
                print("ℹ️  Running in feature_matching mode - skipping detection pipeline")
        self.audio_cue_generator = AudioCueGenerator(self.config)
        self.contextual_assistant = ContextualAssistant(self.config)
        self.path_planner = PathPlanner(self.config)

        # Stair and curb detection (critical safety feature)
        self.stair_detection_enabled = self.config.get('stair_detection.enabled', True)
        if self.stair_detection_enabled:
            self.stair_detector = StairCurbDetector(self.config)
            self.logger.info("✓ Stair/curb detection enabled (critical safety feature)")
        else:
            self.stair_detector = None
            self.logger.warning("⚠️  Stair/curb detection DISABLED - falls risk increased!")

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

        # SLAM and Indoor Navigation - Simplified to use native pySLAM
        self.slam_enabled = self.config.get('slam.enabled', False)
        self.indoor_nav_enabled = self.config.get('indoor_navigation.enabled', False)
        if self.slam_enabled:
            if PYSLAM_AVAILABLE:
                self.logger.info("🚀 Initializing pySLAM (Professional Python SLAM Framework)...")
                # Pass viz_mode to LivePySLAM so it can hide windows in feature_matching mode
                self.slam = LivePySLAM(self.config, viz_mode=self.viz_mode)
                feature_type = self.config.get('slam.feature_type', 'ORB')
                self.logger.info(f"✓ Using pySLAM with {feature_type} features")
                self.logger.info("✓ Loop closure, bundle adjustment, map persistence")
                self.logger.info("✓ Native pySLAM visualization and 3D windows")
                # pySLAM has its own viewer, so we don't need a custom one
                self.slam_map_viewer = None
            else:
                self.logger.error("❌ SLAM enabled but pySLAM not available!")
                self.logger.info("Install pySLAM from third_party/pyslam")
                self.slam = None
                self.slam_map_viewer = None

            if self.slam and self.indoor_nav_enabled:
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


        # Data logging
        self.data_logger = DataLogger()

        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()

        # Camera setup
        self.camera = None
        self.frame_width = self.config.get('camera.width', 640)
        self.frame_height = self.config.get('camera.height', 480)
        self.video_source = video_source  # Store video source override
        self.need_frame_resize = False  # Flag to track if frames need resizing
        self.target_fps = 30  # Default target FPS
        self.video_fps_skip_rate = 1  # Skip rate for frame processing

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
        # Use video_source override if provided, otherwise use config
        if self.video_source is not None:
            camera_source = self.video_source
        else:
            camera_source = self.config.get('camera.source', 0)

        try:
            # Determine if it's a camera index (int or numeric string) or video file path
            is_camera_index = False
            if isinstance(camera_source, int):
                is_camera_index = True
                source_type = "camera"
            elif isinstance(camera_source, str):
                # Check if it's a numeric string (camera index) or file path
                try:
                    camera_source = int(camera_source)
                    is_camera_index = True
                    source_type = "camera"
                except ValueError:
                    # It's a file path, use as-is
                    source_type = "video file"
            
            self.logger.info(f"Initializing {source_type}: {camera_source}")
            
            self.camera = cv2.VideoCapture(camera_source)

            if not self.camera.isOpened():
                self.logger.error(f"Failed to open {source_type}: {camera_source}")
                return False

            # Set resolution (only for cameras, videos have fixed resolution)
            if is_camera_index:
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            else:
                # For video files, use the actual dimensions (don't resize)
                actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Auto-downscale high-resolution videos for better performance
                max_width = self.config.get('camera.max_video_width', 640)  # Landscape videos
                max_height = self.config.get('camera.max_video_height', 480)  # Standard height

                # For portrait videos (height > width), swap the constraints
                is_portrait = actual_height > actual_width
                if is_portrait:
                    # For portrait: constrain width more aggressively to avoid too-narrow FOV
                    max_width = 480   # Allow wider portrait frames
                    max_height = 640  # But not too tall
                    self.logger.info(f"Portrait video detected ({actual_width}x{actual_height})")

                if actual_width > max_width or actual_height > max_height:
                    # Calculate scale to fit within max dimensions while maintaining aspect ratio
                    width_scale = max_width / actual_width if actual_width > max_width else 1.0
                    height_scale = max_height / actual_height if actual_height > max_height else 1.0
                    scale = min(width_scale, height_scale)

                    self.frame_width = int(actual_width * scale)
                    self.frame_height = int(actual_height * scale)
                    self.need_frame_resize = True  # Mark that frames need resizing
                    self.logger.info(f"Video downscaled from {actual_width}x{actual_height} to {self.frame_width}x{self.frame_height} for performance")
                else:
                    # Use original resolution if within limits
                    self.frame_width = actual_width
                    self.frame_height = actual_height
                    self.need_frame_resize = False
                    self.logger.info(f"Using video's original resolution: {actual_width}x{actual_height}")
                
                # Note: SLAM dimensions will be updated dynamically in _process_pyslam_frame

            # Set FPS (only for cameras, videos have fixed FPS)
            if is_camera_index:
                fps = self.config.get('camera.fps', 30)
                self.camera.set(cv2.CAP_PROP_FPS, fps)
            else:
                # For video files, get the actual FPS and optionally cap it
                actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
                max_video_fps = self.config.get('camera.max_video_fps', 30)  # Cap high FPS videos
                if actual_fps > max_video_fps:
                    self.logger.info(f"Video FPS {actual_fps:.1f} exceeds max, will process at {max_video_fps} FPS (skipping frames)")
                    self.target_fps = max_video_fps
                    self.video_fps_skip_rate = actual_fps / max_video_fps  # e.g., 60/30 = 2 (skip every other frame)
                else:
                    self.target_fps = actual_fps
                    self.video_fps_skip_rate = 1  # No skipping
                self.logger.info(f"Video properties: {self.frame_width}x{self.frame_height} @ {actual_fps:.1f} FPS (target: {self.target_fps:.1f} FPS)")

            self.logger.info(f"{source_type.capitalize()} initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"{source_type.capitalize()} initialization error: {e}")
            return False

    def process_frame(self, frame: np.ndarray) -> Optional[tuple]:
        """
        Optimized frame processing pipeline for maximum performance.

        Args:
            frame: Input frame

        Returns:
            Tuple of (annotated_frame, detections, guidance, audio_signal, audio_message, depth_map, slam_result, trajectory_result, stair_result)
        """
        try:
            # Start timer
            self.perf_monitor.start_timer('total')

            # Optimized detection with caching (skip in feature_matching mode)
            self.perf_monitor.start_timer('detection')
            if self.detection_pipeline is not None:
                detections = self.detection_pipeline.detector.detect(frame)
                # Limit detections for performance
                detections = detections[:self.max_detections]
            else:
                detections = []  # No detections in feature_matching mode
            det_time = self.perf_monitor.stop_timer('detection')

            # Depth estimation - ALWAYS run for safety (monocular depth critical for blind users)
            # Even with SLAM enabled, depth estimation is essential for obstacle distance
            depth_map = None

            # Check if depth estimator is available
            has_depth_estimator = (hasattr(self, 'detection_pipeline') and
                                   hasattr(self.detection_pipeline, 'depth_estimator') and
                                   self.detection_pipeline.depth_estimator is not None)

            if has_depth_estimator:
                self.perf_monitor.start_timer('depth')
                # Simplified depth computation with frame skipping
                if self.frame_count % (self.skip_depth_frames + 1) == 0:
                    depth_map = self.detection_pipeline.depth_estimator.estimate_depth(frame)
                    self.last_depth_map = depth_map
                else:
                    # Reuse cached depth map
                    depth_map = self.last_depth_map
                    if depth_map is None:
                        depth_map = self.detection_pipeline.depth_estimator.estimate_depth(frame)
                        self.last_depth_map = depth_map
                depth_time = self.perf_monitor.stop_timer('depth')
            else:
                # Depth estimator not available
                depth_time = 0.0

            # Add depth to detections
            if depth_map is not None and has_depth_estimator and self.detection_pipeline is not None:
                frame_size = (frame.shape[1], frame.shape[0])  # (width, height)
                for detection in detections:
                    bbox = detection['bbox']
                    depth = self.detection_pipeline.depth_estimator.get_depth_at_bbox(depth_map, bbox, frame_size)
                    detection['depth'] = depth
                    detection['depth_uncertain'] = False  # Depth measured
                    detection['is_danger'] = depth < self.detection_pipeline.min_safe_distance
            else:
                # CRITICAL: No depth available - mark as uncertain and treat as potential danger
                for detection in detections:
                    detection['depth'] = None  # Unknown distance
                    detection['depth_uncertain'] = True  # Flag for audio warnings
                    detection['is_danger'] = True  # ASSUME DANGER when distance unknown (safety first)

            # CRITICAL: Stair and curb detection (prevent falls)
            stair_result = None
            if self.stair_detection_enabled and self.stair_detector and depth_map is not None:
                self.perf_monitor.start_timer('stair_detection')
                stair_result = self.stair_detector.detect(depth_map, frame)
                stair_time = self.perf_monitor.stop_timer('stair_detection')

                # Log stair detections
                if stair_result.get('drop_detected', False):
                    hazard_type = stair_result.get('hazard_type', 'unknown')
                    distance = stair_result.get('distance_to_hazard', 'unknown')
                    confidence = stair_result.get('confidence', 0.0)
                    self.logger.warning(f"⚠️  HAZARD DETECTED: {hazard_type} at {distance}m (confidence: {confidence:.1%})")

            # Predict object motion and collision risks
            detections = self.smart_cache.predict_object_motion(detections)
            detections = self.predictive_engine.predict_collision_risk(detections)

            # Get navigation summary (skip in feature_matching mode)
            if self.detection_pipeline is not None:
                nav_summary = self.detection_pipeline.get_navigation_summary(detections)
            else:
                nav_summary = {'path_clear': True, 'danger_objects': [], 'caution_objects': []}

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

            # SLAM tracking (if enabled) - pySLAM handles everything internally
            slam_result = None
            if self.slam_enabled and self.slam is not None:
                self.perf_monitor.start_timer('slam')
                self.logger.debug(f"Processing SLAM on frame {self.frame_count}")
                slam_result = self.slam.process_frame(frame, None)  # pySLAM doesn't need external depth
                slam_time = self.perf_monitor.stop_timer('slam')
                if self.frame_count % 50 == 0 and slam_result:
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
                guidance = self._generate_enhanced_guidance(detections, nav_summary, scene_analysis, slam_result)
            else:
                guidance = self._generate_fast_guidance(detections, nav_summary, slam_result)
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

            # Create UI overlay (no depth visualization - pySLAM handles that)
            # In feature_matching mode, skip overlay or use minimal overlay
            if self.viz_mode == 'feature_matching':
                # Minimal overlay for feature_matching mode
                annotated_frame = frame.copy()
                # Add FPS if available
                if fps > 0:
                    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                annotated_frame = self.robot_ui.draw_clean_overlay(
                    frame, detections, fps, safe_direction, None  # No depth overlay
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

            # Overlay feature matching on main frame if in feature_matching mode
            if self.viz_mode == 'feature_matching' and slam_result is not None and self.slam:
                annotated_frame = self._overlay_feature_matching(annotated_frame, slam_result)

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

            return annotated_frame, detections, guidance, audio_signal, audio_message, depth_map, slam_result, trajectory_result, stair_result
        
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            import traceback
            traceback.print_exc()  # Print the full exception trace
            return None

    def _generate_enhanced_guidance(self, detections: List[Dict], nav_summary: Dict, scene_analysis: Optional[Dict], slam_result: Optional[Dict] = None) -> Dict:
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
                        # Find closest danger, handling None depth (treat as 1.0)
                        closest_danger = min(danger_objects, key=lambda x: x.get('depth') if x.get('depth') is not None else 1.0)
                        depth = closest_danger.get('depth', 0)

                        # Handle None depth
                        if depth is None:
                            depth = 1.0  # Assume close

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
                    closest_caution = min(caution_objects, key=lambda x: x.get('depth') if x.get('depth') is not None else 10)
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
            return self._generate_fast_guidance(detections, nav_summary, slam_result)

        except Exception as e:
            self.logger.error(f"Enhanced guidance generation error: {e}")
            import traceback
            traceback.print_exc()
            return self._generate_fast_guidance(detections, nav_summary, slam_result)

    def _create_feature_matching_view(self, frame: np.ndarray, slam_result: Optional[Dict]) -> Optional[np.ndarray]:
        """
        Create a lightweight feature matching visualization without the full map.

        Args:
            frame: Current frame
            slam_result: SLAM tracking result with feature info

        Returns:
            Visualization image or None
        """
        if slam_result is None or not slam_result.get('is_initialized', False):
            return None

        try:
            # Create a blank canvas
            h, w = frame.shape[:2]
            viz_height = 400
            viz_width = 600
            viz = np.zeros((viz_height, viz_width, 3), dtype=np.uint8)

            # Get feature information from SLAM result
            num_matches = slam_result.get('num_matches', 0)
            num_map_points = slam_result.get('num_map_points', 0)
            tracking_quality = slam_result.get('tracking_quality', 0.0)
            is_keyframe = slam_result.get('is_keyframe', False)

            # Title
            cv2.putText(viz, "SLAM Feature Tracking", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Status indicator
            if tracking_quality > 0.7:
                status_color = (0, 255, 0)  # Green - good
                status_text = "TRACKING: GOOD"
            elif tracking_quality > 0.4:
                status_color = (0, 165, 255)  # Orange - ok
                status_text = "TRACKING: OK"
            else:
                status_color = (0, 0, 255)  # Red - poor
                status_text = "TRACKING: POOR"

            cv2.putText(viz, status_text, (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # Feature statistics
            y_offset = 110
            line_height = 30

            stats = [
                f"Matched Points: {num_matches}",
                f"Map Points: {num_map_points}",
                f"Quality: {tracking_quality:.2f}",
                f"Keyframe: {'YES' if is_keyframe else 'NO'}"
            ]

            for i, stat in enumerate(stats):
                cv2.putText(viz, stat, (20, y_offset + i * line_height),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Visual indicator - feature count bar
            bar_x = 20
            bar_y = 260
            bar_width = 560
            bar_height = 30

            # Background bar
            cv2.rectangle(viz, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         (50, 50, 50), -1)

            # Filled bar based on matches (assume 200 matches is full)
            max_matches = 200
            fill_width = int((min(num_matches, max_matches) / max_matches) * bar_width)

            # Color based on quality
            if num_matches > 100:
                bar_color = (0, 255, 0)  # Green
            elif num_matches > 50:
                bar_color = (0, 165, 255)  # Orange
            else:
                bar_color = (0, 0, 255)  # Red

            if fill_width > 0:
                cv2.rectangle(viz, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                             bar_color, -1)

            # Bar label
            cv2.putText(viz, f"{num_matches} / {max_matches} features",
                       (bar_x + 10, bar_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Instructions
            cv2.putText(viz, "Press 'q' to quit", (20, viz_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

            return viz

        except Exception as e:
            self.logger.error(f"Error creating feature view: {e}")
            return None

    def _overlay_feature_matching(self, frame: np.ndarray, slam_result: Optional[Dict]) -> np.ndarray:
        """
        Overlay feature matching visualization directly on the main frame.
        Similar to the reference image: shows side-by-side frames with matched features connected.

        Args:
            frame: Current annotated frame
            slam_result: SLAM tracking result with feature info

        Returns:
            Frame with feature matching overlay
        """
        try:
            if not self.slam:
                return frame

            # Get feature matching image from SLAM
            feature_match_img = self.slam.get_feature_matching_image()
            if feature_match_img is None or feature_match_img.size == 0:
                return frame

            # Get frame dimensions
            h, w = frame.shape[:2]

            # Resize feature matching image to fit in a corner or overlay
            # We'll create a side-by-side view with the current frame and reference frame
            # Resize to about 40% of frame width for side-by-side display
            overlay_width = int(w * 0.4)
            overlay_height = int(overlay_width * feature_match_img.shape[0] / feature_match_img.shape[1])
            
            # Ensure overlay doesn't exceed frame height
            if overlay_height > h * 0.6:
                overlay_height = int(h * 0.6)
                overlay_width = int(overlay_height * feature_match_img.shape[1] / feature_match_img.shape[0])

            # Resize feature matching image
            feature_match_resized = cv2.resize(feature_match_img, (overlay_width, overlay_height),
                                              interpolation=cv2.INTER_LINEAR)

            # Create a copy of the frame to overlay on
            overlay_frame = frame.copy()

            # Position overlay in top-right corner with some padding
            pad = 10
            overlay_x = w - overlay_width - pad
            overlay_y = pad

            # Create semi-transparent background for better visibility
            overlay = overlay_frame[overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+overlay_width].copy()
            cv2.rectangle(overlay, (0, 0), (overlay_width, overlay_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, overlay_frame[overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+overlay_width], 0.7, 0,
                           overlay_frame[overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+overlay_width])

            # Overlay the feature matching image
            overlay_frame[overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+overlay_width] = feature_match_resized

            # Add border for visibility
            cv2.rectangle(overlay_frame, (overlay_x-2, overlay_y-2), 
                          (overlay_x+overlay_width+2, overlay_y+overlay_height+2), (0, 255, 0), 2)

            # Add label
            label = "Feature Matching"
            cv2.putText(overlay_frame, label, (overlay_x, overlay_y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            return overlay_frame

        except Exception as e:
            self.logger.debug(f"Could not overlay feature matching: {e}")
            return frame

    def _generate_fast_guidance(self, detections: List[Dict], nav_summary: Dict, slam_result: Optional[Dict] = None) -> Dict:
        """
        Generate simple, clear guidance for blind users.

        Args:
            detections: List of detected objects
            nav_summary: Navigation summary
            slam_result: SLAM position and tracking info (optional)

        Returns:
            Guidance dictionary with simple messages
        """
        try:
            safe_direction = nav_summary.get('safe_direction', 'forward')

            # SLAM position context (if available)
            position_info = ""
            if slam_result and slam_result.get('is_initialized', False):
                position = slam_result.get('position', [0, 0, 0])
                # Only announce position if user has moved significantly (>1m from origin)
                distance_from_start = np.sqrt(position[0]**2 + position[2]**2)  # X-Z plane distance
                if distance_from_start > 1.0:
                    # Round to nearest meter for simplicity
                    x_dist = int(abs(position[0]))
                    z_dist = int(abs(position[2]))
                    direction_x = "left" if position[0] < 0 else "right"
                    direction_z = "forward" if position[2] > 0 else "back"

                    # Only add position context if it's meaningful (not near origin)
                    if x_dist > 0 or z_dist > 0:
                        position_info = f" You're {z_dist}m {direction_z}, {x_dist}m {direction_x} from start."

            # Immediate danger - very simple
            danger_objects = nav_summary.get('danger_objects', [])
            if danger_objects:
                closest = min(danger_objects, key=lambda x: x.get('depth') if x.get('depth') is not None else 10)
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
                closest = min(caution_objects, key=lambda x: x.get('depth') if x.get('depth') is not None else 10)
                msg = f"{closest['label']} ahead. Slow down{position_info}"
                return {'narrative': msg, 'predictive': '', 'combined': msg}

            # Clear path - add position info if available
            msg = "Path clear"
            if position_info:
                msg += "." + position_info
            return {'narrative': msg, 'predictive': '', 'combined': ''}

        except Exception as e:
            return {'narrative': 'Continue', 'predictive': '', 'combined': 'Continue'}
    

    def _display_terminal_info(self, fps, detections, total_time, slam_result, trajectory_result):
        """Display information in the terminal using Rich."""
        if not RICH_AVAILABLE:
            return  # Skip if Rich not available
            
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
        
        # Danger/Caution counts (handle None depth)
        danger_objects = [d for d in detections if d.get('depth') is not None and d.get('depth', 10) < self.danger_distance]
        caution_objects = [d for d in detections if d.get('depth') is not None and self.danger_distance <= d.get('depth', 10) < self.caution_distance]
        
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
            closest = min(detections, key=lambda x: x.get('depth') if x.get('depth') is not None else 10)
            # Handle None depth safely
            closest_depth = closest.get('depth')
            if closest_depth is not None:
                closest_text = Text(f"{closest['label']} {closest_depth:.1f}m", style="bold")
                
                # Color code based on distance
                if closest_depth < self.danger_distance:
                    closest_text.stylize("red", 0, len(closest_text))
                elif closest_depth < self.caution_distance:
                    closest_text.stylize("yellow", 0, len(closest_text))
                else:
                    closest_text.stylize("green", 0, len(closest_text))
                    
                table.add_row("Closest Object", closest_text.plain)
            else:
                # Unknown distance
                closest_text = Text(f"{closest['label']} (unknown)", style="bold yellow")
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


    def run(self, display: bool = True, save_video: bool = False, separate_slam: bool = False, show_features: bool = False):
        """
        Main run loop.

        Args:
            display: Show video display
            save_video: Save output video
            separate_slam: Show SLAM map in a separate window
            show_features: Show feature matching window (lightweight, no map)
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

                # Resize frame if needed (for high-resolution videos)
                if self.need_frame_resize and frame is not None:
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)

                # Don't flip - use video as-is (mirroring was causing confusion)

                self.frame_count += 1
                
                # Skip frames to reduce FPS for high-speed videos
                if self.video_fps_skip_rate > 1:
                    if self.frame_count % int(self.video_fps_skip_rate) != 0:
                        continue  # Skip this frame

                self.logger.debug("Frame read successfully. Processing frame.")

                # Process frame
                result = self.process_frame(frame)
                if result is None or len(result) != 9:
                    self.logger.warning("Skipping frame due to invalid processing result.")
                    continue
                annotated_frame, detections, guidance, audio_signal, audio_message, depth_map, slam_result, trajectory_result, stair_result = result
                
                # Get FPS and total_time after processing (from perf_monitor stats)
                stats = self.perf_monitor.get_stats()
                fps = stats.get('fps', 0)
                total_time = stats.get('avg_frame_time_ms', 0)

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
                    if self.detection_pipeline is not None:
                        nav_summary = self.detection_pipeline.get_navigation_summary(detections)
                    else:
                        nav_summary = {'path_clear': True, 'danger_objects': [], 'caution_objects': []}
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

                # HIGHEST PRIORITY: Stair/curb detection warnings (prevent falls)
                if stair_result and stair_result.get('drop_detected', False):
                    warning_level = stair_result.get('warning_level', 'safe')
                    hazard_type = stair_result.get('hazard_type', 'hazard')
                    distance = stair_result.get('distance_to_hazard')

                    if warning_level in ['danger', 'caution']:
                        if (current_time - self.last_audio_time) > 0.3 and not self.audio_manager.is_speaking:
                            # Immediate, urgent warning for stairs/curbs
                            if hazard_type == 'stair_down':
                                msg = f"STOP! Stairs going down ahead!"
                            elif hazard_type == 'stair_up':
                                msg = f"Stairs going up ahead. Use handrail."
                            elif hazard_type == 'curb':
                                msg = f"STOP! Curb ahead!"
                            elif hazard_type == 'drop':
                                msg = f"STOP! Drop detected ahead!"
                            else:
                                msg = f"STOP! {hazard_type.replace('_', ' ')} ahead!"

                            if distance is not None:
                                msg += f" {distance:.1f} meters."

                            self.logger.error(f"🚨 FALL HAZARD: \"{msg}\"")
                            self.audio_manager.speak(msg, priority=True)
                            self.last_audio_time = current_time

                # Check for uncertain depth objects (no depth measurement available)
                uncertain_objects = [d for d in detections if d.get('depth_uncertain', False)]
                has_uncertain_depth = len(uncertain_objects) > 0

                # Check for danger zone objects (< 1m) - PRIORITY ALERT
                danger_objects = [d for d in detections if d.get('depth', None) is not None and d.get('depth', 10) < self.danger_distance]
                has_danger = len(danger_objects) > 0

                # Determine appropriate interval based on danger level
                active_interval = self.danger_audio_interval if (has_danger or has_uncertain_depth) else self.audio_interval

                # CRITICAL: Uncertain depth warning (depth measurement unavailable)
                if has_uncertain_depth:
                    if (current_time - self.last_audio_time) > self.danger_audio_interval and not self.audio_manager.is_speaking:
                        # Warn about objects with unknown distance
                        uncertain_obj = uncertain_objects[0]
                        label = uncertain_obj['label']

                        # Determine direction
                        center = uncertain_obj.get('center', [160, 160])
                        if center[0] < 106:
                            direction = "on your left"
                        elif center[0] > 213:
                            direction = "on your right"
                        else:
                            direction = "ahead"

                        msg = f"Caution! {label} {direction}. Distance unknown, use care"
                        self.logger.warning(f"⚠️  UNCERTAIN DEPTH: \"{msg}\"")
                        self.audio_manager.speak(msg, priority=True)
                        self.last_audio_time = current_time

                # DANGER ZONE - Priority alert (known depth, too close)
                elif has_danger:
                    if (current_time - self.last_audio_time) > self.danger_audio_interval and not self.audio_manager.is_speaking:
                        closest_danger = min(danger_objects, key=lambda x: x.get('depth') if x.get('depth') is not None else 10)
                        depth = closest_danger.get('depth', 0)

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
                    closest = min(detections, key=lambda x: x.get('depth') if x.get('depth') is not None else 10)
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
                        depth = det.get('depth', 10)
                        if depth is not None and depth < 2.5:
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
                
                # Display terminal info window with TTS text (less frequent for performance)
                if self.frame_count % 60 == 0:  # Update every 60 frames (~2 seconds)
                    self._display_terminal_info(fps, detections, total_time, slam_result, trajectory_result)
                
                # Clean robot-style display - only show main camera view
                if display:
                    # In feature_matching mode, show feature matching as main content
                    if self.viz_mode == 'feature_matching' and self.slam:
                        feature_match_img = self.slam.get_feature_matching_image()
                        if feature_match_img is not None and feature_match_img.size > 0:
                            # Convert RGB to BGR if needed (draw_feature_matches returns RGB)
                            if feature_match_img.shape[2] == 3:
                                # Check if it's RGB (opencv uses BGR)
                                # draw_feature_matches should return RGB, but we'll check
                                feature_match_img_bgr = cv2.cvtColor(feature_match_img, cv2.COLOR_RGB2BGR) if len(feature_match_img.shape) == 3 else feature_match_img
                            else:
                                feature_match_img_bgr = feature_match_img
                            
                            # Resize to fit display - make it larger for better visibility
                            h, w = feature_match_img_bgr.shape[:2]
                            display_width = 1200  # Wider for side-by-side view
                            display_height = int(display_width * h / w)
                            if display_height > 800:
                                display_height = 800
                                display_width = int(display_height * w / h)
                            feature_match_display = cv2.resize(feature_match_img_bgr, (display_width, display_height),
                                                              interpolation=cv2.INTER_LINEAR)
                            cv2.imshow('OrbyGlasses', feature_match_display)
                        else:
                            # Fallback to regular view if no feature matching available
                            display_width = 480
                            display_height = 360
                            camera_display = cv2.resize(annotated_frame, (display_width, display_height),
                                                       interpolation=cv2.INTER_LINEAR)
                            cv2.imshow('OrbyGlasses', camera_display)
                    else:
                        # Main camera view - resize to smaller size for cleaner desktop
                        display_width = 480
                        display_height = 360
                        camera_display = cv2.resize(annotated_frame, (display_width, display_height),
                                                   interpolation=cv2.INTER_LINEAR)
                        cv2.imshow('OrbyGlasses', camera_display)

                    # Show feature matching view if requested (for other modes)
                    if show_features and self.slam and self.viz_mode != 'feature_matching':
                        # Try to get actual feature matching image from pyslam
                        feature_match_img = self.slam.get_feature_matching_image()
                        if feature_match_img is not None and feature_match_img.size > 0:
                            # Resize if too large
                            h, w = feature_match_img.shape[:2]
                            if w > 1200:
                                scale = 1200 / w
                                new_w, new_h = int(w * scale), int(h * scale)
                                feature_match_img = cv2.resize(feature_match_img, (new_w, new_h))
                            cv2.imshow('Feature Matching', feature_match_img)
                        else:
                            # Fallback to statistics view if no matching image available
                            feature_viz = self._create_feature_matching_view(frame, slam_result)
                            if feature_viz is not None:
                                cv2.imshow('Feature Tracking Stats', feature_viz)

                    # pySLAM shows its own windows (3D viewer, trajectory, features)
                    # No need for custom SLAM or depth windows unless separate_slam is enabled

                    # Show 3D Occupancy Grid if enabled
                    if self.occupancy_grid_enabled and self.occupancy_grid is not None:
                        occupancy_grid_image = self.occupancy_grid.visualize_3d_interactive(slam_result['position'] if slam_result else None)
                        cv2.imshow('Occupancy Grid', occupancy_grid_image)

                # Save video
                if video_writer:
                    video_writer.write(annotated_frame)

                # Check for quit and handle keyboard controls (minimal delay)
                key = cv2.waitKey(1) & 0xFF  # Reduced from 10ms to 1ms
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
    parser.add_argument('--mode', type=str, choices=['full_slam', 'vo', 'features', 'basic', 'feature_matching'],
                        help='Visualization mode: full_slam (all windows), vo (visual odometry), features (feature matching window), basic (main window only), feature_matching (feature matching in main window only)')
    parser.add_argument('--separate-slam', action='store_true',
                        help='[DEPRECATED] Use --mode full_slam instead')
    parser.add_argument('--show-features', action='store_true',
                        help='[DEPRECATED] Use --mode features instead')
    parser.add_argument('--video', type=str, default=None,
                       help='Path to video file (overrides camera.source in config)')

    args = parser.parse_args()

    # Determine mode (handle deprecated flags)
    mode = args.mode
    if mode is None:
        # Check deprecated flags
        if args.show_features:
            mode = 'features'
            print("⚠️  --show-features is deprecated, use --mode features")
        elif args.separate_slam:
            mode = 'full_slam'
            print("⚠️  --separate-slam is deprecated, use --mode full_slam")
        else:
            # Use config file default
            mode = None

    # Initialize system
    system = OrbyGlasses(config_path=args.config, video_source=args.video, mode=mode)

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
        # Determine flags based on mode
        show_features = (mode == 'features')
        separate_slam = (mode == 'full_slam')
        # Note: feature_matching mode shows feature matching in main window only,
        # so we don't set show_features or separate_slam

        system.run(display=not args.no_display, save_video=args.save_video,
                  separate_slam=separate_slam, show_features=show_features)


if __name__ == "__main__":
    main()
