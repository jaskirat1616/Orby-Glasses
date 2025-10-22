"""
OrbyGlasses - Optimized Main Entry Point
High-performance bio-mimetic navigation engine for visually impaired users.
Optimized for speed, accuracy, and real-time performance.
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

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    ConfigManager, Logger, AudioManager, FrameProcessor,
    DataLogger, PerformanceMonitor, ensure_directories, check_device
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.box import ROUNDED
from detection import DetectionPipeline
from echolocation import AudioCueGenerator
from narrative import ContextualAssistant
from prediction import PathPlanner
from mapping3d import Mapper3D
from conversation import ConversationManager
from slam import MonocularSLAM
from slam_system import SLAMSystem
from indoor_navigation import IndoorNavigator
from trajectory_prediction import TrajectoryPredictionSystem
from occupancy_grid_3d import OccupancyGrid3D
from voxel_map import VoxelMap
from point_cloud_viewer import PointCloudViewer
from movement_visualizer import MovementVisualizer
from coordinate_transformer import CoordinateTransformer
from scene_understanding import EnhancedSceneProcessor


class OrbyGlasses:
    """
    Optimized OrbyGlasses application.
    High-performance navigation assistance for visually impaired users.
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
        if self.slam_enabled:
            # Use the new SLAM system by default
            self.logger.info("Initializing SLAM system...")
            self.slam = SLAMSystem(self.config)
            self.indoor_nav_enabled = self.config.get('indoor_navigation.enabled', False)
            if self.indoor_nav_enabled:
                self.indoor_navigator = IndoorNavigator(self.slam, self.config)
                self.logger.info("✓ SLAM and Indoor Navigation enabled")
            else:
                self.indoor_navigator = None
                self.logger.info("✓ SLAM enabled (indoor navigation disabled)")
        else:
            self.slam = None
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

        # Data logging
        self.data_logger = DataLogger()

        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()

        # Camera setup
        self.camera = None
        self.frame_width = self.config.get('camera.width', 640)
        self.frame_height = self.config.get('camera.height', 480)

        # Optimized state management
        self.running = False
        self.frame_count = 0
        self.last_audio_time = 0
        self.audio_interval = self.config.get('performance.audio_update_interval', 2.0)
        self.danger_audio_interval = self.config.get('performance.danger_audio_interval', 0.8)
        self.skip_depth_frames = self.config.get('performance.depth_skip_frames', 2)
        self.last_depth_map = None  # Cache last depth map
        
        # Performance optimizations
        self.enable_multithreading = self.config.get('performance.enable_multithreading', True)
        self.cache_depth_maps = self.config.get('performance.cache_depth_maps', True)
        self.max_detections = self.config.get('performance.max_detections', 5)
        
        # Thread pool for parallel processing
        if self.enable_multithreading:
            self.executor = ThreadPoolExecutor(max_workers=3)
        
        # Frame processing cache
        self.detection_cache = {}
        self.last_detection_time = 0
        
        # Simple memory system for blind users
        self.location_memory = {}  # Remember places they've been
        self.obstacle_memory = {}  # Remember common obstacles
        self.path_memory = []      # Remember good paths

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

            # Optimized depth estimation with smart caching
            self.perf_monitor.start_timer('depth')
            if self.frame_count % (self.skip_depth_frames + 1) == 0 or self.last_depth_map is None:
                depth_map = self.detection_pipeline.depth_estimator.estimate_depth(frame)
                self.last_depth_map = depth_map
            else:
                depth_map = self.last_depth_map
            depth_time = self.perf_monitor.stop_timer('depth')

            # Add depth to detections
            if depth_map is not None:
                for detection in detections:
                    bbox = detection['bbox']
                    depth = self.detection_pipeline.depth_estimator.get_depth_at_bbox(depth_map, bbox)
                    detection['depth'] = depth
                    detection['is_danger'] = depth < self.detection_pipeline.min_safe_distance
            else:
                for detection in detections:
                    detection['depth'] = 0.0
                    detection['is_danger'] = False

            # Get navigation summary
            nav_summary = self.detection_pipeline.get_navigation_summary(detections)
            
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
                slam_result = self.slam.process_frame(frame, depth_map)
                slam_time = self.perf_monitor.stop_timer('slam')
                if self.frame_count % 50 == 0:  # Log every 50 frames
                    if slam_result:
                        self.logger.debug(f"SLAM completed in {slam_time:.3f}s, pos:({slam_result['position'][0]:.2f}, {slam_result['position'][1]:.2f}), q:{slam_result['tracking_quality']:.2f}, pts:{slam_result['num_map_points']}")

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

            # Optimized narrative generation with smart caching
            self.perf_monitor.start_timer('narrative')
            should_generate = current_time - self.last_audio_time > self.audio_interval
            
            if should_generate:
                # Generate enhanced guidance with VLM scene understanding
                guidance = self._generate_enhanced_guidance(detections, nav_summary, scene_analysis)
            else:
                # Use cached guidance or simple fallback
                guidance = {'narrative': '', 'predictive': '', 'combined': ''}
            narr_time = self.perf_monitor.stop_timer('narrative')

            # Generate audio cues
            self.perf_monitor.start_timer('audio')
            audio_signal, audio_message = self.audio_cue_generator.generate_cues(
                detections,
                (self.frame_height, self.frame_width)
            )
            audio_time = self.perf_monitor.stop_timer('audio')

            # Annotate frame
            annotated_frame = FrameProcessor.annotate_detections(frame, detections)

            # Add performance info to frame
            total_time = self.perf_monitor.stop_timer('total')
            fps = self.perf_monitor.get_avg_fps()

            # Check for danger zone
            danger_objects = [d for d in detections if d.get('depth', 10) < self.danger_distance]
            caution_objects = [d for d in detections if self.danger_distance <= d.get('depth', 10) < self.caution_distance]

            # Performance overlay with danger-aware background
            overlay = annotated_frame.copy()
            bg_color = (0, 0, 100) if danger_objects else (0, 0, 0)  # Red tint if danger
            cv2.rectangle(overlay, (5, 5), (250, 135), bg_color, -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)

            # FPS indicator
            fps_color = (0, 255, 0) if fps > 10 else (0, 165, 255) if fps > 5 else (0, 0, 255)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, fps_color, 2)

            # Danger/Caution/Safe counts
            cv2.putText(annotated_frame, f"Danger: {len(danger_objects)} | Caution: {len(caution_objects)}", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Closest object with distance
            if detections:
                closest = min(detections, key=lambda x: x.get('depth', 10))
                dist_color = (0, 0, 255) if closest['depth'] < self.danger_distance else \
                            (0, 165, 255) if closest['depth'] < self.caution_distance else (0, 255, 0)
                cv2.putText(annotated_frame, f"Closest: {closest['label']} {closest['depth']:.1f}m", (10, 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_color, 1)

            # Processing time
            cv2.putText(annotated_frame, f"Process: {total_time:.0f}ms", (10, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            # Status indicator
            if danger_objects:
                status_text = "⚠ DANGER"
                status_color = (0, 0, 255)
            elif caution_objects:
                status_text = "⚠ CAUTION"
                status_color = (0, 165, 255)
            elif detections:
                status_text = "SAFE"
                status_color = (0, 255, 0)
            else:
                status_text = "CLEAR"
                status_color = (0, 255, 0)

            cv2.putText(annotated_frame, status_text, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # Add SLAM info if enabled
            if slam_result is not None:
                position = slam_result['position']
                quality = slam_result['tracking_quality']

                # SLAM status overlay
                slam_color = (0, 255, 0) if quality > 0.7 else (0, 165, 255) if quality > 0.4 else (0, 0, 255)
                cv2.putText(annotated_frame, f"SLAM: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})",
                           (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.4, slam_color, 1)
                cv2.putText(annotated_frame, f"Quality: {quality:.2f} | Points: {slam_result['num_map_points']}",
                           (10, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

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
        Generate enhanced guidance using VLM scene understanding.
        
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
                navigation_guidance = scene_analysis.get('navigation_guidance', '')
                
                # Check for immediate danger
                danger_objects = nav_summary.get('danger_objects', [])
                if danger_objects:
                    closest_danger = min(danger_objects, key=lambda x: x.get('depth', 10))
                    return {
                        'narrative': f"⚠️ Stop! {closest_danger['label']} ahead - {navigation_guidance}",
                        'predictive': '',
                        'combined': f"⚠️ Stop! {closest_danger['label']} ahead - {navigation_guidance}"
                    }
                
                # Use VLM guidance for other situations
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
            return self._generate_fast_guidance(detections, nav_summary)

    def _generate_fast_guidance(self, detections: List[Dict], nav_summary: Dict) -> Dict:
        """
        Generate clear, helpful guidance for blind navigation.
        
        Args:
            detections: List of detected objects
            nav_summary: Navigation summary
            
        Returns:
            Guidance dictionary with clear messages
        """
        try:
            # Check for immediate danger
            danger_objects = nav_summary.get('danger_objects', [])
            if danger_objects:
                closest_danger = min(danger_objects, key=lambda x: x.get('depth', 10))
                # Give clear direction guidance
                direction = self._get_direction_guidance(closest_danger)
                return {
                    'narrative': f"Stop! {closest_danger['label']} ahead. {direction}",
                    'predictive': '',
                    'combined': f"Stop! {closest_danger['label']} ahead. {direction}"
                }
            
            # Check for caution objects
            caution_objects = nav_summary.get('caution_objects', [])
            if caution_objects:
                closest_caution = min(caution_objects, key=lambda x: x.get('depth', 10))
                direction = self._get_direction_guidance(closest_caution)
                return {
                    'narrative': f"Caution: {closest_caution['label']} {closest_caution['depth']:.1f}m. {direction}",
                    'predictive': '',
                    'combined': f"Caution: {closest_caution['label']} {closest_caution['depth']:.1f}m. {direction}"
                }
            
            # Path clear with helpful info
            if nav_summary.get('path_clear', True):
                if detections:
                    # Tell them what's around them
                    nearby_objects = [d for d in detections if d.get('depth', 10) < 5.0]
                    if nearby_objects:
                        object_names = [d['label'] for d in nearby_objects[:3]]
                        return {
                            'narrative': f"Path clear. You have {', '.join(object_names)} nearby",
                            'predictive': '',
                            'combined': f"Path clear. You have {', '.join(object_names)} nearby"
                        }
                
                return {
                    'narrative': 'Path clear, continue forward',
                    'predictive': '',
                    'combined': 'Path clear, continue forward'
                }
            
            # Default fallback
            return {
                'narrative': 'Continue forward',
                'predictive': '',
                'combined': 'Continue forward'
            }
            
        except Exception as e:
            self.logger.error(f"Guidance generation error: {e}")
            return {
                'narrative': 'Continue forward',
                'predictive': '',
                'combined': 'Continue forward'
            }
    
    def _get_direction_guidance(self, detection: Dict) -> str:
        """Get helpful direction guidance for avoiding obstacles."""
        center = detection.get('center', [160, 160])
        x_center = center[0]
        
        # Simple left/right guidance
        if x_center < 160:  # Left side of frame
            return "Step right to avoid"
        elif x_center > 320:  # Right side of frame
            return "Step left to avoid"
        else:  # Center
            return "Step left or right to avoid"

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


    def run(self, display: bool = True, save_video: bool = False):
        """
        Main run loop.

        Args:
            display: Show video display
            save_video: Save output video
        """
        if not self.initialize_camera():
            self.logger.error("Cannot start without camera")
            return

        self.running = True

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
        self.logger.info(f"Audio interval: {self.audio_interval}s | Depth: every {self.skip_depth_frames + 1} frames")
        if self.slam_enabled:
            self.logger.info("🗺️ SLAM enabled")
        if self.conversation_enabled:
            activation = self.config.get('conversation.activation_phrase', 'hey glasses')
            self.logger.info(f"💬 Voice: '{activation}'")
        self.logger.info(f"Press '{self.config.get('safety.emergency_stop_key', 'q')}' to quit")

        try:
            while self.running:
                # Capture frame
                ret, frame = self.camera.read()

                if not ret:
                    self.logger.warning("Failed to capture frame")
                    break

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
                time_since_last = current_time - self.last_audio_time

                # Check for danger zone objects (< 1m) - PRIORITY ALERT
                danger_objects = [d for d in detections if d.get('depth', 10) < self.danger_distance]
                has_danger = len(danger_objects) > 0

                # Determine appropriate interval based on danger level
                active_interval = self.danger_audio_interval if has_danger else self.audio_interval

                # Only generate and queue new audio if interval passed and not speaking
                if time_since_last > active_interval and not self.audio_manager.is_speaking:
                    # DANGER ZONE - Priority alert
                    if has_danger:
                        closest_danger = min(danger_objects, key=lambda x: x['depth'])
                        depth = closest_danger['depth']

                        # Use relatable distance terms
                        if depth < 0.3:
                            distance_term = "immediately ahead"
                        elif depth < 0.5:
                            distance_term = "arm's length away"
                        else:
                            distance_term = "one step away"

                        # Determine direction
                        center = closest_danger.get('center', [160, 160])
                        if center[0] < 106:
                            direction = "on your left, step right"
                        elif center[0] > 213:
                            direction = "on your right, step left"
                        else:
                            direction = "straight ahead, step aside"

                        msg = f"{closest_danger['label']} {distance_term} {direction}"
                        self.logger.error(f"🚨 DANGER ALERT: \"{msg}\"")
                        self.audio_manager.speak(msg, priority=True)

                    # Use indoor navigation guidance if available, otherwise use Ollama-generated narrative
                    elif indoor_guidance:
                        msg = indoor_guidance
                        self.logger.info(f"🔊 Indoor Navigation: \"{msg}\"")
                        self.audio_manager.speak(msg, priority=False)
                    elif guidance.get('combined'):
                        msg = guidance['combined']
                        self.logger.info(f"🔊 Audio: \"{msg}\"")
                        self.audio_manager.speak(msg, priority=False)

                    # Fallback to simple message
                    elif len(detections) > 0:
                        closest = min(detections, key=lambda x: x.get('depth', 10))
                        msg = f"{closest['label']} at {closest['depth']:.1f} meters"
                        self.logger.info(f"🔊 Audio: \"{msg}\"")
                        self.audio_manager.speak(msg, priority=False)

                    # Path clear
                    else:
                        msg = "Path clear"
                        self.logger.info(f"🔊 Audio: \"{msg}\"")
                        self.audio_manager.speak(msg, priority=False)

                    # Update timer
                    self.last_audio_time = current_time

                # Optimized display
                if display:
                    # Display terminal info less frequently for performance
                    if self.frame_count % 10 == 0:  # Show terminal info every 10 frames
                        total_time = self.perf_monitor.get_stats().get('avg_frame_time_ms', 0)
                        fps = self.perf_monitor.get_avg_fps()
                        self._display_terminal_info(fps, detections, total_time, slam_result, trajectory_result)

                    # Optimized display size
                    display_size = (416, 416)  # Match input size for no scaling

                    # Show main window (no resize needed if same size)
                    if annotated_frame.shape[:2] != display_size:
                        display_frame = cv2.resize(annotated_frame, display_size)
                    else:
                        display_frame = annotated_frame
                    cv2.imshow('OrbyGlasses', display_frame)

                    # Setup mouse callbacks for windows (first time only)
                    if self.frame_count == 1:
                        # Voxel grid mouse callback
                        if self.occupancy_grid_enabled and self.occupancy_grid is not None:
                            def occ_mouse_callback(event, x, y, flags, param):
                                self.occupancy_grid.handle_mouse_events(event, x, y, flags, param)
                            cv2.setMouseCallback('Voxel Grid Map', occ_mouse_callback)

                        # Point cloud mouse callback
                        if self.point_cloud_enabled and self.point_cloud is not None:
                            def pc_mouse_callback(event, x, y, flags, param):
                                self.point_cloud.handle_mouse_events(event, x, y, flags, param)
                            cv2.setMouseCallback('3D Point Cloud', pc_mouse_callback)

                    # Show depth map in separate smaller window (only when freshly calculated)
                    if depth_map is not None and self.frame_count % (self.skip_depth_frames + 1) == 0:
                        # Convert depth map to colormap for visualization
                        depth_colored = cv2.applyColorMap(
                            (depth_map * 255).astype(np.uint8),
                            cv2.COLORMAP_MAGMA
                        )
                        # Resize depth map to standard size
                        depth_display = cv2.resize(depth_colored, display_size)
                        cv2.imshow('Depth Map', depth_display)

                    # Show SLAM visualization if enabled
                    if slam_result is not None and self.config.get('slam.visualize', False):
                        slam_vis = self.slam.visualize_tracking(frame, slam_result)
                        slam_display = cv2.resize(slam_vis, display_size)
                        cv2.imshow('SLAM Tracking', slam_display)
                        
                        # Show 3D real-time SLAM map if mapping is enabled
                        if self.config.get('mapping3d.enabled', False):
                            # Create a 3D visualization of the SLAM map
                            if hasattr(self.slam, 'visualize_3d_map'):
                                slam_3d_vis = self.slam.visualize_3d_map()
                                if slam_3d_vis is not None:
                                    slam_3d_display = cv2.resize(slam_3d_vis, display_size)
                                    cv2.imshow('SLAM 3D Map', slam_3d_display)

                    # Show 3D Point Cloud visualization if enabled
                    if self.point_cloud_enabled and self.point_cloud is not None:
                        if self.config.get('point_cloud_viewer.visualize', True):
                            camera_pos = None
                            if slam_result is not None:
                                camera_pos = np.array(slam_result['position'])

                            pc_vis = self.point_cloud.visualize(camera_pos)
                            pc_display = cv2.resize(pc_vis, display_size)
                            cv2.imshow('3D Point Cloud', pc_display)

                    # Show 3D Occupancy Grid visualization if enabled
                    if self.occupancy_grid_enabled and self.occupancy_grid is not None:
                        if self.config.get('occupancy_grid_3d.visualize', True):
                            # Get camera position from SLAM if available
                            camera_pos = None
                            if slam_result is not None:
                                camera_pos = np.array(slam_result['position'])

                            # Show interactive voxel grid view
                            occ_vis_3d = self.occupancy_grid.visualize_3d_interactive(camera_pos)
                            occ_display_3d = cv2.resize(occ_vis_3d, display_size)
                            cv2.imshow('Voxel Grid Map', occ_display_3d)

                            # Also show 2D slice in separate window
                            if self.config.get('occupancy_grid_3d.show_2d_slice', False):
                                occ_vis_2d = self.occupancy_grid.visualize_2d_slice(z_height=1.5)
                                occ_display_2d = cv2.resize(occ_vis_2d, display_size)
                                cv2.imshow('2D Occupancy Slice', occ_display_2d)

                    # Show Movement Visualizer if enabled
                    if self.movement_visualizer_enabled and self.movement_visualizer is not None:
                        if self.config.get('movement_visualizer.enabled', True):
                            mv_vis = self.movement_visualizer.visualize()
                            mv_display = cv2.resize(mv_vis, display_size)
                            cv2.imshow('Movement Trajectory', mv_display)

                # Save video
                if video_writer:
                    video_writer.write(annotated_frame)

                # Check for quit and handle keyboard controls
                key = cv2.waitKey(1) & 0xFF
                emergency_key = self.config.get('safety.emergency_stop_key', 'q')

                # Handle occupancy grid controls
                if self.occupancy_grid_enabled and self.occupancy_grid is not None:
                    if self.occupancy_grid.update_view_controls(key):
                        pass  # View updated, will refresh on next frame

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
        system.run(display=not args.no_display, save_video=args.save_video)


if __name__ == "__main__":
    main()
