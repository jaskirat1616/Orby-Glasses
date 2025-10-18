"""
OrbyGlasses - Main Entry Point
Bio-mimetic navigation engine for visually impaired users.
"""

import sys
import os
import cv2
import numpy as np
import argparse
import time
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    ConfigManager, Logger, AudioManager, FrameProcessor,
    DataLogger, PerformanceMonitor, ensure_directories, check_device
)
from detection import DetectionPipeline
from echolocation import AudioCueGenerator
from narrative import ContextualAssistant
from prediction import PathPlanner


class OrbyGlasses:
    """
    Main OrbyGlasses application.
    Integrates all components for real-time navigation assistance.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize OrbyGlasses system.

        Args:
            config_path: Path to configuration file
        """
        # Ensure directories exist
        ensure_directories()

        # Load configuration
        self.config = ConfigManager(config_path)

        # Initialize logger
        self.logger = Logger(log_file=self.config.get('logging.log_file', 'data/logs/orbyglass.log'))
        self.logger.info("=" * 50)
        self.logger.info("OrbyGlasses - Bio-Mimetic Navigation System")
        self.logger.info("=" * 50)

        # Check device
        device = check_device()
        self.logger.info(f"Running on device: {device}")

        # Initialize components
        self.logger.info("Initializing components...")

        self.audio_manager = AudioManager(self.config)
        self.detection_pipeline = DetectionPipeline(self.config)
        self.audio_cue_generator = AudioCueGenerator(self.config)
        self.contextual_assistant = ContextualAssistant(self.config)
        self.path_planner = PathPlanner(self.config)

        # Data logging
        self.data_logger = DataLogger()

        # Performance monitoring
        self.perf_monitor = PerformanceMonitor()

        # Camera setup
        self.camera = None
        self.frame_width = self.config.get('camera.width', 640)
        self.frame_height = self.config.get('camera.height', 480)

        # State
        self.running = False
        self.frame_count = 0
        self.last_audio_time = 0
        self.audio_interval = 2.0  # seconds between audio updates

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

    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame through the entire pipeline.

        Args:
            frame: Input frame

        Returns:
            Tuple of (annotated_frame, detections, guidance)
        """
        # Start timer
        self.perf_monitor.start_timer('total')

        # Detection and depth
        self.perf_monitor.start_timer('detection')
        detections, depth_map = self.detection_pipeline.process_frame(frame)
        det_time = self.perf_monitor.stop_timer('detection')

        # Get navigation summary
        nav_summary = self.detection_pipeline.get_navigation_summary(detections)

        # Path planning (RL prediction)
        self.perf_monitor.start_timer('prediction')
        path_plan = self.path_planner.plan_path(detections, nav_summary)
        pred_time = self.perf_monitor.stop_timer('prediction')

        # Generate narrative guidance
        self.perf_monitor.start_timer('narrative')
        guidance = self.contextual_assistant.get_guidance(
            detections,
            frame=frame,
            navigation_summary=nav_summary,
            predicted_path=path_plan
        )
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

        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Det: {det_time:.1f}ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"Objects: {len(detections)}", (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Log frame time
        self.perf_monitor.log_frame_time(total_time)

        return annotated_frame, detections, guidance, audio_signal, audio_message

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

        # Welcome message
        self.audio_manager.speak("OrbyGlasses navigation system activated", priority=True)

        self.logger.info("Starting main loop... Press 'q' to quit")

        try:
            while self.running:
                # Capture frame
                ret, frame = self.camera.read()

                if not ret:
                    self.logger.warning("Failed to capture frame")
                    break

                self.frame_count += 1

                # Process frame
                annotated_frame, detections, guidance, audio_signal, audio_message = \
                    self.process_frame(frame)

                # Audio output (rate-limited)
                current_time = time.time()
                if current_time - self.last_audio_time > self.audio_interval:
                    # Play echolocation audio
                    if self.config.get('audio.echolocation_enabled', True):
                        self.audio_manager.play_sound(audio_signal.T)

                    # Speak guidance
                    if guidance['combined']:
                        self.audio_manager.speak(guidance['combined'])

                    self.last_audio_time = current_time

                    # Log this frame
                    self.data_logger.log_detection(
                        self.frame_count,
                        detections
                    )

                # Display
                if display:
                    # Add guidance text to frame
                    y_offset = self.frame_height - 60
                    text = guidance.get('narrative', 'Processing...')
                    # Wrap text if too long
                    if len(text) > 50:
                        text = text[:47] + "..."

                    cv2.rectangle(annotated_frame, (0, y_offset - 5),
                                (self.frame_width, self.frame_height),
                                (0, 0, 0), -1)
                    cv2.putText(annotated_frame, text, (10, y_offset + 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                    cv2.imshow('OrbyGlasses', annotated_frame)

                # Save video
                if video_writer:
                    video_writer.write(annotated_frame)

                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                emergency_key = self.config.get('safety.emergency_stop_key', 'q')

                if key == ord(emergency_key):
                    self.logger.info("Emergency stop activated")
                    self.audio_manager.speak("Navigation stopped", priority=True)
                    break

                # Performance stats every 100 frames
                if self.frame_count % 100 == 0:
                    stats = self.perf_monitor.get_stats()
                    self.logger.info(f"Performance: {stats}")

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
