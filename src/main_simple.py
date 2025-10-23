"""
OrbyGlasses - Simple and Fast Pipeline
Clean, focused navigation system for blind users.
"""

import sys
import os
import cv2
import numpy as np
import time
from typing import Optional, List, Dict

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import ConfigManager, Logger, AudioManager, ensure_directories
from detection import DetectionPipeline
from safety_system import SafetySystem
from audio_priority import AudioPriorityManager
from blind_navigation import BlindNavigationAssistant
from demo_overlay import DemoOverlay


class OrbyGlasses:
    """Simple and fast navigation system."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize system."""
        ensure_directories()

        # Load config
        self.config = ConfigManager(config_path)

        # Logger
        import logging
        log_level = getattr(logging, self.config.get('logging.level', 'INFO').upper())
        self.logger = Logger(log_level=log_level)
        self.logger.info("OrbyGlasses Starting...")

        # Camera settings (need these first!)
        self.camera = None
        self.frame_width = self.config.get('camera.width', 640)
        self.frame_height = self.config.get('camera.height', 480)

        # Core components
        self.audio_manager = AudioManager(self.config)
        self.detection_pipeline = DetectionPipeline(self.config)
        self.safety_system = SafetySystem(
            focal_length=self.config.get('safety.focal_length', 500),
            frame_height=self.frame_height
        )
        self.audio_priority = AudioPriorityManager(
            max_queue_size=self.config.get('audio.max_queue_size', 5),
            min_message_interval=self.config.get('audio.min_message_interval', 0.5)
        )

        # Blind navigation assistant - REAL help for blind users
        self.blind_nav = BlindNavigationAssistant(
            frame_width=self.frame_width,
            frame_height=self.frame_height
        )
        self.logger.info("✓ Blind navigation assistant initialized")

        # Demo overlay for impressive visualizations
        self.demo_overlay = DemoOverlay(
            frame_width=self.frame_width,
            frame_height=self.frame_height
        )
        self.show_demo_overlay = True  # Set to True for demos/presentations

        # SLAM (if enabled)
        self.slam_enabled = self.config.get('slam.enabled', False)
        if self.slam_enabled:
            from slam_system import SLAMSystem
            self.slam = SLAMSystem(self.config)
            self.logger.info("✓ SLAM enabled")
        else:
            self.slam = None

        # Indoor navigation (if enabled)
        self.nav_enabled = self.config.get('indoor_navigation.enabled', False)
        if self.nav_enabled and self.slam:
            from indoor_navigation import IndoorNavigator
            self.navigator = IndoorNavigator(self.slam, self.config)
            self.logger.info("✓ Navigation enabled")
        else:
            self.navigator = None

        # State
        self.frame_count = 0
        self.last_audio_time = 0
        self.last_depth_map = None
        self.running = False

        # Performance tracking
        self.fps_history = []
        self.last_fps_time = time.time()

        self.logger.info("Initialization complete")

    def initialize_camera(self) -> bool:
        """Initialize camera."""
        source = self.config.get('camera.source', 0)
        self.logger.info(f"Opening camera {source}...")

        try:
            self.camera = cv2.VideoCapture(source)
            if not self.camera.isOpened():
                self.logger.error("Camera failed to open")
                return False

            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.get('camera.fps', 30))

            self.logger.info("✓ Camera ready")
            return True

        except Exception as e:
            self.logger.error(f"Camera error: {e}")
            return False

    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Process frame - simple and fast.

        Returns dict with:
        - annotated_frame
        - detections
        - warnings
        - fps
        """
        start_time = time.time()

        try:
            # 1. DETECT OBJECTS
            detections = self.detection_pipeline.detector.detect(frame)

            # 2. ESTIMATE DEPTH (skip some frames for speed)
            skip_frames = self.config.get('performance.depth_skip_frames', 2)
            if self.frame_count % (skip_frames + 1) == 0 or self.last_depth_map is None:
                depth_map = self.detection_pipeline.depth_estimator.estimate_depth(frame)
                self.last_depth_map = depth_map
            else:
                depth_map = self.last_depth_map

            # 3. ADD DEPTH TO DETECTIONS
            if depth_map is not None:
                for det in detections:
                    bbox = det['bbox']
                    depth = self.detection_pipeline.depth_estimator.get_depth_at_bbox(depth_map, bbox)
                    det['depth'] = depth
            else:
                for det in detections:
                    det['depth'] = 10.0

            # 4. SAFETY CHECKS AND CALIBRATION
            current_fps = self.get_current_fps()
            detections, warnings = self.safety_system.process_detections(detections, current_fps)

            # 5. SLAM (if enabled)
            slam_result = None
            if self.slam_enabled and self.slam:
                slam_result = self.slam.process_frame(frame, depth_map)

                # Update navigator
                if self.nav_enabled and self.navigator:
                    self.navigator.update(slam_result, detections)

            # 6. ANNOTATE FRAME
            annotated_frame = self.annotate_frame(frame, detections, warnings, current_fps, slam_result)

            # 7. CALCULATE FPS
            process_time = time.time() - start_time
            fps = 1.0 / process_time if process_time > 0 else 0
            self.fps_history.append(fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)

            return {
                'annotated_frame': annotated_frame,
                'detections': detections,
                'warnings': warnings,
                'fps': fps,
                'slam_result': slam_result,
                'depth_map': depth_map
            }

        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return None

    def annotate_frame(self, frame: np.ndarray, detections: List[Dict],
                      warnings: List[Dict], fps: float, slam_result: Optional[Dict] = None) -> np.ndarray:
        """Add annotations to frame with optional impressive demo overlay."""
        annotated = frame.copy()

        # Use impressive demo overlay for presentations
        if self.show_demo_overlay:
            # Draw detailed detection boxes
            annotated = self.demo_overlay.draw_detection_details(annotated, detections)

            # Add impressive stats overlay
            annotated = self.demo_overlay.create_impressive_overlay(
                annotated, detections, slam_result, fps
            )
        else:
            # Simple annotations (original)
            h, w = annotated.shape[:2]

            # Draw bounding boxes
            for det in detections:
                bbox = det['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                depth = det.get('depth', 0)
                label = det['label']

                # Color based on distance
                if depth < 0.4:
                    color = (0, 0, 255)  # Red - immediate danger
                elif depth < 1.0:
                    color = (0, 165, 255)  # Orange - danger
                elif depth < 2.0:
                    color = (0, 255, 255)  # Yellow - caution
                else:
                    color = (0, 255, 0)  # Green - safe

                # Draw box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Draw label
                text = f"{label} {depth:.1f}m"
                cv2.putText(annotated, text, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Simple status overlay
            overlay = annotated.copy()
            cv2.rectangle(overlay, (5, 5), (200, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)

            # FPS
            fps_color = (0, 255, 0) if fps > 10 else (0, 165, 255) if fps > 5 else (0, 0, 255)
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)

            # Danger count
            danger_count = len([d for d in detections if d.get('depth', 10) < 1.0])
            cv2.putText(annotated, f"Dangers: {danger_count}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Status
            if warnings and warnings[0]['level'] == 'IMMEDIATE_DANGER':
                status = "STOP!"
                status_color = (0, 0, 255)
            elif danger_count > 0:
                status = "DANGER"
                status_color = (0, 165, 255)
            else:
                status = "SAFE"
                status_color = (0, 255, 0)

            cv2.putText(annotated, status, (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return annotated

    def get_current_fps(self) -> float:
        """Get average FPS."""
        if not self.fps_history:
            return 15.0
        return sum(self.fps_history) / len(self.fps_history)

    def generate_audio_message(self, detections: List[Dict], warnings: List[Dict]) -> Optional[str]:
        """Generate TRULY HELPFUL audio message for blind users."""
        current_time = time.time()

        # Don't speak too often
        if current_time - self.last_audio_time < 1.5:
            return None

        # Use blind navigation assistant - provides CLEAR, ACTIONABLE guidance
        guidance = self.blind_nav.get_navigation_guidance(detections, None)

        # Check if we should speak this guidance now
        if self.blind_nav.should_speak_now(guidance):
            self.last_audio_time = current_time
            return guidance

        return None

    def run(self):
        """Main loop."""
        if not self.initialize_camera():
            self.logger.error("Cannot start without camera")
            return

        self.running = True
        self.audio_manager.speak("Navigation ready", priority=True)

        self.logger.info("OrbyGlasses Running - Press 'q' to quit")

        try:
            while self.running:
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.warning("Frame read failed")
                    break

                self.frame_count += 1

                # Process frame
                result = self.process_frame(frame)
                if not result:
                    continue

                # Generate and speak audio
                audio_msg = self.generate_audio_message(
                    result['detections'],
                    result['warnings']
                )
                if audio_msg and not self.audio_manager.is_speaking:
                    priority = AudioPriorityManager.PRIORITY_CRITICAL if 'STOP' in audio_msg else AudioPriorityManager.PRIORITY_MEDIUM
                    self.audio_manager.speak(audio_msg, priority=(priority >= 8))

                # Display
                cv2.imshow('OrbyGlasses', result['annotated_frame'])

                # Show SLAM if enabled
                if self.slam_enabled and result['slam_result']:
                    slam_vis = self.slam.visualize_tracking(frame, result['slam_result'])
                    cv2.imshow('SLAM', slam_vis)

                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("Quit requested")
                    break

                # Log stats every 100 frames
                if self.frame_count % 100 == 0:
                    avg_fps = self.get_current_fps()
                    self.logger.info(f"Frame {self.frame_count}: {avg_fps:.1f} FPS, "
                                   f"{len(result['detections'])} objects")

        except KeyboardInterrupt:
            self.logger.info("Interrupted")

        except Exception as e:
            self.logger.error(f"Runtime error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources."""
        self.logger.info("Shutting down...")

        if self.camera:
            self.camera.release()

        cv2.destroyAllWindows()
        self.audio_manager.stop()

        self.logger.info("Shutdown complete")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='OrbyGlasses Navigation')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Config file path')
    args = parser.parse_args()

    system = OrbyGlasses(config_path=args.config)
    system.run()


if __name__ == "__main__":
    main()
