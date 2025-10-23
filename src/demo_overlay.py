"""
Demo Overlay - Impressive visual stats for presentations
Shows technical metrics in real-time to wow audiences
"""
import cv2
import numpy as np
import time
from typing import List, Dict, Optional
from collections import deque


class DemoOverlay:
    """Creates impressive visual overlays for demos and presentations."""

    def __init__(self, frame_width: int = 640, frame_height: int = 480):
        """Initialize demo overlay."""
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.process_times = deque(maxlen=30)

        # Stats tracking
        self.total_objects_detected = 0
        self.total_frames_processed = 0
        self.session_start = time.time()

        # Colors (BGR)
        self.color_green = (0, 255, 0)
        self.color_red = (0, 0, 255)
        self.color_orange = (0, 165, 255)
        self.color_yellow = (0, 255, 255)
        self.color_white = (255, 255, 255)
        self.color_cyan = (255, 255, 0)

    def create_impressive_overlay(self, frame: np.ndarray, detections: List[Dict],
                                 slam_result: Optional[Dict], fps: float) -> np.ndarray:
        """
        Create impressive overlay with all stats.

        Args:
            frame: Input frame
            detections: Current detections
            slam_result: SLAM result
            fps: Current FPS

        Returns:
            Frame with overlay
        """
        overlay = frame.copy()

        # Update stats
        self.fps_history.append(fps)
        self.total_objects_detected += len(detections)
        self.total_frames_processed += 1

        # Top left: Real-time performance
        self._draw_performance_panel(overlay, fps)

        # Top right: Detection stats
        self._draw_detection_panel(overlay, detections)

        # Bottom left: SLAM stats (if available)
        if slam_result:
            self._draw_slam_panel(overlay, slam_result)

        # Bottom right: Session stats
        self._draw_session_panel(overlay)

        # Center: Zone analysis visualization
        self._draw_zone_analysis(overlay, detections)

        return overlay

    def _draw_performance_panel(self, frame: np.ndarray, fps: float):
        """Draw performance metrics panel."""
        # Background panel
        cv2.rectangle(frame, (10, 10), (250, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, 120), self.color_cyan, 2)

        # Title
        cv2.putText(frame, "PERFORMANCE", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_cyan, 2)

        # FPS with color coding
        fps_color = self.color_green if fps > 15 else self.color_orange if fps > 10 else self.color_red
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, fps_color, 2)

        # Average FPS
        avg_fps = np.mean(list(self.fps_history)) if self.fps_history else 0
        cv2.putText(frame, f"Avg: {avg_fps:.1f}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_white, 1)

        # Latency
        latency = (1000 / fps) if fps > 0 else 0
        cv2.putText(frame, f"Latency: {latency:.0f}ms", (20, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_white, 1)

    def _draw_detection_panel(self, frame: np.ndarray, detections: List[Dict]):
        """Draw detection stats panel."""
        # Background panel
        x_start = self.frame_width - 250
        cv2.rectangle(frame, (x_start, 10), (self.frame_width - 10, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (x_start, 10), (self.frame_width - 10, 140), self.color_cyan, 2)

        # Title
        cv2.putText(frame, "DETECTION", (x_start + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_cyan, 2)

        # Current detections
        cv2.putText(frame, f"Objects: {len(detections)}", (x_start + 10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_green, 2)

        # Tracked objects
        tracked = len([d for d in detections if 'track_id' in d])
        cv2.putText(frame, f"Tracked: {tracked}", (x_start + 10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_white, 1)

        # Approaching objects
        approaching = len([d for d in detections if d.get('is_approaching', False)])
        if approaching > 0:
            cv2.putText(frame, f"Approaching: {approaching}", (x_start + 10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_red, 2)

        # Danger zone
        danger = len([d for d in detections if d.get('depth', 10) < 1.0])
        if danger > 0:
            cv2.putText(frame, f"DANGER: {danger}", (x_start + 10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_red, 2)

    def _draw_slam_panel(self, frame: np.ndarray, slam_result: Dict):
        """Draw SLAM stats panel."""
        # Background panel
        y_start = self.frame_height - 120
        cv2.rectangle(frame, (10, y_start), (220, self.frame_height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, y_start), (220, self.frame_height - 10), self.color_cyan, 2)

        # Title
        cv2.putText(frame, "SLAM", (20, y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_cyan, 2)

        # Position
        pos = slam_result.get('position', [0, 0, 0])
        cv2.putText(frame, f"X: {pos[0]:.2f}m  Y: {pos[1]:.2f}m", (20, y_start + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.color_white, 1)

        # Map points
        map_points = slam_result.get('num_map_points', 0)
        cv2.putText(frame, f"Map: {map_points} points", (20, y_start + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.color_white, 1)

        # Tracking quality
        quality = slam_result.get('tracking_quality', 0)
        quality_color = self.color_green if quality > 0.7 else self.color_orange if quality > 0.4 else self.color_red
        cv2.putText(frame, f"Quality: {quality:.0%}", (20, y_start + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, quality_color, 1)

    def _draw_session_panel(self, frame: np.ndarray):
        """Draw session stats panel."""
        # Background panel
        x_start = self.frame_width - 220
        y_start = self.frame_height - 100
        cv2.rectangle(frame, (x_start, y_start), (self.frame_width - 10, self.frame_height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (x_start, y_start), (self.frame_width - 10, self.frame_height - 10), self.color_cyan, 2)

        # Title
        cv2.putText(frame, "SESSION", (x_start + 10, y_start + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_cyan, 2)

        # Runtime
        runtime = time.time() - self.session_start
        minutes = int(runtime // 60)
        seconds = int(runtime % 60)
        cv2.putText(frame, f"Time: {minutes}:{seconds:02d}", (x_start + 10, y_start + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.color_white, 1)

        # Total frames
        cv2.putText(frame, f"Frames: {self.total_frames_processed}", (x_start + 10, y_start + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.color_white, 1)

    def _draw_zone_analysis(self, frame: np.ndarray, detections: List[Dict]):
        """Draw zone analysis visualization."""
        # Draw zone boundaries (semi-transparent)
        overlay = frame.copy()
        alpha = 0.2

        # Left zone
        left_clear = True
        for det in detections:
            if det['center'][0] < self.frame_width / 3 and det.get('depth', 10) < 3.0:
                left_clear = False
                break

        color_left = (0, 255, 0) if left_clear else (0, 0, 255)
        cv2.rectangle(overlay, (0, 0), (self.frame_width // 3, self.frame_height),
                     color_left, -1)

        # Center zone
        center_clear = True
        for det in detections:
            center_x = det['center'][0]
            if self.frame_width / 3 < center_x < 2 * self.frame_width / 3 and det.get('depth', 10) < 3.0:
                center_clear = False
                break

        color_center = (0, 255, 0) if center_clear else (0, 0, 255)
        cv2.rectangle(overlay, (self.frame_width // 3, 0),
                     (2 * self.frame_width // 3, self.frame_height),
                     color_center, -1)

        # Right zone
        right_clear = True
        for det in detections:
            if det['center'][0] > 2 * self.frame_width / 3 and det.get('depth', 10) < 3.0:
                right_clear = False
                break

        color_right = (0, 255, 0) if right_clear else (0, 0, 255)
        cv2.rectangle(overlay, (2 * self.frame_width // 3, 0),
                     (self.frame_width, self.frame_height),
                     color_right, -1)

        # Blend with original
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw zone labels at bottom
        y_pos = self.frame_height - 140
        # Left
        label_left = "LEFT: CLEAR" if left_clear else "LEFT: BLOCKED"
        color_left_text = self.color_green if left_clear else self.color_red
        cv2.putText(frame, label_left, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_left_text, 2)

        # Center
        label_center = "CENTER: CLEAR" if center_clear else "CENTER: BLOCKED"
        color_center_text = self.color_green if center_clear else self.color_red
        cv2.putText(frame, label_center, (self.frame_width // 2 - 60, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_center_text, 2)

        # Right
        label_right = "RIGHT: CLEAR" if right_clear else "RIGHT: BLOCKED"
        color_right_text = self.color_green if right_clear else self.color_red
        cv2.putText(frame, label_right, (self.frame_width - 150, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_right_text, 2)

    def draw_detection_details(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detailed detection boxes with all info.

        Args:
            frame: Input frame
            detections: Detections with tracking info

        Returns:
            Frame with detection boxes
        """
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            depth = det.get('depth', 0)
            label = det.get('label', 'object')

            # Color by depth
            if depth < 1.0:
                color = self.color_red
                thickness = 3
            elif depth < 2.0:
                color = self.color_orange
                thickness = 2
            else:
                color = self.color_green
                thickness = 2

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Prepare label with info
            info_lines = [f"{label}"]

            # Add depth
            info_lines.append(f"{depth:.1f}m")

            # Add tracking info
            if 'track_id' in det:
                info_lines.append(f"ID:{det['track_id']}")

            # Add approaching indicator
            if det.get('is_approaching', False):
                info_lines.append("APPROACHING!")

            # Draw background for text
            label_height = 20 * len(info_lines)
            cv2.rectangle(frame, (x1, y1 - label_height - 5), (x2, y1), color, -1)

            # Draw text lines
            for i, line in enumerate(info_lines):
                y_pos = y1 - label_height + (i * 20) + 15
                cv2.putText(frame, line, (x1 + 5, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw velocity vector if moving
            if det.get('is_moving', False):
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.arrowedLine(frame, (center_x, center_y),
                              (center_x + 30, center_y - 30),
                              self.color_yellow, 2)

        return frame
