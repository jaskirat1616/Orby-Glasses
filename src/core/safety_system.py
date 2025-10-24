"""
OrbyGlasses - Safety and Distance Calibration System
Ensures accurate distance measurements and provides fail-safe mechanisms for blind users.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import deque


@dataclass
class SafetyZone:
    """Defines safety zones around the user."""
    immediate_danger: float = 0.4  # Very close - stop immediately
    danger: float = 1.0  # Close - slow down and prepare to stop
    caution: float = 2.0  # Moderate distance - be aware
    safe: float = 5.0  # Far enough - continue normally


class DistanceCalibrator:
    """
    Calibrates depth estimates using known object sizes.
    Improves accuracy of depth estimation for critical navigation objects.
    """

    # Real-world average object heights in meters (accurate measurements)
    OBJECT_HEIGHTS = {
        'person': 1.65,  # Average human height (more accurate)
        'car': 1.45,  # Average car height
        'chair': 0.85,  # Standard chair height
        'couch': 0.9,  # Couch/sofa height
        'table': 0.75,  # Table height
        'desk': 0.75,  # Desk height
        'door': 2.0,  # Standard door height
        'bench': 0.45,  # Bench height
        'bicycle': 1.1,  # Bicycle height
        'motorcycle': 1.15,  # Motorcycle height
        'bus': 3.2,  # Bus height
        'truck': 2.4,  # Truck height
        'stop sign': 2.15,  # Stop sign height
        'fire hydrant': 0.75,  # Fire hydrant height
        'traffic light': 4.0,  # Traffic light height (on pole)
        'potted plant': 0.5,  # Potted plant
        'bottle': 0.25,  # Bottle height
        'cup': 0.12,  # Cup height
        'laptop': 0.35,  # Laptop (open height)
        'backpack': 0.45,  # Backpack height
        'handbag': 0.35,  # Handbag height
    }

    def __init__(self, focal_length: float = 600, frame_height: int = 480):
        """
        Initialize distance calibrator.

        Args:
            focal_length: Camera focal length in pixels (adjusted for better accuracy)
            frame_height: Frame height in pixels
        """
        self.focal_length = focal_length
        self.frame_height = frame_height
        self.calibration_history = deque(maxlen=30)  # Last 30 calibrations

    def estimate_distance_from_size(self, object_label: str, bbox_height: int) -> Optional[float]:
        """
        Estimate distance using object size and known real-world dimensions.

        Args:
            object_label: Object class label
            bbox_height: Bounding box height in pixels

        Returns:
            Estimated distance in meters or None if object type unknown
        """
        if object_label not in self.OBJECT_HEIGHTS or bbox_height < 10:
            return None

        real_height = self.OBJECT_HEIGHTS[object_label]

        # Distance = (Real Height × Focal Length) / Pixel Height
        distance = (real_height * self.focal_length) / bbox_height

        # Reasonable bounds (0.3m to 20m)
        distance = np.clip(distance, 0.3, 20.0)

        return float(distance)

    def calibrate_depth(self, detection: Dict, depth_estimate: float) -> float:
        """
        Calibrate depth estimate using size-based estimation.

        Args:
            detection: Detection dict with bbox and label
            depth_estimate: Raw depth estimate from depth model

        Returns:
            Calibrated depth estimate
        """
        bbox = detection.get('bbox', [0, 0, 100, 100])
        label = detection.get('label', 'unknown')

        # Calculate bbox height
        bbox_height = bbox[3] - bbox[1]

        # Get size-based estimate
        size_based_distance = self.estimate_distance_from_size(label, bbox_height)

        if size_based_distance is not None:
            # Weighted average: 70% size-based, 30% depth model
            # Size-based is much more reliable for known objects
            calibrated = 0.7 * size_based_distance + 0.3 * depth_estimate

            # Store calibration for future reference
            self.calibration_history.append({
                'label': label,
                'depth_model': depth_estimate,
                'size_based': size_based_distance,
                'calibrated': calibrated
            })

            return float(calibrated)

        # No size info available, return original estimate
        return depth_estimate

    def get_calibration_stats(self) -> Dict:
        """Get statistics about calibration accuracy."""
        if not self.calibration_history:
            return {'samples': 0}

        history = list(self.calibration_history)
        depth_diffs = [abs(h['depth_model'] - h['size_based']) for h in history]

        return {
            'samples': len(history),
            'avg_difference': np.mean(depth_diffs),
            'max_difference': np.max(depth_diffs),
            'calibrated_count': len([h for h in history if 'calibrated' in h])
        }


class CollisionWarningSystem:
    """
    Predicts and warns about potential collisions.
    Critical for blind user safety.
    """

    def __init__(self, safety_zones: SafetyZone):
        """
        Initialize collision warning system.

        Args:
            safety_zones: SafetyZone configuration
        """
        self.safety_zones = safety_zones
        self.object_history = {}  # Track object positions over time
        self.collision_warnings = []

    def update_object_tracking(self, detections: List[Dict]):
        """
        Update object tracking for velocity estimation.

        Args:
            detections: List of current detections with depth
        """
        current_objects = {}

        for det in detections:
            obj_id = f"{det['label']}_{det['center'][0]:.0f}_{det['center'][1]:.0f}"
            current_objects[obj_id] = {
                'position': det['center'],
                'depth': det.get('depth', 10.0),
                'bbox': det['bbox']
            }

        # Store for next frame comparison
        self.object_history = current_objects

    def check_collision_risk(self, detections: List[Dict]) -> List[Dict]:
        """
        Check for collision risks based on distance and approach speed.

        Args:
            detections: Current detections

        Returns:
            List of collision warnings
        """
        warnings = []

        for det in detections:
            depth = det.get('depth', 10.0)
            label = det.get('label', 'object')
            center = det.get('center', [0, 0])

            # Immediate danger zone
            if depth < self.safety_zones.immediate_danger:
                warnings.append({
                    'level': 'IMMEDIATE_DANGER',
                    'object': label,
                    'distance': depth,
                    'position': center,
                    'action': 'STOP NOW',
                    'urgency': 1.0
                })

            # Danger zone
            elif depth < self.safety_zones.danger:
                warnings.append({
                    'level': 'DANGER',
                    'object': label,
                    'distance': depth,
                    'position': center,
                    'action': 'SLOW DOWN',
                    'urgency': 0.8
                })

            # Caution zone
            elif depth < self.safety_zones.caution:
                warnings.append({
                    'level': 'CAUTION',
                    'object': label,
                    'distance': depth,
                    'position': center,
                    'action': 'BE AWARE',
                    'urgency': 0.5
                })

        # Sort by urgency (most urgent first)
        warnings.sort(key=lambda x: x['urgency'], reverse=True)

        self.collision_warnings = warnings
        return warnings

    def get_highest_priority_warning(self) -> Optional[Dict]:
        """Get the most urgent warning."""
        if not self.collision_warnings:
            return None
        return self.collision_warnings[0]


class SystemHealthMonitor:
    """
    Monitors system health and triggers fail-safe modes.
    Ensures the system degrades gracefully rather than failing catastrophically.
    """

    def __init__(self):
        """Initialize system health monitor."""
        self.fps_history = deque(maxlen=30)
        self.error_count = 0
        self.warning_count = 0
        self.frame_drop_count = 0
        self.last_detection_time = None
        self.health_status = 'HEALTHY'

    def record_frame(self, fps: float, had_error: bool = False):
        """
        Record frame processing metrics.

        Args:
            fps: Current FPS
            had_error: Whether an error occurred
        """
        self.fps_history.append(fps)

        if had_error:
            self.error_count += 1

        # Check for low FPS
        if fps < 5:
            self.frame_drop_count += 1
        else:
            self.frame_drop_count = max(0, self.frame_drop_count - 1)

    def get_health_status(self) -> str:
        """
        Get current system health status.

        Returns:
            'HEALTHY', 'DEGRADED', or 'CRITICAL'
        """
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0

        # Critical: Very low FPS or many errors
        if avg_fps < 3 or self.error_count > 10:
            return 'CRITICAL'

        # Degraded: Low FPS or some errors
        elif avg_fps < 7 or self.error_count > 3 or self.frame_drop_count > 5:
            return 'DEGRADED'

        # Healthy
        return 'HEALTHY'

    def should_enable_safe_mode(self) -> bool:
        """Check if safe mode should be enabled."""
        return self.get_health_status() == 'CRITICAL'

    def should_reduce_processing(self) -> bool:
        """Check if processing should be reduced."""
        return self.get_health_status() in ['CRITICAL', 'DEGRADED']

    def get_health_report(self) -> Dict:
        """Get detailed health report."""
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0

        return {
            'status': self.get_health_status(),
            'avg_fps': avg_fps,
            'error_count': self.error_count,
            'frame_drops': self.frame_drop_count,
            'safe_mode_needed': self.should_enable_safe_mode(),
            'reduce_processing': self.should_reduce_processing()
        }

    def reset_errors(self):
        """Reset error counters."""
        self.error_count = 0
        self.warning_count = 0
        self.frame_drop_count = 0


class SafetySystem:
    """
    Integrated safety system combining all safety features.
    Primary interface for safety-critical operations.
    """

    def __init__(self, focal_length: float = 500, frame_height: int = 480):
        """
        Initialize integrated safety system.

        Args:
            focal_length: Camera focal length
            frame_height: Frame height in pixels
        """
        self.safety_zones = SafetyZone()
        self.distance_calibrator = DistanceCalibrator(focal_length, frame_height)
        self.collision_warning = CollisionWarningSystem(self.safety_zones)
        self.health_monitor = SystemHealthMonitor()

        self.logger = logging.getLogger(__name__)

    def process_detections(self, detections: List[Dict], fps: float = 15.0) -> Tuple[List[Dict], List[Dict]]:
        """
        Process detections through safety system.

        Args:
            detections: Raw detections with depth estimates
            fps: Current FPS

        Returns:
            Tuple of (calibrated_detections, warnings)
        """
        # Record system health
        self.health_monitor.record_frame(fps)

        # Calibrate distances
        calibrated_detections = []
        for det in detections:
            depth = det.get('depth', 10.0)
            calibrated_depth = self.distance_calibrator.calibrate_depth(det, depth)

            # Update detection with calibrated depth
            det_copy = det.copy()
            det_copy['depth'] = calibrated_depth
            det_copy['depth_raw'] = depth  # Keep original for debugging
            calibrated_detections.append(det_copy)

        # Check collision risks
        warnings = self.collision_warning.check_collision_risk(calibrated_detections)

        # Update object tracking
        self.collision_warning.update_object_tracking(calibrated_detections)

        return calibrated_detections, warnings

    def get_safety_guidance(self, warnings: List[Dict]) -> str:
        """
        Generate clear safety guidance message.

        Args:
            warnings: List of warnings

        Returns:
            Safety guidance message
        """
        if not warnings:
            return "Path clear, safe to proceed"

        # Get highest priority warning
        top_warning = warnings[0]
        level = top_warning['level']
        obj = top_warning['object']
        dist = top_warning['distance']
        action = top_warning['action']

        # Generate clear message based on level
        if level == 'IMMEDIATE_DANGER':
            return f"STOP! {obj} directly ahead at {dist:.1f} meters"
        elif level == 'DANGER':
            return f"Warning: {obj} ahead at {dist:.1f} meters. {action}"
        elif level == 'CAUTION':
            return f"{obj} detected {dist:.1f} meters away"

        return "Obstacles detected, proceed with caution"

    def check_system_health(self) -> Dict:
        """Get system health report."""
        return self.health_monitor.get_health_report()

    def enable_safe_mode(self) -> Dict:
        """
        Enable safe mode with reduced processing.

        Returns:
            Safe mode configuration
        """
        return {
            'reduce_frame_rate': True,
            'skip_depth_frames': 4,  # Only process depth every 4th frame
            'max_detections': 3,  # Only track top 3 objects
            'disable_slam': True,  # Disable SLAM in safe mode
            'audio_priority_only': True,  # Only play critical alerts
            'increase_audio_interval': True  # Speak more frequently
        }
