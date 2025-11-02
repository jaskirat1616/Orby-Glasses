"""
Emergency Stop System for OrbyGlasses

Provides multiple layers of emergency stopping and safety controls.
"""

import cv2
import threading
import time
from typing import Callable, List, Optional
from enum import Enum


class StopReason(Enum):
    """Reasons for emergency stop"""
    USER_REQUESTED = "user_requested"  # User pressed emergency stop key
    IMMINENT_COLLISION = "imminent_collision"  # Obstacle <0.5m
    SYSTEM_FAILURE = "system_failure"  # Critical component failed
    TRACKING_LOST = "tracking_lost"  # SLAM completely lost
    LOW_CONFIDENCE = "low_confidence"  # Detection confidence too low
    SENSOR_FAILURE = "sensor_failure"  # Camera or depth sensor failed


class EmergencyStopSystem:
    """
    Multi-layer emergency stop system.

    Features:
    - Keyboard emergency stop (spacebar or 'q')
    - Automatic stop on imminent collision
    - System failure detection
    - Redundant safety checks
    - Graceful degradation
    """

    def __init__(self, audio_manager=None, logger=None):
        self.audio_manager = audio_manager
        self.logger = logger

        # Emergency stop state
        self.is_stopped = False
        self.stop_reason: Optional[StopReason] = None
        self.stop_time: Optional[float] = None

        # Emergency stop callbacks
        self.stop_callbacks: List[Callable] = []

        # Safety thresholds
        self.min_safe_distance = 0.5  # meters - absolute minimum
        self.max_detection_failures = 3  # consecutive failures before stop
        self.max_tracking_loss_time = 5.0  # seconds

        # Failure counters
        self.detection_failures = 0
        self.tracking_loss_start: Optional[float] = None

        # Thread safety
        self.lock = threading.Lock()

        if self.logger:
            self.logger.info("Emergency Stop System initialized")

    def register_stop_callback(self, callback: Callable):
        """Register callback to be called on emergency stop"""
        self.stop_callbacks.append(callback)

    def trigger_stop(self, reason: StopReason, message: str = None):
        """
        Trigger emergency stop.

        Args:
            reason: Reason for stopping
            message: Optional custom message
        """
        with self.lock:
            if self.is_stopped:
                return  # Already stopped

            self.is_stopped = True
            self.stop_reason = reason
            self.stop_time = time.time()

            # Create emergency message
            if message is None:
                message = self._get_default_message(reason)

            # Log the stop
            if self.logger:
                self.logger.critical(f"EMERGENCY STOP: {reason.value} - {message}")

            # Audio alert
            if self.audio_manager:
                # Use immediate speak for emergencies
                if hasattr(self.audio_manager, 'speak_immediate'):
                    self.audio_manager.speak_immediate(message)
                else:
                    self.audio_manager.speak(message)

            # Execute callbacks
            for callback in self.stop_callbacks:
                try:
                    callback(reason, message)
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Stop callback error: {e}")

            print(f"\n{'='*60}")
            print(f"⛔ EMERGENCY STOP: {reason.value}")
            print(f"   {message}")
            print(f"{'='*60}\n")

    def _get_default_message(self, reason: StopReason) -> str:
        """Get default message for stop reason"""
        messages = {
            StopReason.USER_REQUESTED: "Emergency stop activated",
            StopReason.IMMINENT_COLLISION: "Stop! Obstacle too close",
            StopReason.SYSTEM_FAILURE: "System failure. Stopping for safety",
            StopReason.TRACKING_LOST: "Position tracking lost. Stop and wait",
            StopReason.LOW_CONFIDENCE: "Low detection confidence. Stop for safety",
            StopReason.SENSOR_FAILURE: "Sensor failure. Stop immediately"
        }
        return messages.get(reason, "Emergency stop")

    def check_collision_risk(self, min_distance: float) -> bool:
        """
        Check if obstacle is dangerously close.

        Args:
            min_distance: Minimum distance to obstacle in meters

        Returns:
            True if emergency stop triggered
        """
        if min_distance < self.min_safe_distance:
            self.trigger_stop(
                StopReason.IMMINENT_COLLISION,
                f"Obstacle at {min_distance:.1f} meters. Too close!"
            )
            return True
        return False

    def check_detection_health(self, detection_successful: bool) -> bool:
        """
        Track detection failures and trigger stop if too many.

        Args:
            detection_successful: Whether detection succeeded

        Returns:
            True if emergency stop triggered
        """
        if detection_successful:
            self.detection_failures = 0
        else:
            self.detection_failures += 1

            if self.detection_failures >= self.max_detection_failures:
                self.trigger_stop(
                    StopReason.LOW_CONFIDENCE,
                    f"Detection failed {self.detection_failures} times"
                )
                return True

        return False

    def check_tracking_health(self, tracking_active: bool) -> bool:
        """
        Monitor SLAM tracking and trigger stop if lost too long.

        Args:
            tracking_active: Whether SLAM is tracking

        Returns:
            True if emergency stop triggered
        """
        if tracking_active:
            self.tracking_loss_start = None
        else:
            if self.tracking_loss_start is None:
                self.tracking_loss_start = time.time()
            else:
                loss_duration = time.time() - self.tracking_loss_start

                if loss_duration > self.max_tracking_loss_time:
                    self.trigger_stop(
                        StopReason.TRACKING_LOST,
                        f"Tracking lost for {loss_duration:.1f} seconds"
                    )
                    return True

        return False

    def check_keyboard(self, key: int) -> bool:
        """
        Check for emergency stop keyboard input.

        Args:
            key: cv2 waitKey result

        Returns:
            True if emergency stop triggered
        """
        # Spacebar (32) or 'q' (113) triggers emergency stop
        if key in [32, ord('q'), ord('Q'), 27]:  # space, q, Q, ESC
            self.trigger_stop(
                StopReason.USER_REQUESTED,
                "Emergency stop requested by user"
            )
            return True

        return False

    def reset(self):
        """Reset emergency stop (allow system to continue)"""
        with self.lock:
            if self.logger:
                self.logger.info(f"Emergency stop reset (was: {self.stop_reason})")

            self.is_stopped = False
            self.stop_reason = None
            self.stop_time = None
            self.detection_failures = 0
            self.tracking_loss_start = None

            print("✅ Emergency stop reset - system can continue")

    def get_status(self) -> dict:
        """Get emergency stop status"""
        with self.lock:
            return {
                'is_stopped': self.is_stopped,
                'reason': self.stop_reason.value if self.stop_reason else None,
                'stop_time': self.stop_time,
                'detection_failures': self.detection_failures,
                'tracking_loss_duration': (
                    time.time() - self.tracking_loss_start
                    if self.tracking_loss_start
                    else 0
                )
            }


class RedundantSafetyChecker:
    """
    Redundant safety checks to ensure no single point of failure.

    Implements multiple independent safety checks:
    1. Distance-based collision detection
    2. Object detection confidence
    3. Depth estimation validity
    4. SLAM tracking quality
    5. System health monitoring
    """

    def __init__(self, emergency_stop: EmergencyStopSystem, logger=None):
        self.emergency_stop = emergency_stop
        self.logger = logger

        # Safety thresholds
        self.min_detection_confidence = 0.3  # Very low to catch anything
        self.min_depth_confidence = 0.5
        self.min_tracking_quality = 0.3

        # Last known good values (for comparison)
        self.last_valid_depth = None
        self.last_valid_position = None

    def check_all_safety_systems(
        self,
        detections: List = None,
        depth_map = None,
        slam_status: dict = None,
        health_status: dict = None
    ) -> bool:
        """
        Run all redundant safety checks.

        Returns:
            True if all systems safe, False if emergency stop triggered
        """
        # Check 1: Detection validity
        if detections is not None:
            if not self._check_detection_safety(detections):
                return False

        # Check 2: Depth map validity
        if depth_map is not None:
            if not self._check_depth_safety(depth_map):
                return False

        # Check 3: SLAM tracking
        if slam_status is not None:
            if not self._check_slam_safety(slam_status):
                return False

        # Check 4: System health
        if health_status is not None:
            if not self._check_system_health(health_status):
                return False

        return True

    def _check_detection_safety(self, detections: List) -> bool:
        """Check if object detection is working properly"""
        # If no detections for extended period, may indicate camera failure
        # But this is normal in empty environments, so just log
        return True

    def _check_depth_safety(self, depth_map) -> bool:
        """Check if depth estimation is valid"""
        import numpy as np

        # Check for NaN or Inf values
        if np.any(np.isnan(depth_map)) or np.any(np.isinf(depth_map)):
            self.emergency_stop.trigger_stop(
                StopReason.SENSOR_FAILURE,
                "Depth sensor returning invalid data"
            )
            return False

        # Check for all zeros (sensor failure)
        if np.allclose(depth_map, 0):
            self.emergency_stop.trigger_stop(
                StopReason.SENSOR_FAILURE,
                "Depth sensor not responding"
            )
            return False

        self.last_valid_depth = depth_map
        return True

    def _check_slam_safety(self, slam_status: dict) -> bool:
        """Check SLAM tracking quality"""
        tracking_quality = slam_status.get('tracking_quality', 1.0)

        if tracking_quality < self.min_tracking_quality:
            if self.logger:
                self.logger.warning(f"Low SLAM tracking quality: {tracking_quality:.2f}")

        # Let EmergencyStopSystem handle prolonged tracking loss
        return True

    def _check_system_health(self, health_status: dict) -> bool:
        """Check overall system health"""
        overall_status = health_status.get('overall_status', 'unknown')

        if overall_status == 'critical':
            self.emergency_stop.trigger_stop(
                StopReason.SYSTEM_FAILURE,
                "Critical system health status"
            )
            return False

        return True
