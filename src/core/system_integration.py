"""
System Integration

Makes all the safety and speed features work together.
"""

import time
import cv2
from typing import Optional, Dict, List, Tuple
import numpy as np

# Import all production systems
try:
    from core.fast_audio import FastAudioManager, AudioPriority, emergency_alert, danger_warning, navigation_guidance
    FAST_AUDIO_AVAILABLE = True
except ImportError:
    FAST_AUDIO_AVAILABLE = False
    print("⚠️  FastAudioManager not available, using fallback")

try:
    from core.gpu_check import check_gpu_availability, get_optimal_device, verify_gpu_acceleration, DEVICE, OPTIMAL_SETTINGS
    GPU_CHECK_AVAILABLE = True
except ImportError:
    GPU_CHECK_AVAILABLE = False
    DEVICE = 'cpu'
    OPTIMAL_SETTINGS = {}
    print("⚠️  GPU checker not available")

try:
    from core.emergency_stop import EmergencyStopSystem, RedundantSafetyChecker, StopReason
    EMERGENCY_STOP_AVAILABLE = True
except ImportError:
    EMERGENCY_STOP_AVAILABLE = False
    print("⚠️  Emergency stop system not available")

try:
    from core.health_monitor import HealthMonitor, initialize_health_monitor
    HEALTH_MONITOR_AVAILABLE = True
except ImportError:
    HEALTH_MONITOR_AVAILABLE = False
    print("⚠️  Health monitor not available")


class IntegratedOrbyGlasses:
    """
    Fully integrated OrbyGlasses system with all production features.

    Integrates:
    - Fast audio (<500ms latency)
    - GPU acceleration (auto-configured)
    - Emergency stop (multi-layer safety)
    - Health monitoring (auto-recovery)
    - Redundant safety checks
    """

    def __init__(self, config, logger, base_audio_manager=None):
        """
        Initialize integrated system.

        Args:
            config: ConfigManager instance
            logger: Logger instance
            base_audio_manager: Existing AudioManager (for compatibility)
        """
        self.config = config
        self.logger = logger

        # Initialize GPU acceleration
        self._init_gpu()

        # Initialize fast audio system
        self._init_audio(base_audio_manager)

        # Initialize health monitoring
        self._init_health_monitor()

        # Initialize emergency stop
        self._init_emergency_stop()

        # Initialize safety checker
        self._init_safety_checker()

        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.last_audio_time = 0
        self.min_audio_interval = config.get('performance.audio_update_interval', 1.5)

        self.logger.info("✅ Integrated OrbyGlasses system initialized")

    def _init_gpu(self):
        """Initialize GPU acceleration"""
        if GPU_CHECK_AVAILABLE:
            gpu_info = check_gpu_availability()
            self.device = gpu_info['recommended_device']
            self.gpu_info = gpu_info

            # Verify GPU works
            success, message = verify_gpu_acceleration(self.device)
            if success:
                self.logger.info(f"✅ GPU: {gpu_info['device_name']} ({self.device})")
                self.logger.info(f"   Speedup: {gpu_info['acceleration_factor']:.1f}x vs CPU")
            else:
                self.logger.warning(f"⚠️  {message}")
                self.device = 'cpu'
        else:
            self.device = 'cpu'
            self.gpu_info = None
            self.logger.warning("⚠️  GPU acceleration not available")

    def _init_audio(self, base_audio_manager):
        """Initialize fast audio system"""
        if FAST_AUDIO_AVAILABLE:
            # Use FastAudioManager for <500ms latency
            rate = self.config.get('audio.tts_rate', 220)
            self.audio = FastAudioManager(rate=rate)
            self.use_fast_audio = True
            self.logger.info(f"✅ FastAudioManager: <500ms latency, {rate} WPM")
        else:
            # Fallback to existing audio manager
            self.audio = base_audio_manager
            self.use_fast_audio = False
            self.logger.warning("⚠️  Using fallback audio manager")

    def _init_health_monitor(self):
        """Initialize health monitoring"""
        if HEALTH_MONITOR_AVAILABLE:
            self.health_monitor = initialize_health_monitor(
                logger=self.logger,
                audio_manager=self.audio
            )

            # Register components
            self.health_monitor.register_component('camera')
            self.health_monitor.register_component('detection')
            self.health_monitor.register_component('depth')
            self.health_monitor.register_component('slam')
            self.health_monitor.register_component('audio')

            # Start monitoring
            self.health_monitor.start_monitoring()
            self.logger.info("✅ Health monitoring active")
        else:
            self.health_monitor = None
            self.logger.warning("⚠️  Health monitoring not available")

    def _init_emergency_stop(self):
        """Initialize emergency stop system"""
        if EMERGENCY_STOP_AVAILABLE:
            self.emergency_stop = EmergencyStopSystem(
                audio_manager=self.audio,
                logger=self.logger
            )

            # Register stop callback
            self.emergency_stop.register_stop_callback(self._on_emergency_stop)

            # Configure thresholds
            self.emergency_stop.min_safe_distance = self.config.get('safety.danger_distance', 0.5)

            self.logger.info("✅ Emergency stop system active")
        else:
            self.emergency_stop = None
            self.logger.warning("⚠️  Emergency stop not available")

    def _init_safety_checker(self):
        """Initialize redundant safety checker"""
        if EMERGENCY_STOP_AVAILABLE and self.emergency_stop:
            self.safety_checker = RedundantSafetyChecker(
                emergency_stop=self.emergency_stop,
                logger=self.logger
            )
            self.logger.info("✅ Redundant safety checks active")
        else:
            self.safety_checker = None

    def _on_emergency_stop(self, reason, message):
        """Callback for emergency stop"""
        self.logger.critical(f"EMERGENCY STOP TRIGGERED: {reason.value}")
        # Additional actions can be added here

    def process_frame(self, frame, detections, depth_map, slam_status=None, key=-1):
        """
        Process frame with integrated safety and performance systems.

        Args:
            frame: Camera frame
            detections: List of detected objects
            depth_map: Depth estimation map
            slam_status: SLAM tracking status (optional)
            key: Keyboard input from cv2.waitKey() (optional, default -1)

        Returns:
            (should_continue, audio_message)
        """
        self.frame_count += 1
        frame_start = time.time()

        # Check emergency stop (keyboard)
        if self.emergency_stop and key != -1:
            if self.emergency_stop.check_keyboard(key):
                return False, "Emergency stop activated"

        # Update health metrics
        if self.health_monitor:
            fps = self.frame_count / (time.time() - self.start_time)
            self.health_monitor.update_component('camera', fps=fps)

        # Run redundant safety checks
        if self.safety_checker:
            health_status = self.health_monitor.get_health_report() if self.health_monitor else None

            is_safe = self.safety_checker.check_all_safety_systems(
                detections=detections,
                depth_map=depth_map,
                slam_status=slam_status,
                health_status=health_status
            )

            if not is_safe:
                return False, "Safety check failed"

        # Analyze scene for dangers
        audio_message = self._analyze_safety(detections, depth_map)

        # Record frame time for health monitoring
        frame_time_ms = (time.time() - frame_start) * 1000
        if self.health_monitor:
            self.health_monitor.record_frame_time(frame_time_ms)

        return True, audio_message

    def _analyze_safety(self, detections, depth_map) -> Optional[str]:
        """
        Analyze scene for dangers and generate appropriate audio.

        Returns:
            Audio message to speak (or None if silent)
        """
        if not detections or depth_map is None:
            return None

        # Find closest object
        min_distance = float('inf')
        closest_object = None

        for det in detections:
            # Get object bounding box
            bbox = det.get('bbox', None)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox

            # Extract depth region
            try:
                depth_roi = depth_map[int(y1):int(y2), int(x1):int(x2)]
                if depth_roi.size > 0:
                    distance = np.median(depth_roi)

                    if distance < min_distance:
                        min_distance = distance
                        closest_object = det
            except:
                continue

        # Generate audio based on distance
        if closest_object and min_distance < float('inf'):
            # Check emergency stop for imminent collision
            if self.emergency_stop:
                self.emergency_stop.check_collision_risk(min_distance)

            # Generate appropriate audio message
            current_time = time.time()

            # Rate limit audio (except emergencies)
            if min_distance >= 1.0 and (current_time - self.last_audio_time) < self.min_audio_interval:
                return None

            obj_class = closest_object.get('class', 'obstacle')
            direction = self._get_direction(closest_object)

            # Create message based on distance
            if min_distance < 1.0:
                # DANGER - immediate action needed
                message = f"Stop! {obj_class} ahead. {min_distance:.1f} meters."

                if self.use_fast_audio:
                    danger_warning(self.audio, message)
                else:
                    self.audio.speak(message)

                self.last_audio_time = current_time
                return message

            elif min_distance < 2.5:
                # CAUTION - be aware
                message = f"{obj_class} {direction}. {min_distance:.1f} meters."

                if self.use_fast_audio:
                    self.audio.speak(message, priority=AudioPriority.WARNING)
                else:
                    self.audio.speak(message)

                self.last_audio_time = current_time
                return message

        return None

    def _get_direction(self, detection) -> str:
        """Get direction of object (left/right/ahead)"""
        bbox = detection.get('bbox', None)
        if bbox is None:
            return "ahead"

        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2

        # Assume frame width of 640 (adjust based on actual)
        frame_width = 640
        left_third = frame_width / 3
        right_third = 2 * frame_width / 3

        if center_x < left_third:
            return "on your left"
        elif center_x > right_third:
            return "on your right"
        else:
            return "ahead"

    def get_optimal_device(self):
        """Get optimal PyTorch device for models"""
        return self.device

    def get_optimal_settings(self):
        """Get optimal settings for current hardware"""
        if GPU_CHECK_AVAILABLE:
            return OPTIMAL_SETTINGS
        return {'batch_size': 1, 'use_half_precision': False}

    def shutdown(self):
        """Clean shutdown of all systems"""
        self.logger.info("Shutting down integrated systems...")

        if self.use_fast_audio and hasattr(self.audio, 'shutdown'):
            self.audio.shutdown()

        if self.health_monitor:
            self.health_monitor.stop_monitoring()

        if self.emergency_stop and not self.emergency_stop.is_stopped:
            self.emergency_stop.trigger_stop(
                StopReason.USER_REQUESTED,
                "System shutdown"
            )

        self.logger.info("✅ Clean shutdown complete")

    def get_status_report(self) -> Dict:
        """Get comprehensive system status"""
        report = {
            'uptime_seconds': time.time() - self.start_time,
            'frames_processed': self.frame_count,
            'average_fps': self.frame_count / (time.time() - self.start_time) if self.frame_count > 0 else 0,
            'gpu': {
                'device': self.device,
                'info': self.gpu_info
            }
        }

        if self.health_monitor:
            report['health'] = self.health_monitor.get_health_report()

        if self.emergency_stop:
            report['emergency_stop'] = self.emergency_stop.get_status()

        if self.use_fast_audio:
            report['audio'] = self.audio.get_latency_stats()

        return report


def print_integration_status():
    """Print status of all integrated systems"""
    print("\n" + "="*60)
    print("ORBGLASSES PRODUCTION SYSTEM STATUS")
    print("="*60)

    print(f"\n✅ Fast Audio: {'Available' if FAST_AUDIO_AVAILABLE else 'Not available'}")
    if FAST_AUDIO_AVAILABLE:
        print("   - Target latency: <500ms")
        print("   - Emergency alerts: <200ms")

    print(f"\n✅ GPU Acceleration: {'Available' if GPU_CHECK_AVAILABLE else 'Not available'}")
    if GPU_CHECK_AVAILABLE:
        print(f"   - Device: {DEVICE}")
        print(f"   - Settings: {OPTIMAL_SETTINGS}")

    print(f"\n✅ Emergency Stop: {'Available' if EMERGENCY_STOP_AVAILABLE else 'Not available'}")
    if EMERGENCY_STOP_AVAILABLE:
        print("   - Multi-layer safety active")
        print("   - Keyboard: spacebar, 'q', ESC")

    print(f"\n✅ Health Monitor: {'Available' if HEALTH_MONITOR_AVAILABLE else 'Not available'}")
    if HEALTH_MONITOR_AVAILABLE:
        print("   - Auto-recovery enabled")
        print("   - Component tracking active")

    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    print_integration_status()
