"""
Health Monitor

Watches all parts of OrbyGlasses and fixes problems automatically.
"""

import time
import psutil
import threading
from typing import Dict, Optional, Callable
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class ComponentStatus(Enum):
    """Component health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"


@dataclass
class HealthMetrics:
    """Health metrics for a component"""
    name: str
    status: ComponentStatus = ComponentStatus.HEALTHY
    last_update: float = field(default_factory=time.time)
    fps: float = 0.0
    error_count: int = 0
    latency_ms: float = 0.0
    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    consecutive_failures: int = 0


class HealthMonitor:
    """
    System health monitoring and auto-recovery

    Features:
    - Real-time component health tracking
    - Automatic failure detection
    - Recovery strategies
    - Performance monitoring
    - Resource usage tracking
    """

    def __init__(self, logger=None, audio_manager=None):
        self.logger = logger
        self.audio_manager = audio_manager

        # Component tracking
        self.components: Dict[str, HealthMetrics] = {}
        self.recovery_callbacks: Dict[str, Callable] = {}

        # Performance tracking
        self.fps_history = deque(maxlen=100)
        self.frame_times = deque(maxlen=30)

        # Thresholds
        self.min_fps = 5.0
        self.max_consecutive_failures = 3
        self.health_check_interval = 5.0  # seconds

        # Monitoring thread
        self.monitoring = False
        self.monitor_thread = None

        # System metrics
        self.process = psutil.Process()
        self.start_time = time.time()

    def register_component(self, name: str, recovery_callback: Optional[Callable] = None):
        """Register a component for health monitoring"""
        self.components[name] = HealthMetrics(name=name)
        if recovery_callback:
            self.recovery_callbacks[name] = recovery_callback

        if self.logger:
            self.logger.debug(f"Registered component for health monitoring: {name}")

    def update_component(self, name: str, **kwargs):
        """Update component health metrics"""
        if name not in self.components:
            self.register_component(name)

        component = self.components[name]
        component.last_update = time.time()

        # Update provided metrics
        if 'fps' in kwargs:
            component.fps = kwargs['fps']
            self.fps_history.append(kwargs['fps'])

        if 'error' in kwargs and kwargs['error']:
            component.error_count += 1
            component.consecutive_failures += 1
        else:
            component.consecutive_failures = 0

        if 'latency_ms' in kwargs:
            component.latency_ms = kwargs['latency_ms']

        # Determine status
        self._update_status(component)

    def _update_status(self, component: HealthMetrics):
        """Update component status based on metrics"""
        # Check for failures
        if component.consecutive_failures >= self.max_consecutive_failures:
            component.status = ComponentStatus.FAILED
            self._handle_component_failure(component)

        # Check FPS
        elif component.fps > 0 and component.fps < self.min_fps:
            component.status = ComponentStatus.DEGRADED
            self._handle_degradation(component)

        # Check staleness (no updates in 30 seconds)
        elif time.time() - component.last_update > 30:
            component.status = ComponentStatus.DEGRADED
            if self.logger:
                self.logger.warning(f"Component {component.name} hasn't updated in 30s")

        else:
            component.status = ComponentStatus.HEALTHY

    def _handle_component_failure(self, component: HealthMetrics):
        """Handle component failure"""
        if self.logger:
            self.logger.error(f"Component FAILED: {component.name} "
                            f"({component.consecutive_failures} consecutive failures)")

        # Notify user via audio
        if self.audio_manager:
            self.audio_manager.speak(f"{component.name} system failed. Attempting recovery.")

        # Attempt recovery
        if component.name in self.recovery_callbacks:
            component.status = ComponentStatus.RECOVERING
            try:
                self.recovery_callbacks[component.name]()
                component.consecutive_failures = 0
                component.status = ComponentStatus.HEALTHY

                if self.logger:
                    self.logger.info(f"Successfully recovered {component.name}")

                if self.audio_manager:
                    self.audio_manager.speak(f"{component.name} recovered.")

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Recovery failed for {component.name}: {e}")

    def _handle_degradation(self, component: HealthMetrics):
        """Handle component degradation"""
        if self.logger:
            self.logger.warning(f"Component DEGRADED: {component.name} "
                              f"(FPS: {component.fps:.1f})")

    def record_frame_time(self, duration_ms: float):
        """Record frame processing time"""
        self.frame_times.append(duration_ms)

    def get_average_fps(self) -> float:
        """Get average FPS over recent frames"""
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)

    def get_average_frame_time(self) -> float:
        """Get average frame processing time in ms"""
        if not self.frame_times:
            return 0.0
        return sum(self.frame_times) / len(self.frame_times)

    def get_system_metrics(self) -> Dict[str, float]:
        """Get system resource usage metrics"""
        try:
            return {
                'cpu_percent': self.process.cpu_percent(),
                'memory_mb': self.process.memory_info().rss / 1024 / 1024,
                'memory_percent': self.process.memory_percent(),
                'uptime_minutes': (time.time() - self.start_time) / 60
            }
        except:
            return {}

    def get_health_report(self) -> Dict:
        """Get comprehensive health report"""
        report = {
            'overall_status': self._get_overall_status(),
            'components': {},
            'system': self.get_system_metrics(),
            'performance': {
                'avg_fps': self.get_average_fps(),
                'avg_frame_time_ms': self.get_average_frame_time()
            }
        }

        for name, metrics in self.components.items():
            report['components'][name] = {
                'status': metrics.status.value,
                'fps': metrics.fps,
                'error_count': metrics.error_count,
                'latency_ms': metrics.latency_ms,
                'consecutive_failures': metrics.consecutive_failures
            }

        return report

    def _get_overall_status(self) -> str:
        """Determine overall system health"""
        if not self.components:
            return "unknown"

        statuses = [c.status for c in self.components.values()]

        if any(s == ComponentStatus.FAILED for s in statuses):
            return "critical"
        elif any(s == ComponentStatus.DEGRADED for s in statuses):
            return "degraded"
        elif any(s == ComponentStatus.RECOVERING for s in statuses):
            return "recovering"
        else:
            return "healthy"

    def start_monitoring(self):
        """Start background health monitoring"""
        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        if self.logger:
            self.logger.info("Health monitoring started")

    def stop_monitoring(self):
        """Stop background health monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        if self.logger:
            self.logger.info("Health monitoring stopped")

    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # Check component health
                report = self.get_health_report()

                # Log periodic health status
                if self.logger and report['overall_status'] != "healthy":
                    self.logger.warning(f"System health: {report['overall_status']}")
                    self.logger.debug(f"Health report: {report}")

                # Check for resource issues
                sys_metrics = report['system']
                if sys_metrics.get('cpu_percent', 0) > 90:
                    if self.logger:
                        self.logger.warning(f"High CPU usage: {sys_metrics['cpu_percent']:.1f}%")

                if sys_metrics.get('memory_percent', 0) > 80:
                    if self.logger:
                        self.logger.warning(f"High memory usage: {sys_metrics['memory_percent']:.1f}%")

            except Exception as e:
                if self.logger:
                    self.logger.error(f"Health monitor error: {e}")

            time.sleep(self.health_check_interval)

    def is_healthy(self) -> bool:
        """Check if system is healthy overall"""
        return self._get_overall_status() in ["healthy", "recovering"]

    def should_restart(self) -> bool:
        """Determine if system should restart"""
        # Restart if multiple critical components failed
        failed_count = sum(
            1 for c in self.components.values()
            if c.status == ComponentStatus.FAILED
        )

        return failed_count >= 2


# Global instance for easy access
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> Optional[HealthMonitor]:
    """Get global health monitor instance"""
    return _health_monitor


def initialize_health_monitor(logger=None, audio_manager=None) -> HealthMonitor:
    """Initialize global health monitor"""
    global _health_monitor
    _health_monitor = HealthMonitor(logger=logger, audio_manager=audio_manager)
    return _health_monitor
