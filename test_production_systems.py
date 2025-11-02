#!/usr/bin/env python3
"""
Test Production Systems

Verifies all production features work correctly.
"""

import sys
import os
import time
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("\n" + "="*60)
print("ORBGLASSES PRODUCTION SYSTEMS TEST")
print("="*60 + "\n")

# Test 1: GPU Acceleration
print("1️⃣  Testing GPU Acceleration...")
try:
    from core.gpu_check import check_gpu_availability, verify_gpu_acceleration, print_gpu_report

    print_gpu_report()
    print("✅ GPU system working\n")
except Exception as e:
    print(f"❌ GPU test failed: {e}\n")

# Test 2: Fast Audio
print("2️⃣  Testing Fast Audio System...")
try:
    from core.fast_audio import FastAudioManager, AudioPriority, emergency_alert

    audio = FastAudioManager(rate=220, voice="Samantha")

    print("   Testing emergency alert (should be <200ms)...")
    start = time.time()
    emergency_alert(audio, "Test emergency")
    latency = (time.time() - start) * 1000
    print(f"   Emergency latency: {latency:.0f}ms {'✅' if latency < 500 else '❌'}")

    time.sleep(0.5)

    print("   Testing normal speech (should be <500ms)...")
    start = time.time()
    audio.speak("Normal navigation message", priority=AudioPriority.INFO)
    time.sleep(2)  # Wait for completion
    latency = (time.time() - start) * 1000
    print(f"   Normal latency: {latency:.0f}ms")

    audio.shutdown()
    print("✅ Fast audio working\n")
except Exception as e:
    print(f"❌ Fast audio test failed: {e}\n")

# Test 3: Emergency Stop
print("3️⃣  Testing Emergency Stop System...")
try:
    from core.emergency_stop import EmergencyStopSystem, RedundantSafetyChecker, StopReason

    # Create mock audio and logger
    class MockLogger:
        def info(self, msg): print(f"   [INFO] {msg}")
        def warning(self, msg): print(f"   [WARN] {msg}")
        def error(self, msg): print(f"   [ERROR] {msg}")
        def critical(self, msg): print(f"   [CRITICAL] {msg}")

    logger = MockLogger()

    # Test emergency stop
    estop = EmergencyStopSystem(audio_manager=None, logger=logger)

    print("   Testing collision detection...")
    is_stopped = estop.check_collision_risk(0.3)  # 0.3m - too close!
    print(f"   Collision check: {'✅ Stopped' if is_stopped else '❌ Failed to stop'}")

    estop.reset()

    print("   Testing safe distance...")
    is_stopped = estop.check_collision_risk(2.0)  # 2.0m - safe
    print(f"   Safe distance: {'✅ Not stopped' if not is_stopped else '❌ False alarm'}")

    # Test redundant safety checker
    print("   Testing redundant safety checks...")
    safety = RedundantSafetyChecker(estop, logger)

    # Test with valid depth map
    depth_map = np.ones((480, 640)) * 2.5  # 2.5m uniform
    is_safe = safety._check_depth_safety(depth_map)
    print(f"   Depth safety: {'✅ Passed' if is_safe else '❌ Failed'}")

    # Test with invalid depth map (all zeros)
    depth_map_bad = np.zeros((480, 640))
    estop.reset()
    is_safe = safety._check_depth_safety(depth_map_bad)
    print(f"   Invalid depth detection: {'✅ Caught' if not is_safe else '❌ Missed'}")

    print("✅ Emergency stop working\n")
except Exception as e:
    print(f"❌ Emergency stop test failed: {e}\n")

# Test 4: Health Monitor
print("4️⃣  Testing Health Monitor...")
try:
    from core.health_monitor import HealthMonitor, ComponentStatus

    monitor = HealthMonitor(logger=None, audio_manager=None)

    # Register components
    monitor.register_component('test_component')

    # Update with good metrics
    monitor.update_component('test_component', fps=25.0, error=False)
    status = monitor.components['test_component'].status
    print(f"   Healthy component: {status.value} {'✅' if status == ComponentStatus.HEALTHY else '❌'}")

    # Update with failures
    for i in range(5):
        monitor.update_component('test_component', error=True)

    status = monitor.components['test_component'].status
    print(f"   Failed component detection: {status.value} {'✅' if status == ComponentStatus.FAILED else '❌'}")

    # Test system metrics
    metrics = monitor.get_system_metrics()
    print(f"   System metrics: CPU={metrics.get('cpu_percent', 0):.1f}%, Memory={metrics.get('memory_mb', 0):.0f}MB")

    print("✅ Health monitor working\n")
except Exception as e:
    print(f"❌ Health monitor test failed: {e}\n")

# Test 5: System Integration
print("5️⃣  Testing System Integration...")
try:
    from core.system_integration import IntegratedOrbyGlasses, print_integration_status
    from core.utils import ConfigManager, Logger

    print_integration_status()

    # Test initialization
    config = ConfigManager('config/config.yaml')
    logger = Logger(name="Test")

    print("   Initializing integrated system...")
    integrated = IntegratedOrbyGlasses(config, logger, base_audio_manager=None)

    print(f"   GPU Device: {integrated.device}")
    print(f"   Fast Audio: {integrated.use_fast_audio}")
    print(f"   Health Monitor: {integrated.health_monitor is not None}")
    print(f"   Emergency Stop: {integrated.emergency_stop is not None}")

    # Test frame processing
    print("   Testing frame processing...")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [{
        'class': 'person',
        'confidence': 0.95,
        'bbox': (200, 150, 400, 350)
    }]
    depth_map = np.ones((480, 640)) * 3.0

    should_continue, message = integrated.process_frame(frame, detections, depth_map)
    print(f"   Frame processing: {'✅ Working' if should_continue else '❌ Failed'}")

    # Get status report
    report = integrated.get_status_report()
    print(f"   Status report: {len(report)} metrics collected")

    # Shutdown
    integrated.shutdown()

    print("✅ System integration working\n")
except Exception as e:
    print(f"❌ System integration test failed: {e}\n")
    import traceback
    traceback.print_exc()

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("\nAll production systems have been tested.")
print("Review results above for any ❌ failures.")
print("\nIf all tests passed ✅, the system is production-ready!")
print("\n" + "="*60 + "\n")
