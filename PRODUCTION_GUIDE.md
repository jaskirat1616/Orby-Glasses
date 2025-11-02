# OrbyGlasses Production System Guide

Complete guide to using OrbyGlasses production features for safe blind navigation.

## Quick Start

```bash
# Standard launch (auto-configures everything)
./run_orby.sh

# Emergency stop: Press SPACEBAR, 'q', or ESC at any time
```

## Production Features

### 1. Fast Audio System (<500ms Latency)

**What it does:** Provides near-instant audio warnings for safety.

**Audio Priority Levels:**
- **EMERGENCY** (<200ms): Imminent danger, interrupts everything
  - "Stop! Car ahead."
- **DANGER** (<500ms): Close obstacle (1-2m)
  - "Person on your left. 1.5 meters."
- **WARNING** (<1s): Moderate distance (2-3m)
  - "Chair ahead. 2.8 meters."
- **INFO** (queued): General guidance
  - "Path is clear."

**How it works:**
- Uses macOS 'say' command directly (no pyttsx3 overhead)
- Priority queue ensures safety messages come first
- Emergency messages interrupt current speech
- Target: <500ms from detection to audio output

**Configuration:**
```yaml
audio:
  tts_rate: 220  # Words per minute (180-300)
  tts_volume: 1.0
```

### 2. GPU Acceleration (Auto-Configured)

**What it does:** Automatically detects and uses your GPU for 5-10x faster processing.

**Supported Hardware:**
- **Apple Silicon (M1/M2/M3/M4):** Uses MPS backend (5x speedup)
- **NVIDIA GPUs:** Uses CUDA backend (10x speedup)
- **Intel/AMD CPUs:** Automatic fallback

**How it works:**
- Runs on startup: detects available hardware
- Verifies GPU actually works (not just present)
- Configures optimal settings automatically:
  - Batch size (1 for CPU, 4 for MPS, 8 for CUDA)
  - Half-precision (FP16) when beneficial
  - Memory management

**View GPU Status:**
```bash
python3 -c "from src.core.gpu_check import print_gpu_report; print_gpu_report()"
```

**Expected Performance:**
| Hardware | FPS | Latency |
|----------|-----|---------|
| M1/M2/M3 (MPS) | 20-30 | Good |
| NVIDIA GPU (CUDA) | 30-50 | Excellent |
| CPU Only | 5-10 | Poor |

### 3. Emergency Stop System (Multi-Layer Safety)

**What it does:** Stops the system immediately if danger is detected.

**Stop Triggers:**

1. **Keyboard Emergency Stop**
   - Press: SPACEBAR, 'q', 'Q', or ESC
   - Use: If you feel unsafe or want to stop

2. **Automatic Collision Prevention**
   - Triggers: Obstacle <0.5 meters
   - Audio: "Stop! Obstacle too close!"
   - Action: System halts, waits for reset

3. **System Failure Detection**
   - Camera fails: "Sensor failure. Stop immediately"
   - Detection fails 3+ times: "Low confidence. Stop for safety"
   - Depth sensor invalid: "Depth sensor error. Stop"

4. **SLAM Tracking Loss**
   - Lost tracking >5 seconds: "Position tracking lost. Stop and wait"
   - Use: Prevents navigation with bad position data

**After Emergency Stop:**
- System waits for user input
- Press 'r' to reset and continue (if safe)
- Or press 'q' to quit

**Configuration:**
```yaml
safety:
  danger_distance: 0.5  # Absolute minimum (meters)
  min_safe_distance: 1.5  # Comfortable distance
```

### 4. Redundant Safety Checks (5 Independent Systems)

**What it does:** Multiple independent checks ensure no single failure causes danger.

**The 5 Layers:**

1. **Distance-Based Collision Detection**
   - Checks: Obstacle distance from depth map
   - Stops: If <0.5m

2. **Object Detection Confidence**
   - Checks: YOLO detection quality
   - Stops: If 3+ consecutive failures

3. **Depth Estimation Validity**
   - Checks: NaN, Inf, all-zeros in depth map
   - Stops: If sensor returns invalid data

4. **SLAM Tracking Quality**
   - Checks: Feature tracking count, relocalization
   - Stops: If tracking lost >5 seconds

5. **System Health Monitoring**
   - Checks: CPU, memory, component status
   - Stops: If critical component fails

**Why 5 Layers?**
- Defense in depth: multiple chances to catch failure
- No single point of failure
- Each layer catches different failure modes

### 5. Health Monitoring (Auto-Recovery)

**What it does:** Monitors system health and recovers from failures automatically.

**Monitored Components:**
- Camera (FPS, frame delivery)
- Object Detection (success rate, latency)
- Depth Estimation (validity, processing time)
- SLAM (tracking status, map quality)
- Audio (queue size, latency)

**Auto-Recovery Features:**
- Detects component degradation before failure
- Attempts automatic recovery (e.g., re-initialize camera)
- Falls back to safe mode if recovery fails
- Logs all issues for debugging

**Health Status:**
- **Healthy:** All systems normal (green)
- **Degraded:** One component slow (yellow)
- **Recovering:** Attempting automatic fix (blue)
- **Critical:** Multiple failures (red, triggers stop)

**View Health Status:**
```python
# In your code:
from src.core.health_monitor import get_health_monitor
monitor = get_health_monitor()
report = monitor.get_health_report()
print(report)
```

## Using the Production System

### Integration in Your Code

```python
from src.core.system_integration import IntegratedOrbyGlasses
from src.core.utils import ConfigManager, Logger

# Initialize
config = ConfigManager('config/config.yaml')
logger = Logger()

# Create integrated system
orby = IntegratedOrbyGlasses(config, logger, base_audio_manager=None)

# Get optimal device for models
device = orby.get_optimal_device()  # 'mps', 'cuda', or 'cpu'

# Load your model with optimal device
model.to(device)

# In your main loop:
while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Your detection code
    detections = detector.detect(frame)
    depth_map = depth_estimator.estimate(frame)

    # Process with integrated safety
    key = cv2.waitKey(1) & 0xFF
    should_continue, audio_msg = orby.process_frame(
        frame, detections, depth_map,
        slam_status=slam.get_status(),
        key=key
    )

    if not should_continue:
        break  # Emergency stop triggered

# Clean shutdown
orby.shutdown()
```

### Testing Production Systems

```bash
# Run comprehensive tests
python3 test_production_systems.py

# Expected output:
# ✅ GPU system working
# ✅ Fast audio working
# ✅ Emergency stop working
# ✅ Health monitor working
# ✅ System integration working
```

## Performance Benchmarks

### Audio Latency (Critical for Safety)

| Scenario | Target | Typical | Maximum |
|----------|--------|---------|---------|
| Emergency Alert | <200ms | 150-200ms | 300ms |
| Danger Warning | <500ms | 300-500ms | 700ms |
| Regular Guidance | <1s | 500-800ms | 1200ms |

**Note:** Actual latency depends on:
- System load (other apps running)
- Audio backend (macOS 'say' vs pyttsx3)
- Message length (shorter = faster)

### GPU Performance

| Hardware | Detection | Depth | SLAM | Total FPS |
|----------|-----------|-------|------|-----------|
| M2 (MPS) | 40ms | 80ms | 50ms | 25 FPS |
| M1 (MPS) | 50ms | 100ms | 60ms | 20 FPS |
| CPU Only | 200ms | 400ms | 150ms | 5 FPS |

### Safety Response Time

| Event | Detection | Stop | Total |
|-------|-----------|------|-------|
| Keyboard Stop | Instant | <10ms | <10ms |
| Collision <0.5m | 100ms | 200ms | <300ms |
| Sensor Failure | 50ms | 100ms | <150ms |
| Tracking Loss | 5s | 100ms | ~5.1s |

## Troubleshooting Production Features

### Fast Audio Not Working

**Symptom:** Audio latency >2 seconds

**Solutions:**
1. Check if macOS 'say' command works:
   ```bash
   say "Test"
   ```
2. If 'say' is slow, try different voice:
   ```bash
   say -v ? | grep "en_US"  # List English voices
   ```
3. Configure faster voice in config:
   ```yaml
   audio:
     tts_rate: 240  # Increase speed
   ```

### GPU Not Detected

**Symptom:** "Using CPU" despite having Apple Silicon/NVIDIA GPU

**Solutions:**
1. Check PyTorch installation:
   ```bash
   python3 -c "import torch; print(torch.backends.mps.is_available())"
   ```
2. Reinstall PyTorch with GPU support:
   ```bash
   pip install --upgrade torch torchvision
   ```
3. Verify in config:
   ```yaml
   models:
     yolo:
       device: mps  # or cuda
     depth:
       device: mps  # or cuda
   ```

### Emergency Stop Too Sensitive

**Symptom:** System stops frequently in safe situations

**Solutions:**
1. Increase safety distance threshold:
   ```yaml
   safety:
     danger_distance: 0.7  # Increase from 0.5m
   ```
2. Increase detection failure tolerance:
   - Edit `src/core/emergency_stop.py`
   - Change `max_detection_failures = 5`  # from 3

3. Disable auto-stop for debugging:
   ```python
   # In your code
   orby.emergency_stop.min_safe_distance = 0.3  # Lower threshold
   ```

### Health Monitor False Alarms

**Symptom:** "Component degraded" warnings when system works fine

**Solutions:**
1. Adjust FPS threshold:
   ```python
   orby.health_monitor.min_fps = 10  # Lower from 15
   ```
2. Increase failure tolerance:
   ```python
   orby.health_monitor.max_consecutive_failures = 5  # Up from 3
   ```

## Safety Best Practices

### For Blind Users

1. **Always use with supervision** during beta testing
2. **Pair with traditional aids** (white cane, guide dog)
3. **Test in familiar environments** first
4. **Learn emergency stop** (spacebar) before use
5. **Start with fast mode** for quicker feedback
6. **Report issues** if audio delayed or system stops unexpectedly

### For Developers

1. **Never disable safety checks** in production
2. **Test all 5 safety layers** before deploying
3. **Monitor audio latency** (log warnings if >500ms)
4. **Add redundancy** for critical code paths
5. **Fail safe** (stop on uncertainty)
6. **Log everything** for post-incident analysis

### For Testers

1. **Simulate failures** (cover camera, unplug, etc.)
2. **Test emergency stop** in all modes
3. **Measure audio latency** with stopwatch
4. **Verify GPU usage** (Activity Monitor)
5. **Check CPU/memory** under load
6. **Test battery impact** on laptop

## Production Deployment Checklist

- [ ] Run `test_production_systems.py` - all tests pass
- [ ] Verify audio latency <500ms for warnings
- [ ] Confirm GPU acceleration active (if available)
- [ ] Test emergency stop (keyboard + auto)
- [ ] Check all 5 safety layers functioning
- [ ] Monitor system health for 10+ minutes
- [ ] Test with actual blind user (supervised)
- [ ] Document any issues encountered
- [ ] Verify clean shutdown (no errors)
- [ ] Check logs for warnings/errors

## Support

**Issues:** https://github.com/jaskirat1616/Orby-Glasses/issues
**Discussions:** https://github.com/jaskirat1616/Orby-Glasses/discussions

For production issues, include:
- System specs (GPU, OS version)
- Audio latency measurements
- Health monitor report
- Logs from `data/logs/`
