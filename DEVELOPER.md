# Developer Documentation

Technical documentation for developers building on OrbyGlasses.

## Architecture Overview

OrbyGlasses is a real-time monocular vision-based navigation system with multi-modal perception pipeline.

### Core Pipeline

```
Camera Input (640x480 @ 30fps)
    │
    ├─→ Object Detection (YOLOv11n)
    │   └─→ COCO 80 classes, confidence threshold 0.55
    │
    ├─→ Depth Estimation (Depth Anything V2 Small)
    │   └─→ Monocular depth, 0-10m range, FP16 inference
    │
    ├─→ SLAM (pySLAM)
    │   └─→ ORB features, loop closure, bundle adjustment
    │
    └─→ Safety & Navigation
        └─→ Distance-based warnings, emergency stop, health monitoring
```

### Technology Stack

**Computer Vision:**
- **Detection:** Ultralytics YOLOv11n (6MB model)
- **Depth:** Depth-Anything-V2-Small (100MB model, ViT-based)
- **SLAM:** pySLAM (ORB-SLAM2 Python implementation)

**Frameworks:**
- **ML:** PyTorch 2.0+, torchvision
- **CV:** OpenCV 4.8+
- **Audio:** pyttsx3 (macOS 'say' command)
- **Numerical:** NumPy 1.24+

**Performance:**
- **GPU:** MPS (Apple Silicon) or CUDA (NVIDIA)
- **Precision:** FP16 for depth model, FP32 for YOLO
- **Target:** 20-30 FPS on M1/M2, 5-10 FPS on CPU

## System Components

### 1. Detection Pipeline (`src/core/detection.py`)

**Class:** `DetectionPipeline`

**Responsibilities:**
- Load YOLOv11n model
- Perform object detection on frames
- Filter by confidence threshold
- Return bounding boxes with class labels

**Key Methods:**
```python
def detect(self, frame: np.ndarray) -> List[Dict]:
    """
    Detect objects in frame.

    Args:
        frame: RGB image (HxWx3)

    Returns:
        List of detections: [{'class': str, 'confidence': float, 'bbox': (x1,y1,x2,y2)}, ...]
    """
```

**Configuration:**
```yaml
models:
  yolo:
    model: yolov11n.pt
    confidence: 0.55
    device: mps  # or cuda, cpu
    half_precision: true
```

### 2. Depth Estimation (`src/core/depth_anything_v2.py`)

**Class:** `DepthAnythingV2`

**Method:**
- Monocular depth estimation using ViT backbone
- Metric depth in meters (0-10m range)
- Processes 256x256 patches for efficiency

**Key Methods:**
```python
def estimate(self, frame: np.ndarray) -> np.ndarray:
    """
    Estimate depth map.

    Args:
        frame: RGB image (HxWx3)

    Returns:
        Depth map (HxW) in meters
    """
```

**Performance Optimization:**
- Frame skipping (process every Nth frame)
- Half-precision (FP16) inference
- Smart caching (reuse recent depth maps)

### 3. SLAM System (`src/navigation/pyslam_live.py`)

**Class:** `LivePySLAM`

**Features:**
- Visual odometry and mapping
- ORB feature extraction (5000-8000 features)
- Loop closure detection
- Bundle adjustment optimization
- 3D map visualization (Pangolin)

**Key Methods:**
```python
def track(self, frame: np.ndarray) -> Dict:
    """
    Track camera pose and update map.

    Args:
        frame: Grayscale image (HxW)

    Returns:
        {'pose': (x,y,z,rx,ry,rz), 'tracking': bool, 'num_features': int}
    """
```

**Tunable Parameters:**
```yaml
slam:
  orb_features: 5000
  scale_factor: 1.2
  n_levels: 8
  loop_closure: true
  relocalization_threshold: 0.75
```

### 4. Production Systems

#### Fast Audio (`src/core/fast_audio.py`)

**Class:** `FastAudioManager`

**Latency Targets:**
- Emergency: <200ms
- Warning: <500ms
- Info: <1s

**Priority Queue:**
```python
class AudioPriority(Enum):
    EMERGENCY = 0  # Immediate, interrupts
    DANGER = 1     # High priority
    WARNING = 2    # Medium priority
    INFO = 3       # Normal
    LOW = 4        # Background
```

**Usage:**
```python
from src.core.fast_audio import FastAudioManager, emergency_alert

audio = FastAudioManager(rate=220, voice="Samantha")
emergency_alert(audio, "Stop! Obstacle ahead")  # <200ms
```

#### Emergency Stop (`src/core/emergency_stop.py`)

**Class:** `EmergencyStopSystem`

**Stop Triggers:**
1. User input (spacebar, 'q', ESC)
2. Collision risk (distance < 0.5m)
3. Detection failure (3+ consecutive)
4. SLAM tracking loss (>5 seconds)
5. Sensor failure (NaN, Inf, zeros)

**Redundant Safety:**
```python
class RedundantSafetyChecker:
    """
    5 independent safety checks:
    1. Distance-based collision detection
    2. Object detection confidence monitoring
    3. Depth sensor validity checks
    4. SLAM tracking quality verification
    5. System health status monitoring
    """
```

#### GPU Acceleration (`src/core/gpu_check.py`)

**Auto-Detection:**
```python
from src.core.gpu_check import get_optimal_device, OPTIMAL_SETTINGS

device = get_optimal_device()  # 'mps', 'cuda', or 'cpu'
model.to(device)

# Get hardware-specific settings
settings = OPTIMAL_SETTINGS
# {'batch_size': 4, 'use_half_precision': True, ...}
```

**Speedup Factors:**
- MPS (Apple Silicon): 5x vs CPU
- CUDA (NVIDIA): 10x vs CPU

#### Health Monitor (`src/core/health_monitor.py`)

**Class:** `HealthMonitor`

**Monitored Metrics:**
- FPS (frames per second)
- Latency (ms per component)
- Error rate (failures/attempts)
- Memory usage (MB)
- CPU usage (%)

**Auto-Recovery:**
```python
monitor = HealthMonitor(logger, audio_manager)
monitor.register_component('detection', recovery_callback=reinit_detector)
monitor.start_monitoring()

# Automatic recovery if component fails
```

## Integration Guide

### Basic Integration

```python
from src.core.system_integration import IntegratedOrbyGlasses
from src.core.utils import ConfigManager, Logger
import cv2

# Initialize
config = ConfigManager('config/config.yaml')
logger = Logger()
orby = IntegratedOrbyGlasses(config, logger)

# Get optimal device
device = orby.get_optimal_device()  # Auto-configured
print(f"Using device: {device}")  # mps, cuda, or cpu

# Load your models with optimal device
model.to(device)

# Main loop
camera = cv2.VideoCapture(0)
while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Your detection code
    detections = detector.detect(frame)
    depth_map = depth_estimator.estimate(frame)

    # Process with all safety systems
    key = cv2.waitKey(1) & 0xFF
    should_continue, audio_msg = orby.process_frame(
        frame, detections, depth_map,
        slam_status={'tracking': True, 'features': 1000},
        key=key
    )

    if not should_continue:
        print(f"Emergency stop: {audio_msg}")
        break

    # Display (optional)
    cv2.imshow('OrbyGlasses', frame)

# Cleanup
orby.shutdown()
camera.release()
cv2.destroyAllWindows()
```

### Advanced Integration

#### Custom Detection Models

```python
from src.core.detection import DetectionPipeline

class CustomDetector(DetectionPipeline):
    def __init__(self, config):
        super().__init__(config)
        # Load your custom model
        self.model = YourCustomModel()

    def detect(self, frame):
        # Your detection logic
        results = self.model(frame)
        return self.format_results(results)
```

#### Custom Audio System

```python
from src.core.fast_audio import AudioPriority

class CustomAudio:
    def speak(self, text, priority=AudioPriority.INFO):
        # Your audio implementation
        pass

# Use in IntegratedOrbyGlasses
orby = IntegratedOrbyGlasses(config, logger, custom_audio)
```

#### Custom Safety Checks

```python
from src.core.emergency_stop import EmergencyStopSystem

estop = EmergencyStopSystem(audio, logger)

# Add custom stop condition
def check_custom_condition():
    if your_condition:
        estop.trigger_stop(StopReason.USER_DEFINED, "Custom stop reason")

estop.register_stop_callback(on_custom_stop)
```

## Performance Tuning

### Accuracy vs Speed Tradeoffs

**High Accuracy (15-20 FPS):**
```yaml
camera:
  width: 640
  height: 480

models:
  yolo:
    confidence: 0.45  # Lower = more detections

slam:
  orb_features: 8000  # More features
  loop_closure: true
```

**High Speed (25-30 FPS):**
```yaml
camera:
  width: 320
  height: 240

models:
  yolo:
    confidence: 0.60  # Higher = fewer detections

slam:
  orb_features: 2000  # Fewer features
  visualization: false

performance:
  depth_skip_frames: 3  # Process every 3rd frame
```

### Memory Optimization

```yaml
slam:
  max_keyframes: 100
  max_map_points: 5000
  cull_redundant_keyframes: true

depth:
  cache_size: 10  # Or 0 to disable
```

### GPU Optimization

```python
# Use mixed precision
torch.set_float32_matmul_precision('high')

# Enable TF32 on CUDA
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Compile models (PyTorch 2.0+)
model = torch.compile(model, mode='max-autotune')
```

## API Reference

### Core Classes

#### `IntegratedOrbyGlasses`

Main integration class combining all systems.

**Methods:**
- `get_optimal_device() -> str`: Returns 'mps', 'cuda', or 'cpu'
- `get_optimal_settings() -> Dict`: Returns hardware-specific settings
- `process_frame(frame, detections, depth_map, slam_status, key) -> Tuple[bool, str]`
- `get_status_report() -> Dict`: Returns comprehensive system status
- `shutdown()`: Clean shutdown of all systems

#### `FastAudioManager`

Low-latency audio for safety warnings.

**Methods:**
- `speak(text: str, priority: AudioPriority)`: Queue audio
- `speak_immediate(text: str)`: Bypass queue (<200ms)
- `clear_queue()`: Clear all pending messages
- `get_latency_stats() -> Dict`: Returns latency metrics

#### `EmergencyStopSystem`

Multi-layer emergency stop.

**Methods:**
- `trigger_stop(reason: StopReason, message: str)`: Manual stop
- `check_collision_risk(distance: float) -> bool`: Auto-stop if too close
- `check_detection_health(success: bool) -> bool`: Monitor detection
- `check_tracking_health(active: bool) -> bool`: Monitor SLAM
- `reset()`: Reset and continue

#### `HealthMonitor`

System health monitoring.

**Methods:**
- `register_component(name: str, recovery_callback: Callable)`
- `update_component(name: str, **metrics)`
- `get_health_report() -> Dict`
- `start_monitoring()`: Start background thread
- `stop_monitoring()`: Stop monitoring

## Testing

### Unit Tests

```bash
# Test individual components
pytest tests/test_detection.py -v
pytest tests/test_utils.py -v
pytest tests/test_slam.py -v
```

### Integration Tests

```bash
# Test production systems
python3 test_production_systems.py
```

### Benchmarking

```python
import time
import numpy as np

# Benchmark detection
frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(100)]
start = time.time()
for frame in frames:
    detections = detector.detect(frame)
elapsed = time.time() - start
print(f"Detection: {elapsed/100*1000:.1f}ms per frame, {100/elapsed:.1f} FPS")

# Benchmark depth
start = time.time()
for frame in frames:
    depth = depth_estimator.estimate(frame)
elapsed = time.time() - start
print(f"Depth: {elapsed/100*1000:.1f}ms per frame, {100/elapsed:.1f} FPS")
```

## Configuration Reference

See `config/config.yaml` for full configuration options.

**Key Settings:**
- `camera.source`: Camera device index
- `models.yolo.confidence`: Detection threshold (0-1)
- `models.depth.skip_frames`: Process every Nth frame
- `slam.orb_features`: Number of features to extract
- `safety.danger_distance`: Emergency stop threshold (meters)
- `audio.tts_rate`: Speech rate (words per minute)

## Contributing

### Code Style

- Follow PEP 8
- Use type hints
- Document with docstrings
- Keep functions under 50 lines

### Adding New Features

1. Create feature in `src/features/`
2. Add config options to `config/config.yaml`
3. Add tests to `tests/`
4. Update this documentation
5. Submit pull request

### Debugging

Enable debug logging:
```yaml
logging:
  level: DEBUG
  log_file: data/logs/orbyglass.log
```

View logs:
```bash
tail -f data/logs/orbyglass.log
```

## Known Limitations

1. **Monocular depth**: Scale ambiguity, less accurate than stereo
2. **Indoor only**: SLAM works best indoors with visual features
3. **Lighting**: Requires adequate lighting (>100 lux)
4. **Texture**: Blank walls cause tracking loss
5. **Speed**: Fast movement (>1 m/s) may lose tracking
6. **macOS only**: Currently optimized for Apple Silicon

## Future Improvements

1. **Stereo depth**: Add stereo camera support for metric depth
2. **IMU fusion**: Integrate IMU for robust tracking
3. **Semantic mapping**: Build semantic 3D maps
4. **Path planning**: A* pathfinding for navigation
5. **Cross-platform**: Linux and Windows support
6. **Mobile**: iOS/Android app

## Resources

- **pySLAM**: https://github.com/luigifreda/pyslam
- **YOLOv11**: https://docs.ultralytics.com
- **Depth Anything V2**: https://depth-anything-v2.github.io/
- **PyTorch**: https://pytorch.org/docs

## Support

- **Issues**: https://github.com/jaskirat1616/Orby-Glasses/issues
- **Discussions**: https://github.com/jaskirat1616/Orby-Glasses/discussions
- **Email**: [Add email if desired]
