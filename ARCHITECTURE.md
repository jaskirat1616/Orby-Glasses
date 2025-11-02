# OrbyGlasses Architecture

System design and component overview for OrbyGlasses.

## System Overview

OrbyGlasses is a real-time AI navigation system designed to help blind and visually impaired users navigate safely. The system processes camera input through multiple AI models to detect objects, estimate distances, track position, and provide audio guidance.

```
┌─────────────────────────────────────────────────────────────────┐
│                        OrbyGlasses System                       │
└─────────────────────────────────────────────────────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │   Camera Input      │
                    │   (OpenCV)          │
                    └──────────┬──────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   ┌────▼────┐          ┌─────▼─────┐         ┌─────▼─────┐
   │ Object  │          │   Depth   │         │   SLAM    │
   │Detection│          │Estimation │         │  System   │
   │(YOLOv11)│          │(DepthV2)  │         │ (pySLAM)  │
   └────┬────┘          └─────┬─────┘         └─────┬─────┘
        │                     │                     │
        └─────────┬───────────┴───────────┬─────────┘
                  │                       │
           ┌──────▼──────┐         ┌─────▼──────┐
           │   Safety    │         │Navigation  │
           │   System    │         │   System   │
           └──────┬──────┘         └─────┬──────┘
                  │                      │
                  └──────────┬───────────┘
                             │
                      ┌──────▼──────┐
                      │    Audio    │
                      │   Manager   │
                      │   (TTS)     │
                      └─────────────┘
```

## Core Components

### 1. Main Application (`src/main.py`)

**Purpose**: Orchestrates all components and manages the main processing loop.

**Key Classes**:
- `OrbyGlasses`: Main application class (1700+ lines)
  - Initializes all components
  - Manages frame processing pipeline
  - Handles user input and shutdown

**Processing Flow**:
```python
def process_frame(frame):
    1. Detect objects (YOLOv11)
    2. Estimate depth (Depth Anything V2)
    3. Update SLAM position (pySLAM)
    4. Check safety zones (Safety System)
    5. Plan navigation (Navigation System)
    6. Generate audio feedback (Audio Manager)
    7. Update visualization (UI)
    return annotated_frame
```

**Frame Rate**:
- Standard mode: 15-25 FPS
- Fast mode: 20-30 FPS

### 2. Object Detection (`src/core/detection.py`)

**Purpose**: Identify objects in camera frames using YOLOv11.

**Key Classes**:
- `YOLODetector`: Wrapper for Ultralytics YOLOv11

**Input**: RGB frame (320x240 or 640x480)
**Output**: List of detected objects with:
- Class name (car, person, chair, etc.)
- Bounding box coordinates
- Confidence score (0-1)

**Performance**:
- Model: YOLOv11n (nano, ~6MB)
- Inference time: 20-50ms per frame
- Device: Apple MPS (GPU) or CPU
- Classes: 80 COCO objects

**Configuration**:
```yaml
models:
  yolo:
    model: yolov11n.pt
    confidence: 0.55  # Detection threshold
    device: mps       # Apple Silicon GPU
```

### 3. Depth Estimation (`src/core/depth_anything_v2.py`)

**Purpose**: Estimate distance to each pixel in the image.

**Key Classes**:
- `DepthAnythingV2`: Monocular depth estimation wrapper

**Input**: RGB frame
**Output**: Depth map (metric depth in meters)

**Model Details**:
- Architecture: Vision Transformer (ViT)
- Size: Depth-Anything-V2-Small (~100MB)
- Range: 0-10 meters
- Resolution: Same as input frame

**Performance**:
- Inference time: 70-200ms per frame
- Frame skipping: Process every Nth frame
- Half-precision (FP16): Enabled for speed
- Caching: Reuse depth maps for unchanged frames

**Distance Calculation**:
```python
def get_object_distance(bbox, depth_map):
    # Extract region of interest
    roi = depth_map[y1:y2, x1:x2]
    # Use median depth (robust to outliers)
    distance = np.median(roi)
    return distance
```

### 4. SLAM System (`src/navigation/pyslam_live.py`)

**Purpose**: Track camera position and build 3D map of environment.

**Integration**: pySLAM (professional monocular SLAM)

**Key Classes**:
- `LivePySLAM`: Wrapper for pySLAM library
- `PySLAMConfig`: Configuration manager

**Features**:
- **Feature Extraction**: ORB/SIFT/SuperPoint
- **Feature Tracking**: KLT optical flow
- **Map Building**: Sparse 3D reconstruction
- **Loop Closure**: Recognize previously visited locations
- **Relocalization**: Recover from tracking loss
- **Bundle Adjustment**: Optimize camera poses and map points

**Data Structures**:
```python
# Camera pose (6-DOF)
pose = {
    'position': [x, y, z],     # meters
    'rotation': [rx, ry, rz],  # radians
    'timestamp': time.time()
}

# Map point
point = {
    'position': [x, y, z],     # 3D coordinates
    'descriptor': [128-D],     # ORB descriptor
    'observations': [...]      # Keyframes that see this point
}
```

**Modes**:
1. **Full SLAM**: Position tracking + map building
2. **VO (Visual Odometry)**: Position tracking only (faster)

**Performance**:
- Features per frame: 2000-8000 ORB
- Keyframes: 100-500 stored
- Processing time: 50-150ms per frame
- Tracking loss recovery: Automatic relocalization

### 5. Safety System (`src/core/safety_system.py`)

**Purpose**: Monitor surroundings and issue warnings based on distance.

**Key Classes**:
- `SafetySystem`: Danger zone detection and prioritization

**Safety Zones**:
```python
DANGER = distance < 1.0m       # Red - immediate warning
CAUTION = 1.0m ≤ distance < 2.5m  # Yellow - advisory
SAFE = distance ≥ 2.5m           # Green - quiet
```

**Priority System**:
1. Closest object first
2. Dangerous objects (car, stairs) prioritized
3. Moving objects prioritized (future enhancement)

**Warning Generation**:
```python
def generate_warning(object, distance, direction):
    if distance < DANGER:
        return f"Stop! {object} ahead. {distance:.1f} meters. Go {direction}."
    elif distance < CAUTION:
        return f"{object} on your {direction}. {distance:.1f} meters away."
    else:
        return "Path is clear."
```

### 6. Navigation System (`src/navigation/indoor_navigation.py`)

**Purpose**: Provide turn-by-turn navigation to saved locations.

**Key Classes**:
- `IndoorNavigator`: Path planning and guidance

**Features**:
- Save landmarks (e.g., "kitchen", "desk")
- Calculate routes using SLAM map
- Turn-by-turn directions
- Distance to destination

**Navigation Algorithm**:
```python
def navigate_to(destination):
    current_pos = slam.get_position()
    path = a_star(current_pos, destination, slam.map)

    for waypoint in path:
        direction = calculate_bearing(current_pos, waypoint)
        distance = calculate_distance(current_pos, waypoint)
        speak(f"Turn {direction}. Walk {distance:.1f} meters.")
        wait_for_waypoint_reached(waypoint)
```

### 7. Audio Manager (`src/core/utils.py`)

**Purpose**: Convert text to speech for user guidance.

**Key Classes**:
- `AudioManager`: TTS wrapper with queueing

**Engine**: pyttsx3 (offline TTS)

**Features**:
- Message queueing (prevent overlap)
- Priority messages (interrupt queue)
- Rate control (words per minute)
- Volume control

**Performance**:
- Text-to-speech latency: 0.8-2 seconds
- Queue processing: FIFO
- Interruption: Danger messages only

**Configuration**:
```yaml
audio:
  enabled: true
  rate: 175                      # words per minute
  volume: 0.9                    # 0-1
  min_time_between_warnings: 2.0 # seconds
```

### 8. Configuration Manager (`src/core/utils.py`)

**Purpose**: Load and validate configuration from YAML.

**Key Classes**:
- `ConfigManager`: Singleton config loader

**Configuration Files**:
- `config/config.yaml`: Standard mode
- `config/config_fast.yaml`: Fast mode

**Validation**:
- Check required fields
- Validate ranges (e.g., 0 ≤ confidence ≤ 1)
- Set defaults for missing values

### 9. Visualization (`src/visualization/`)

**Purpose**: Display annotated video and navigation UI.

**Components**:

**Robot UI** (`robot_ui.py`):
- Annotated camera feed
- Object bounding boxes
- Distance labels
- FPS counter

**Navigation Panel** (`advanced_nav_panel.py`):
- Overhead compass view
- Position tracking
- Trajectory visualization
- Map overlay

**Depth Visualizer** (`depth_visualizer_2025.py`):
- Depth map heatmap
- Distance color coding
- 3D point cloud view

## Data Flow

### Complete Frame Processing Pipeline

```
Camera Frame (640x480 RGB)
    │
    ├──> Object Detection (YOLOv11)
    │        │
    │        └──> Detections: [(class, bbox, conf), ...]
    │
    ├──> Depth Estimation (Depth Anything V2)
    │        │
    │        └──> Depth Map: (640x480 float32)
    │
    ├──> SLAM Processing (pySLAM)
    │        │
    │        ├──> Camera Pose: (x, y, z, rx, ry, rz)
    │        └──> Map Points: [(x, y, z), ...]
    │
    └──> Merge Data
             │
             ├──> Safety Analysis
             │        │
             │        └──> Warnings: ["Stop! Car ahead.", ...]
             │
             ├──> Navigation Planning
             │        │
             │        └──> Directions: ["Turn left.", ...]
             │
             └──> Audio Output (TTS)
                      │
                      └──> Spoken Guidance
```

### Timing Budget (30 FPS target = 33ms per frame)

| Component | Time | Percentage |
|-----------|------|------------|
| Camera capture | 5ms | 15% |
| Object detection | 20-50ms | 60% |
| Depth estimation | 70-200ms | (every Nth frame) |
| SLAM tracking | 50-150ms | (async) |
| Safety analysis | 5ms | 15% |
| Audio generation | 1-2s | (async) |
| Visualization | 10ms | 30% |

**Optimization Strategies**:
1. **Frame skipping**: Depth every 2-5 frames
2. **Async processing**: SLAM and audio run in separate threads
3. **Half-precision**: FP16 inference for depth model
4. **GPU acceleration**: MPS backend for Apple Silicon

## Directory Structure

```
OrbyGlasses/
├── src/
│   ├── main.py                    # Main application (entry point)
│   │
│   ├── core/                      # Core processing modules
│   │   ├── detection.py           # Object detection (YOLOv11)
│   │   ├── depth_anything_v2.py   # Depth estimation
│   │   ├── utils.py               # Config, logging, audio
│   │   ├── safety_system.py       # Danger detection
│   │   ├── error_handler.py       # Error recovery
│   │   ├── smart_cache.py         # Result caching
│   │   ├── narrative.py           # Natural language generation
│   │   ├── echolocation.py        # Audio-based distance sensing
│   │   ├── llm_manager.py         # Local LLM integration (Ollama)
│   │   ├── adaptive_system.py     # Dynamic parameter tuning
│   │   └── yolo_world_detector.py # Open-vocabulary detection
│   │
│   ├── navigation/                # Navigation and SLAM
│   │   ├── pyslam_live.py         # pySLAM integration (PRIMARY)
│   │   ├── pyslam_vo_integration.py # Visual odometry mode
│   │   ├── indoor_navigation.py   # Path planning
│   │   └── simple_slam.py         # Fallback SLAM
│   │
│   ├── visualization/             # Display and UI
│   │   ├── robot_ui.py            # Main camera view with annotations
│   │   ├── advanced_nav_panel.py  # Navigation panel (compass, map)
│   │   ├── depth_visualizer_2025.py # Depth visualization
│   │   └── fast_depth.py          # Optimized depth rendering
│   │
│   └── features/                  # Optional/experimental features
│       ├── conversation.py        # Voice commands
│       ├── scene_understanding.py # Scene descriptions (LLM)
│       ├── trajectory_prediction.py # Predict moving objects (GNN)
│       ├── occupancy_grid_3d.py   # Voxel mapping
│       ├── mapping3d.py           # 3D map visualization
│       ├── haptic_feedback_2025.py # Vibration feedback
│       └── social_navigation.py   # Human-aware path planning
│
├── config/
│   ├── config.yaml                # Standard configuration
│   └── config_fast.yaml           # Fast mode configuration
│
├── tests/                         # Test suite
│   ├── test_detection.py
│   ├── test_utils.py
│   ├── test_slam.py
│   └── test_integration.py
│
├── third_party/
│   └── pyslam/                    # pySLAM SLAM library (submodule)
│
├── models/                        # Downloaded AI models
│   └── [auto-downloaded]
│
├── data/
│   ├── logs/                      # Application logs
│   └── maps/                      # Saved SLAM maps
│
├── docs/                          # Documentation
│   ├── README.md
│   ├── SETUP.md
│   ├── TROUBLESHOOTING.md
│   ├── ARCHITECTURE.md (this file)
│   ├── CHANGELOG.md
│   ├── CONTRIBUTING.md
│   └── CODE_OF_CONDUCT.md
│
└── scripts/                       # Launch scripts
    ├── run_orby.sh                # Main launcher
    ├── run_vo_mode.sh             # VO mode
    ├── run_slam_mode.sh           # Full SLAM mode
    ├── install_pyslam.sh          # pySLAM installation
    └── switch_mode.sh             # Mode switcher
```

## Threading Model

OrbyGlasses uses a multi-threaded architecture for performance:

```
Main Thread:
    ├── Camera capture (blocking)
    ├── Object detection (blocking)
    └── UI rendering (blocking)

SLAM Thread:
    └── Feature tracking, mapping, loop closure

Depth Thread:
    └── Depth map estimation (every Nth frame)

Audio Thread:
    ├── TTS generation
    └── Audio playback queue

UI Thread (optional):
    └── 3D visualization (pySLAM viewer)
```

**Synchronization**:
- Thread-safe queues for inter-thread communication
- Locks for shared data structures (map, pose)
- Atomic flags for shutdown signaling

## Configuration System

### Hierarchical Configuration

```yaml
# Top-level categories
camera:          # Camera settings
models:          # AI model configuration
slam:            # SLAM parameters
safety:          # Safety zone thresholds
audio:           # Audio feedback
features:        # Optional feature flags
debug:           # Debugging options
```

### Feature Flags

```yaml
features:
  conversation: false        # Voice commands (experimental)
  mapping3d: false           # 3D map visualization
  occupancy_grid_3d: false   # Voxel mapping
  trajectory_prediction: false # Moving object prediction
  haptic_feedback: false     # Vibration feedback
  social_navigation: false   # Human-aware navigation
```

**Default**: Most experimental features are disabled for performance.

## Error Handling and Recovery

### Error Handler (`src/core/error_handler.py`)

**Strategy**:
1. **Graceful degradation**: Continue with reduced functionality
2. **Automatic recovery**: Retry failed operations
3. **User notification**: Inform user of issues via audio
4. **Fallback modes**: Switch to simpler algorithms if advanced ones fail

**Examples**:
```python
# SLAM tracking loss
if slam.tracking_lost():
    slam.try_relocalize()
    if still_lost():
        switch_to_vo_mode()  # Fallback to visual odometry

# Depth estimation failure
if depth_model.fails():
    use_cached_depth_map()
    if no_cache():
        estimate_from_object_size()  # Heuristic fallback

# Audio system failure
if audio.fails():
    display_text_warnings()  # Visual fallback
```

## Performance Optimization

### Key Optimizations

1. **GPU Acceleration**: Apple MPS for YOLOv11 and Depth models
2. **Half-Precision Inference**: FP16 for depth model (2x speedup)
3. **Frame Skipping**: Depth every 2-5 frames (5x speedup)
4. **Async Processing**: SLAM and audio in separate threads
5. **Smart Caching**: Reuse results for unchanged frames
6. **Resolution Scaling**: 320x240 for speed, 640x480 for quality

### Profiling Results

```
Standard Mode (320x240, 20 FPS):
- Camera capture:     5ms   (10%)
- Object detection:   25ms  (50%)
- Depth estimation:   100ms (every 3rd frame)
- SLAM tracking:      80ms  (async)
- Safety/navigation:  5ms   (10%)
- Rendering:          10ms  (20%)
- Total:              45ms/frame (22 FPS)

Fast Mode (320x240, 25 FPS):
- Camera capture:     5ms   (12%)
- Object detection:   20ms  (50%)
- Depth estimation:   100ms (every 5th frame)
- SLAM:               disabled
- Safety:             3ms   (7%)
- Rendering:          8ms   (20%)
- Total:              36ms/frame (28 FPS)
```

## Dependencies

### Core Libraries

| Library | Purpose | Version |
|---------|---------|---------|
| OpenCV | Camera, image processing | 4.8+ |
| PyTorch | Deep learning backend | 2.0+ |
| Ultralytics | YOLOv11 detection | 8.0+ |
| Transformers | Depth Anything V2 | 4.36+ |
| pySLAM | SLAM system | latest |
| pyttsx3 | Text-to-speech | 2.90+ |
| NumPy | Numerical computing | 1.24+ |
| PyYAML | Configuration parsing | 6.0+ |

### Optional Libraries

| Library | Purpose |
|---------|---------|
| Ollama | Local LLM (scene understanding) |
| SpeechRecognition | Voice commands |
| PyAudio | Audio capture |

## Future Architecture Improvements

### Planned Enhancements

1. **Microservices Architecture**: Split into independent services
2. **Message Queue**: RabbitMQ/Redis for inter-component communication
3. **Database**: PostgreSQL for map persistence and user data
4. **REST API**: Web interface for configuration and monitoring
5. **WebRTC**: Remote assistance mode
6. **Mobile App**: iOS/Android companion app
7. **Cloud Sync**: Optional cloud backup for maps (privacy-preserving)

### Scalability Considerations

- **Horizontal scaling**: Multiple cameras, distributed processing
- **Edge deployment**: Raspberry Pi, NVIDIA Jetson
- **Cloud offloading**: Heavy processing (optional, privacy-aware)

---

## Glossary

- **SLAM**: Simultaneous Localization and Mapping
- **VO**: Visual Odometry (position tracking without mapping)
- **ORB**: Oriented FAST and Rotated BRIEF (feature detector)
- **MPS**: Metal Performance Shaders (Apple GPU API)
- **TTS**: Text-to-Speech
- **FP16**: 16-bit floating point (half-precision)
- **Keyframe**: Important camera frame used for mapping
- **Loop Closure**: Detecting previously visited locations
- **Relocalization**: Recovering position after tracking loss

## References

- [pySLAM Documentation](https://github.com/luigifreda/pyslam)
- [YOLOv11 Paper](https://docs.ultralytics.com)
- [Depth Anything V2](https://depth-anything-v2.github.io/)
- [OpenCV Documentation](https://docs.opencv.org/)
