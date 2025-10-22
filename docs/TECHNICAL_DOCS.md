# OrbyGlasses Technical Documentation

## Project Overview

OrbyGlasses is an advanced bio-mimetic navigation system designed to assist visually impaired users in navigating their environment safely using computer vision, spatial audio, and AI. The system combines object detection, depth estimation, simultaneous localization and mapping (SLAM), and natural language processing to provide real-time navigation assistance through both audio and spatial cues.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        OrbyGlasses System                               │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│  │   Camera    │  │   Audio     │  │   Ollama    │  │   Models    │   │
│  │   Feed      │  │   System    │  │   LLMs      │  │   (YOLO,    │   │
│  │             │  │             │  │             │  │ Depth, Etc.)│   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│         │               │               │               │              │
│         ▼               ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Main Processing Pipeline                     │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────┐  │   │
│  │  │  Detection  │  │   Depth     │  │    SLAM     │  │ Audio  │  │   │
│  │  │  Pipeline   │  │ Estimation  │  │    (3D)     │  │ Cues   │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────┘  │   │
│  │         │               │               │               │       │   │
│  │         ▼               ▼               ▼               ▼       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────┐  │   │
│  │  │ Navigation  │  │   Mapping   │  │   Spatial   │  │  Text  │  │   │
│  │  │   Logic     │  │    3D       │  │   Audio     │  │  to    │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │ Speech │  │   │
│  │         │               │               │          └────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         │               │               │               │              │
│         ▼               ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                  Output & Interaction                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────┐  │   │
│  │  │  Visual     │  │   Audio     │  │   Voice     │  │  UI    │  │   │
│  │  │ Annotation  │  │   Output    │  │  Control    │  │Output  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Main Application (src/main.py)

The main application orchestrates all components of the OrbyGlasses system. Key responsibilities include:

- **Component Initialization**: Initializes all subsystems including detection, depth estimation, audio, and AI components
- **Real-time Processing**: Processes camera frames at high frequency (targeted 20 FPS)
- **State Management**: Manages system state, frame counts, timing, and component interactions
- **User Interface**: Provides visual feedback through annotated video feed with overlays for FPS, object counts, and navigation status
- **Safety Monitoring**: Continuously monitors for dangerous obstacles (within 1.5m threshold)

### 2. Object Detection & Depth Estimation

#### Detection Pipeline (src/detection.py)

The detection system combines two AI models:

**YOLOv12 Object Detector:**
- Uses YOLOv12n model for real-time object detection
- Optimized for Apple Silicon (MPS) acceleration
- Detects 80+ object classes from COCO dataset
- Configurable confidence thresholds (default: 0.65)
- Priority classes for navigation (person, vehicle, obstacle types)

**Depth Estimation:**
- Uses Depth Anything V2 Small model for monocular depth estimation
- Converts RGB images to depth maps for spatial understanding
- Implements fallback depth estimation for performance
- Maps normalized depth values (0-1) to real-world distances (0.3-15m)

**Navigation Summary:**
- Categorizes objects by distance: danger zone (<1.5m), caution zone (1.5-3m), safe zone (>3m)
- Provides closest object information
- Identifies path clearance status

### 3. Audio & Spatial Audio System

#### Echolocation & Audio Cues (src/echolocation.py)

The audio system implements multiple layers of audio feedback:

**Adaptive Audio Beaconing:**
- 440Hz chime for safe path indicators
- 880Hz warning tone for obstacles
- Directional panning based on object position
- Priority-based alert system for imminent dangers

**Spatial Audio:**
- Simulates bio-mimetic echolocation
- Creates binaural audio cues based on object positions
- Distance-based frequency mapping (closer = higher pitch)
- Pan-based spatial positioning for multi-object scenarios

**Audio Management:**
- Integration with macOS 'say' command for text-to-speech
- Thread-safe audio queuing system
- Priority-based message system for danger alerts

### 4. SLAM & 3D Mapping

#### Monocular SLAM (src/slam.py)

The SLAM (Simultaneous Localization and Mapping) system provides:

**Visual Odometry:**
- ORB feature detection and tracking
- Essential matrix-based pose estimation
- Motion model for smooth tracking between frames
- Scale estimation for monocular setup

**Map Building:**
- Keyframe-based mapping
- Map point storage and management
- Pose graph optimization (simplified)

**Localization:**
- Real-time camera position tracking
- Trajectory logging and visualization
- Scale-aware movement estimation

#### 3D Occupancy Grid (src/occupancy_grid_3d.py)

**Volumetric Mapping:**
- Sparse voxel grid implementation
- 3D ray casting from depth maps
- Log-odds Bayesian update for occupancy
- Real-time visualization with interactive controls

### 5. AI & Navigation Logic

#### Narrative Generation (src/narrative.py)

**Multimodal AI Processing:**
- Moondream vision model for scene understanding
- Gemma3:4b for navigation narrative generation
- Context-aware prompt engineering
- Relatable distance terms ("arm's length", "one step")

**Navigation Guidance:**
- Actionable directional guidance
- Context-aware descriptions
- Temporal consistency through history tracking
- Fallback mechanisms for offline scenarios

#### Conversational Navigation (src/conversation.py)

**Voice Interface:**
- Speech recognition using Google Speech API
- Configurable activation phrases ("hey orby")
- Background listening with activation detection
- Non-blocking audio processing

**Natural Language Processing:**
- Ollama-based conversation management
- Goal-oriented navigation commands
- Location saving and recall functionality
- Context-aware response generation

### 6. Social Navigation (src/social_navigation.py)

**Crowd Analysis:**
- Region-specific social norms (US, UK, Japan conventions)
- Gap detection in crowds
- Density analysis (sparse, moderate, dense)
- Suggested path recommendations

**Social Norms Implementation:**
- Right-side convention for US
- Left-side convention for UK/Japan
- Dynamic path planning based on crowd layout
- Safety margin maintenance

### 7. Indoor Navigation (src/indoor_navigation.py)

**Path Planning:**
- A* algorithm implementation
- Grid-based navigation
- Occupancy grid integration
- Dynamic replanning for obstacles

**Location Management:**
- Named location saving
- Goal-oriented navigation
- Waypoint-based guidance
- Arrival detection

## Configuration System

The system uses a comprehensive YAML configuration file that manages:

**Camera Settings:**
- Resolution (320x320 default for performance)
- Frame rate (20 FPS target)
- Source selection

**Model Configuration:**
- YOLO model path and settings
- Depth estimation model
- LLM models for narrative and vision
- Device selection (MPS, CUDA, CPU)

**Safety Parameters:**
- Minimum safe distance (1.5m)
- Danger and caution zones
- Emergency stop key ('q')

**Performance Tuning:**
- Audio update intervals
- Depth calculation frequency
- Frame processing optimization
- Multi-threading parameters

## Technical Stack

### Primary Dependencies:
- **Python 3.12**: Core language
- **PyTorch**: Deep learning framework with MPS support
- **OpenCV**: Computer vision and image processing
- **NumPy**: Numerical computations
- **Ollama**: Local LLM serving
- **UltraLytics**: YOLO model integration

### Audio Processing:
- **PyAudio**: Audio I/O
- **PyDub**: Audio manipulation
- **SpeechRecognition**: Voice input processing
- **macOS 'say'**: Text-to-speech

### 3D Processing:
- **Open3D**: 3D visualization (optional)
- **PyRoomAcoustics**: Echolocation simulation
- **NumPy**: 3D transformations

### AI Models:
- **YOLOv12**: Real-time object detection
- **Depth Anything V2**: Monocular depth estimation
- **Gemma3:4b**: Navigation narrative generation
- **Moondream**: Vision-language understanding

## Performance Optimization

### Real-time Constraints:
- Target 20 FPS processing on Apple Silicon M-series
- Asynchronous audio processing to prevent blocking
- Depth estimation every 4th frame to reduce computation
- Selective object detection (top 3 most relevant objects)

### Memory Management:
- Efficient data structures for frame processing
- Selective data logging to reduce I/O overhead
- Thread-safe resource management
- Model loading optimization

### Hardware Acceleration:
- MPS (Metal Performance Shaders) for Apple Silicon
- Configurable device selection (CPU fallback)
- Optimized model sizes for real-time performance

## Safety Features

### Proximity Detection:
- Real-time obstacle monitoring
- Categorized distance warnings
- Immediate danger alerts for objects <0.5m
- Visual and audio feedback integration

### Emergency Systems:
- Emergency stop ('q' key)
- Multiple safety thresholds
- Audio priority for danger alerts
- System status monitoring

### Navigation Safety:
- Minimum safe distance enforcement
- Collision prediction
- Social navigation awareness
- Path clearance verification

## Deployment & Setup

### Platform Requirements:
- **macOS** with Apple Silicon (M1/M2/M3) recommended
- **Python 3.12** with virtual environment support
- **16GB RAM** minimum, 32GB recommended
- **USB webcam** or compatible video source

### Setup Process:
1. **Environment Creation**: Virtual environment with Python 3.12
2. **Dependency Installation**: All required packages via pip
3. **System Dependencies**: Homebrew packages for audio processing
4. **Model Downloads**: YOLO, depth estimation, and LLM models
5. **Ollama Setup**: Local LLM serving with required models

### Configuration Tuning:
- Camera-specific calibration
- Performance parameter adjustment
- Safety threshold customization
- Audio system optimization

## Data Flow

The system processes data through the following pipeline:

```
Camera Frame → Detection → Depth Estimation → SLAM Tracking → 
Occupancy Grid Update → Audio Generation → Narrative Generation → 
Visual Annotation → Audio Output → User Feedback
```

Each component operates in a thread-safe manner with appropriate buffering to maintain real-time performance.

## Key Technical Innovations

1. **Bio-mimetic Audio Navigation**: Simulates echolocation for spatial awareness
2. **Multimodal AI Integration**: Combines computer vision with large language models
3. **Real-time SLAM**: Monocular SLAM without external sensors (IMU)
4. **Adaptive Audio Beaconing**: Dynamic audio cues based on environment
5. **Social Navigation AI**: Region-aware crowd navigation with social norms
6. **Efficient 3D Mapping**: Sparse voxel grid for real-time 3D environment modeling

## Future Enhancements Potential

- **IMU Integration**: Enhanced SLAM with inertial measurements
- **Multi-camera Support**: Stereo vision for improved depth estimation
- **Advanced Path Planning**: More sophisticated navigation algorithms
- **Cloud Integration**: Remote assistance and learning features
- **Custom Hardware**: Optimized form factor for daily use

## Conclusion

OrbyGlasses represents a sophisticated integration of computer vision, AI, and audio processing technologies specifically designed for visually impaired navigation. The system's architecture balances real-time performance with rich feature sets, providing a comprehensive navigation solution that adapts to individual user needs and environmental conditions.