# OrbyGlasses

**Bio-Mimetic Navigation Engine for Visually Impaired Users**

OrbyGlasses is an innovative, AI-powered navigation assistance system designed to help visually impaired individuals navigate safely using smart glasses with a webcam. The system runs entirely locally on Apple Silicon (M2 Max) using cutting-edge computer vision, depth estimation, spatial audio, and AI-generated contextual guidance.

---

## Features

### Core Capabilities

- **Real-Time Object Detection**: YOLOv11n optimized for Apple Silicon MPS acceleration.
- **Depth Estimation**: Apple Depth Pro (with MiDaS fallback) for accurate distance measurement.
- **Visual SLAM System**: Tracks user position indoors using only a USB webcam (no GPS or IMU needed).
- **3D Occupancy Grid Mapping**: Real-time volumetric environment representation with probabilistic obstacle detection.
- **Indoor Navigation System**: Enables goal-oriented navigation (e.g., "Take me to the kitchen").
- **Trajectory Prediction**: Uses a simplified Graph Neural Network (GNN) to predict the future positions of moving objects.
- **Social Navigation**: Provides guidance in crowded areas based on social norms and conventions.
- **Bio-Mimetic Echolocation**: Spatial audio cues simulating bat echolocation using binaural sound.
- **AI-Powered Narratives**: Contextual navigation guidance using Ollama (Gemma 3 + Moondream vision models).
- **Predictive Navigation**: Reinforcement learning (PPO) to learn user patterns and predict optimal paths.
- **Text-to-Speech**: Real-time audio feedback for obstacle alerts and navigation guidance.
- **Non-blocking Voice Input**: Voice commands work without impacting camera feed performance.
- **Privacy-First**: 100% local processing, no cloud dependencies.

### Features in Depth

#### Visual SLAM and Indoor Navigation

The latest version of OrbyGlasses introduces a powerful Visual SLAM (Simultaneous Localization and Mapping) system, which, combined with an Indoor Navigation module, transforms the device from a reactive obstacle avoidance tool into a proactive navigation assistant.

-   **Visual SLAM (`src/slam.py`)**:
    -   Tracks the user's position in 3D space in real-time using a standard webcam.
    -   Builds and saves maps of indoor environments for persistent navigation.
    -   Achieves real-time performance (10-50 FPS depending on the environment) on Apple Silicon.

-   **Indoor Navigation (`src/indoor_navigation.py`)**:
    -   Utilizes A* pathfinding on a 2D occupancy grid to plan routes around obstacles.
    -   Allows users to save and navigate to named locations (e.g., "kitchen," "desk").
    -   Provides turn-by-turn voice guidance to the destination.

This combination allows for true indoor navigation, enabling users to move from point A to point B with confidence, rather than just reacting to their immediate surroundings.

#### 3D Occupancy Grid Mapping

OrbyGlasses now includes a sophisticated 3D occupancy grid system that creates a volumetric representation of the environment in real-time.

-   **Sparse Voxel Storage (`src/occupancy_grid_3d.py`)**:
    -   Efficiently stores only observed voxels using a probabilistic approach.
    -   Memory-efficient representation (~1 MB for typical indoor rooms).
    -   Uses Bayesian log-odds updates for robust obstacle detection.

-   **Ray Casting & Sensor Fusion**:
    -   Combines depth estimation with SLAM camera poses for accurate 3D mapping.
    -   Marks space along sensor rays as free, endpoints as occupied.
    -   Handles sensor noise and uncertainty through probabilistic updates.

-   **Real-Time Visualization**:
    -   Displays 2D slices at head height for navigation awareness.
    -   Color-coded occupancy: Blue (free), Red (occupied), Gray (unknown).
    -   Updates at 2 Hz with minimal performance impact (<5% FPS reduction).

-   **Path Planning Integration**:
    -   Provides collision-free path planning information.
    -   Enables spatial memory and environment understanding.
    -   Supports future features like map persistence and revisit locations.

See `docs/OCCUPANCY_GRID_3D.md` for detailed documentation, API reference, and usage examples.

#### Trajectory Prediction

OrbyGlasses can predict the future movement of objects in the user's environment. This is a crucial feature for proactive navigation, especially in dynamic environments with moving people or vehicles.

-   **Object Tracking (`src/trajectory_prediction.py`)**:
    -   Tracks objects across frames to build a history of their movement.
    -   Calculates a velocity for each tracked object.

-   **Social Force Model & GNN (`src/trajectory_prediction.py`)**:
    -   A simplified Graph Neural Network (GNN) and a Social Force Model are used to predict the future path of tracked objects.
    -   The system can anticipate if an object is on a collision course with the user and provide an early warning.

#### Social Navigation

Navigating crowded spaces is a major challenge for visually impaired individuals. The social navigation feature provides guidance that is aware of social norms and conventions.

-   **Social Norms (`src/social_navigation.py`)**:
    -   The system can be configured for different regions (e.g., "stay to the right" in the US, "stay to the left" in the UK).
    -   It analyzes the density of a crowd and identifies social gaps for safe passage.
-   **Contextual Guidance**:
    -   Provides advice like "Gap opening in crowd ahead on your right. Safe to proceed to the right."

---

## System Requirements

### Hardware
- **Computer**: MacBook with Apple Silicon (M2 Max recommended)
- **Camera**: Built-in webcam or IP camera (e.g., smart glasses with WiFi streaming)
- **Audio**: Speakers or headphones for spatial audio output

### Software
- **OS**: macOS 13.0+ (Ventura or later)
- **Python**: 3.12
- **Homebrew**: For system dependencies
- **Ollama**: For LLM inference
- **Dependencies**: See `requirements.txt` for a full list of Python packages.

---

## Quick Start

### 1. Clone the Repository

```bash
cd ~/Desktop
git clone <your-repo-url> OrbyGlasses
cd OrbyGlasses
```

### 2. Run Setup Script

The setup script automates the entire installation process:

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies from `requirements.txt`
- Install Homebrew packages (portaudio, ffmpeg)
- Install and start Ollama
- Download AI models (Gemma 3, Moondream, YOLOv11n)
- Set up project directories

### 3. Activate Environment

```bash
source venv/bin/activate
```

### 4. Run OrbyGlasses

```bash
python src/main.py
```

Press `q` to stop the system.

---

## Configuration

Edit `config/config.yaml` to customize behavior:

```yaml
camera:
  source: 0  # 0 for built-in webcam, or IP camera URL
  width: 640
  height: 480

models:
  yolo:
    confidence: 0.5  # Detection confidence threshold
  llm:
    primary: "gemma2:2b"  # Primary narrative model
    vision: "moondream"   # Vision understanding model

safety:
  min_safe_distance: 1.5  # Meters

slam:
  enabled: true
  visualize: true

trajectory_prediction:
  enabled: true
  visualize: true

social_navigation:
  region: "us" # "us", "uk", "japan"
```

---

## Architecture

### Pipeline Flow

```
Camera Feed
    ↓
Object Detection (YOLOv11)
    ↓
Depth Estimation (Depth Pro/MiDaS)
    ↓
SLAM (ORB-SLAM)
    ↓
Trajectory Prediction (GNN)
    ↓
Navigation Summary
    ↓
┌─────────────┬──────────────┬──────────────┬──────────────────┬──────────────────┐
│             │              │              │                  │                  │
Echolocation  AI Narrative   RL Prediction  Indoor Navigation  Social Navigation
(Spatial      (Gemma +       (PPO)          (A* Pathfinding)   (Social Norms)
 Audio)        Moondream)
│             │              │                  │                  │
└─────────────┴──────────────┴──────────────┴──────────────────┴──────────────────┘
    ↓
Audio Output (TTS + Beeps)
```

### Module Descriptions

- **`main.py`**: Main application entry point.
- **`detection.py`**: YOLO object detection and depth estimation.
- **`echolocation.py`**: Spatial audio generation.
- **`narrative.py`**: AI narrative generation with Ollama.
- **`prediction.py`**: Reinforcement learning path prediction.
- **`slam.py`**: Visual SLAM for localization and mapping.
- **`indoor_navigation.py`**: Goal-oriented navigation with A* pathfinding.
- **`trajectory_prediction.py`**: Predicts the future movement of objects.
- **`social_navigation.py`**: Provides guidance in crowded areas.
- **`utils.py`**: Configuration, logging, audio management.

---

## Performance

- **Without SLAM**: ~50ms per frame (~20 FPS)
- **With SLAM**: ~70-100ms per frame (~10-14 FPS)

SLAM introduces a performance overhead, but enables true indoor navigation. It can be disabled in the `config.yaml` for scenarios where maximum FPS is critical. Trajectory prediction has a negligible performance impact (~0.1ms per frame).

---

## Project Structure

```
OrbyGlasses/
├── src/
│   ├── main.py
│   ├── detection.py
│   ├── echolocation.py
│   ├── narrative.py
│   ├── prediction.py
│   ├── slam.py
│   ├── indoor_navigation.py
│   ├── trajectory_prediction.py
│   ├── social_navigation.py
│   └── utils.py
├── models/
│   ├── yolo/
│   ├── depth/
│   └── rl/
├── data/
│   ├── logs/
│   └── maps/
├── config/
│   └── config.yaml
├── tests/
│   ├── test_detection.py
│   ├── test_echolocation.py
│   ├── test_slam.py
│   ├── test_trajectory_prediction.py
│   └── test_utils.py
├── docs/
│   ├── ...
├── requirements.txt
├── setup.sh
└── README.md
```

---

## Testing

The project includes a comprehensive suite of unit tests to ensure the reliability of each component.

-   **`test_detection.py`**: Tests for the object detection and depth estimation pipeline.
-   **`test_echolocation.py`**: Tests for the spatial audio generation.
-   **`test_slam.py`**: Tests for the SLAM and indoor navigation system.
-   **`test_trajectory_prediction.py`**: Tests for the trajectory prediction system.
-   **`test_utils.py`**: Tests for the utility functions.

Run all unit tests:

```bash
pytest tests/ -v
```

Run a specific test file:

```bash
pytest tests/test_slam.py -v
```

---

## User Study

A comprehensive user study protocol has been designed to validate the effectiveness of OrbyGlasses with real users. The study aims to measure the impact of the device on navigation safety, speed, and user confidence. For more details, see the [User Study Guide](docs/USER_STUDY_GUIDE.md).

---

## Ethical Considerations

This project has a direct impact on the lives of visually impaired individuals. The following ethical considerations have been taken into account:

-   **Privacy**: All processing is done locally on the user's device. No data is sent to the cloud.
-   **Safety**: The system is designed to be a navigation aid and not a replacement for traditional mobility tools like a white cane. The user is always in control.
-   **Reliability**: The system is designed to be robust and fail gracefully. In case of a failure, the user is notified with an audio cue.
-   **User-Centered Design**: The development process is guided by feedback from visually impaired users.

---

## Roadmap

- [x] Voice command interface (implemented with non-blocking recognition)
- [x] Visual SLAM and Indoor Navigation
- [x] Trajectory Prediction
- [x] Social Navigation
- [ ] Integration with actual smart glasses (e.g., Vuzix, Xreal)
- [ ] Multi-user federated learning with Flower
- [ ] GPS integration for outdoor navigation
- [ ] Obstacle avoidance path planning
- [ ] Mobile app companion
- [ ] Cloud-optional model updates

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

## License

MIT License - See LICENSE file for details.
