# OrbyGlasses 🚀

**2024-2025 State-of-the-Art Navigation System for Blind and Visually Impaired Users**

OrbyGlasses is a breakthrough AI-powered navigation assistant that helps blind and visually impaired users navigate safely and independently. Built using the latest research in computer vision, robotics navigation, and assistive technology (2024-2025).

## 🎯 What Makes It Special (2024-2025 Optimized)

- **🔬 Apple Depth Pro (2024)**: Sharpest depth maps available - 2.25MP in <0.3s
- **⚡ YOLOv11n (2024)**: Latest object detection - 22% fewer params, 2% faster
- **📊 Enhanced Visualizations**: Depth zones, safety arrows, clear distance labels
- **🎯 Production Ready**: 15-25 FPS on Apple Silicon, robust error handling
- **♿ User-Centered Design**: Researched-backed features for blind navigation
- **🔊 Smart Audio**: Priority-based alerts - "Stop. Car ahead. Go left"
- **🗺️ Visual SLAM**: Indoor navigation without GPS
- **✅ Not Over-Engineered**: Essential features only, disabled unnecessary components

## ⚡ Quick Start

```bash
./run.sh
```

See `QUICK_START.md` for 10-second setup.

---

## Table of Contents

*   [Features](#features)
    *   [Core Capabilities (Full Version)](#core-capabilities-full-version)
    *   [Simplified Version](#simplified-version)
*   [Requirements](#requirements)
*   [Installation](#installation)
*   [Usage](#usage)
*   [Configuration](#configuration)
*   [Key Differences from Full Version](#key-differences-from-full-version)
*   [Features in Depth (Full Version)](#features-in-depth-full-version)
    *   [Visual SLAM and Indoor Navigation](#visual-slam-and-indoor-navigation)
    *   [3D Occupancy Grid Mapping](#3d-occupancy-grid-mapping)
    *   [Trajectory Prediction](#trajectory-prediction)
    *   [Social Navigation](#social-navigation)
*   [System Requirements (Full Version)](#system-requirements-full-version)
    *   [Hardware](#hardware)
    *   [Software](#software)
*   [Quick Start (Full Version)](#quick-start-full-version)
    *   [1. Clone the Repository](#1-clone-the-repository)
    *   [2. Run Setup Script](#2-run-setup-script)
    *   [3. Activate Environment](#3-activate-environment)
    *   [4. Run OrbyGlasses](#4-run-orbyglasses)
*   [Configuration (Full Version)](#configuration-full-version)
*   [Architecture (Full Version)](#architecture-full-version)
    *   [Pipeline Flow](#pipeline-flow)
    *   [Module Descriptions](#module-descriptions)
*   [Performance (Full Version)](#performance-full-version)
*   [Project Structure](#project-structure)
*   [Testing](#testing)
*   [User Study](#user-study)
*   [Ethical Considerations](#ethical-considerations)
*   [Roadmap](#roadmap)
*   [Contributing](#contributing)
*   [License](#license)

---

## Features

### Core Capabilities (2024-2025 Optimized)

#### Computer Vision (State-of-the-Art 2024-2025)

*   **Object Detection - YOLOv11n (Ultralytics 2024)**
    - Latest real-time detection model
    - 22% fewer parameters than YOLOv10
    - 2% faster inference
    - Optimized for Apple Silicon MPS

*   **Depth Estimation - Apple Depth Pro (October 2024)**
    - **Sharpest depth maps available** - 2.25 megapixels
    - Metric depth in <0.3 seconds on standard GPU
    - Zero-shot generalization without camera intrinsics
    - Research: Apple ML Research "Depth Pro: Sharp Monocular Metric Depth"
    - Fallback: Depth Anything V2 (CVPR 2024)

#### Navigation & Robotics (Best Practices 2024)

*   **Visual SLAM System (Camera-Only)**
    - Tracks user position indoors without GPS/IMU
    - ORB features with temporal consistency
    - Pose smoothing and motion models
    - Based on: ACM Computing Surveys (2024) autonomous navigation review

*   **Indoor Navigation with A* Path Planning**
    - Goal-oriented navigation to named locations
    - Dynamic obstacle avoidance
    - Safe path suggestions with directional arrows
    - Research: Nature Scientific Reports (2024) IA-DWA algorithm

#### Enhanced Visualizations (User-Centered Design 2024)

*   **Depth Zone Overlay**
    - Semi-transparent colored zones: Red (danger <1.5m), Yellow (caution 1.5-3.5m), Green (safe >3.5m)
    - LANCZOS4 interpolation for sharp rendering
    - Perceptually uniform color progression

*   **Safety Direction Arrows**
    - Large, clear arrows when danger detected
    - "GO LEFT" / "GO RIGHT" text instructions
    - Based on: MDPI Sensors (2024) assistive systems research

*   **Improved Labels**
    - Large text with black backgrounds for readability
    - Distance in meters with object names
    - Color-coded by safety level

#### Audio Guidance (Assistive Technology Best Practices 2024)

*   **Priority-Based Audio Alerts**
    - Immediate danger warnings (<1m): 0.5s interval
    - Normal announcements: 2.0s interval
    - Relatable distance terms: "arm's length away", "one step away"
    - Research: ScienceDirect (2024) assistive systems survey

*   **Clear, Simple Directions**
    - "Stop. Car ahead. Go left" - actionable guidance
    - Text-to-speech with optimized rate (180 WPM)
    - Non-blocking voice input (background thread)

#### Additional Features

*   **Smart Caching**: Motion-based depth recomputation
*   **Object Tracking**: Temporal consistency for smooth depth
*   **Safety System**: Multi-level danger zones with calibration
*   **Privacy-First**: 100% local processing, no cloud

### Simplified Version

For users seeking a lightweight alternative, the Simplified Version of OrbyGlasses prioritizes performance and ease of setup by omitting complex features like SLAM and 3D mapping.

**Features:**

*   Real-time object detection using `YOLOv11`.
*   Depth estimation with `Depth Anything V2` for accurate distance measurements.
*   Audio feedback via simple text-to-speech for navigation guidance.
*   Danger alerts for objects within `0.4m`; caution alerts for objects within `1.5m`.
*   No SLAM, 3D mapping, or occupancy grids for reduced computational load.
*   Faster startup and response times.

---

## Requirements

*   **OS:** macOS (for text-to-speech) or Linux/Windows with `pyttsx3`.
*   **Python:** 3.8+.
*   **Camera:** Built-in webcam or IP camera.
*   **Ollama:** For local LLM processing.

---

## Installation

1.  Run the setup script:
    ```bash
    ./setup_simple.sh
    ```
2.  Ensure Ollama is running:
    ```bash
    ollama serve
    ```
3.  Start the application:
    ```bash
    python simple_orbyglasses.py
    ```

---

## Usage

*   Processes video from the default camera.
*   Provides audio guidance through system speakers.
*   Displays objects with bounding boxes.
*   Issues danger/caution alerts based on object proximity.
*   Press `q` to quit.

---

## Configuration

Hardcoded settings in `simple_orbyglasses.py` (e.g., detection thresholds, audio intervals, safety distances, camera settings).

---

## Key Differences from Full Version

*   Excludes SLAM, 3D mapping, and occupancy grids.
*   Uses `Depth Anything V2` instead of size-based estimation.
*   Simplified audio system (macOS `say` command or `pyttsx3`).
*   Depth estimation runs every Nth frame to reduce computational load.

---

## Features in Depth (Full Version)

### Visual SLAM and Indoor Navigation

The Visual SLAM (Simultaneous Localization and Mapping) system, paired with the Indoor Navigation module, transforms OrbyGlasses into a proactive navigation assistant.

**Visual SLAM (`src/slam.py`):**

*   Tracks user position in 3D space using a standard webcam.
*   Builds and saves indoor environment maps for persistent navigation.
*   Achieves 10-50 FPS on Apple Silicon, depending on the environment.

**Indoor Navigation (`src/indoor_navigation.py`):**

*   Uses A* pathfinding on a 2D occupancy grid to plan obstacle-free routes.
*   Supports navigation to saved locations (e.g., "kitchen," "desk").
*   Provides turn-by-turn voice guidance.

### 3D Occupancy Grid Mapping

The system constructs a real-time 3D volumetric map to enhance spatial awareness.

**Sparse Voxel Storage (`src/occupancy_grid_3d.py`):**

*   Stores observed voxels efficiently (~1 MB for typical rooms).
*   Uses Bayesian log-odds updates for robust obstacle detection.

**Ray Casting & Sensor Fusion:**

*   Combines depth estimation with SLAM camera poses for accurate 3D mapping.
*   Marks free space along sensor rays and occupied space at endpoints.
*   Handles sensor noise via probabilistic updates.

**Real-Time Visualization:**

*   Displays 2D slices at head height (Blue: free, Red: occupied, Gray: unknown).
*   Updates at 2 Hz with minimal performance impact (<5% FPS reduction).

**Path Planning Integration:**

*   Supports collision-free path planning and spatial memory.
*   Enables map persistence and location revisits.

Refer to `docs/OCCUPANCY_GRID_3D.md` for detailed documentation.

### Trajectory Prediction

Predicts object movements for proactive alerts in dynamic environments.

**Object Tracking (`src/trajectory_prediction.py`):**

*   Tracks objects across frames to build movement histories.
*   Calculates object velocities.

**Social Force Model & GNN (`src/trajectory_prediction.py`):**

*   Uses a simplified GNN and Social Force Model to forecast paths.
*   Issues early warnings for potential collisions.

### Social Navigation

Guides users through crowded spaces with culturally aware instructions.

**Social Norms (`src/social_navigation.py`):**

*   Configurable for regional conventions (e.g., "stay to the right" in the US, "stay to the left" in the UK).
*   Identifies safe passage gaps by analyzing crowd density.

**Contextual Guidance:**

*   Provides instructions like "Gap opening on your right. Safe to proceed."

---

## System Requirements (Full Version)

### Hardware

*   **Computer:** MacBook with Apple Silicon (M2 Max recommended).
*   **Camera:** Built-in webcam or IP camera (e.g., smart glasses with WiFi streaming).
*   **Audio:** Speakers or headphones for spatial audio.

### Software

*   **OS:** macOS 13.0+ (Ventura or later).
*   **Python:** 3.12.
*   **Homebrew:** For system dependencies.
*   **Ollama:** For local LLM inference.
*   **Dependencies:** See `requirements.txt`.

---

## Quick Start (Full Version)

### 1. Clone the Repository

```bash
cd ~/Desktop
git clone <your-repo-url> OrbyGlasses
cd OrbyGlasses
```

### 2. Run Setup Script

```bash
chmod +x setup.sh
./setup.sh
```

This script:

*   Creates a Python virtual environment.
*   Installs dependencies from `requirements.txt`.
*   Installs Homebrew packages (`portaudio`, `ffmpeg`).
*   Sets up Ollama and downloads AI models (`Gemma2`, `Moondream`, `YOLOv11n`).
*   Configures project directories.

### 3. Activate Environment

```bash
source venv/bin/activate
```

### 4. Run OrbyGlasses

```bash
python src/main.py
```

Press `q` to stop.

---

## Configuration (Full Version)

Edit `config/config.yaml` to customize settings:

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

## Architecture (Full Version)

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
│ Echolocation│ AI Narrative │ RL Prediction│ Indoor Navigation│ Social Navigation│
│ (Spatial    │ (Gemma2 +    │ (PPO)        │ (A* Pathfinding) │ (Social Norms)   │
│  Audio)     │ Moondream)   │              │                  │                  │
└─────────────┴──────────────┴──────────────┴──────────────────┴──────────────────┘
    ↓
Audio Output (TTS + Beeps)
```

### Module Descriptions

*   `main.py`: Application entry point.
*   `detection.py`: YOLO-based object detection and depth estimation.
*   `echolocation.py`: Spatial audio generation.
*   `narrative.py`: AI narrative generation via Ollama.
*   `prediction.py`: Reinforcement learning for path prediction.
*   `slam.py`: Visual SLAM for localization and mapping.
*   `indoor_navigation.py`: A* pathfinding for goal-oriented navigation.
*   `trajectory_prediction.py`: Object movement forecasting.
*   `social_navigation.py`: Crowd navigation with social norms.
*   `utils.py`: Configuration, logging, and audio utilities.

---

## Performance (Full Version)

*   **Without SLAM:** 50ms per frame (20 FPS).
*   **With SLAM:** 70-100ms per frame (10-14 FPS).

SLAM enables advanced navigation but adds overhead. Disable it in `config.yaml` for higher FPS. Trajectory prediction has minimal impact (~0.1ms per frame).

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
│   ├── OCCUPANCY_GRID_3D.md
│   ├── USER_STUDY_GUIDE.md
├── requirements.txt
├── setup.sh
├── setup_simple.sh
├── simple_orbyglasses.py
└── README.md
```

---

## Testing

A test suite ensures reliability:

*   `test_detection.py`: Validates object detection and depth estimation.
*   `test_echolocation.py`: Verifies spatial audio.
*   `test_slam.py`: Tests SLAM and navigation.
*   `test_trajectory_prediction.py`: Checks trajectory prediction.
*   `test_utils.py`: Tests utility functions.

Run all tests:

```bash
pytest tests/ -v
```

Run a specific test:

```bash
pytest tests/test_slam.py -v
```

---

## User Study

A user study protocol evaluates navigation safety, speed, and confidence. See `User Study Guide`.

---

## Ethical Considerations

OrbyGlasses prioritizes user well-being:

*   **Privacy:** Fully local processing; no cloud data transmission.
*   **Safety:** Acts as a navigation aid, not a replacement for tools like white canes.
*   **Reliability:** Graceful failure with audio alerts for issues.
*   **User-Centered Design:** Developed with feedback from visually impaired users.

---

## Roadmap

*   Voice command interface (non-blocking).
*   Visual SLAM and Indoor Navigation.
*   Trajectory Prediction.
*   Social Navigation.
*   Smart glasses integration (e.g., Vuzix, Xreal).
*   Multi-user federated learning with Flower.
*   GPS for outdoor navigation.
*   Obstacle avoidance path planning.
*   Mobile app companion.
*   Optional cloud-based model updates.

---

## Contributing

Contributions are welcome! To contribute:

1.  Fork the repository.
2.  Create a feature branch.
3.  Add tests for new functionality.
4.  Ensure all tests pass.
5.  Submit a pull request.

---

## License

MIT License - See `LICENSE` file for details.
