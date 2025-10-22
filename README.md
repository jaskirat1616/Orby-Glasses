# OrbyGlasses

**Bio-Mimetic Navigation Engine for Visually Impaired Users**

OrbyGlasses is an AI-powered navigation assistance system designed to empower visually impaired individuals to navigate safely using smart glasses with a webcam. Running entirely locally on Apple Silicon (M2 Max recommended), it leverages advanced computer vision, depth estimation, spatial audio, and AI-driven contextual guidance for a seamless and private experience. A lightweight Simplified Version is available for users prioritizing speed and ease of setup, omitting advanced features like SLAM and 3D mapping.

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

### Core Capabilities (Full Version)

*   **Real-Time Object Detection:** Uses `YOLOv11n` optimized for Apple Silicon MPS acceleration to identify obstacles instantly.
*   **Depth Estimation:** Employs Apple `Depth Pro` (with `MiDaS` fallback) for precise distance measurements.
*   **Visual SLAM System:** Tracks user position indoors using a USB webcam, eliminating the need for GPS or IMU sensors.
*   **3D Occupancy Grid Mapping:** Creates real-time volumetric environment maps with probabilistic obstacle detection.
*   **Indoor Navigation System:** Supports goal-oriented navigation to specific locations (e.g., "kitchen").
*   **Trajectory Prediction:** Applies a simplified Graph Neural Network (GNN) to forecast moving object paths.
*   **Social Navigation:** Guides users through crowded areas while respecting regional social norms.
*   **Bio-Mimetic Echolocation:** Mimics bat echolocation with spatial audio cues via binaural sound.
*   **AI-Powered Narratives:** Generates contextual guidance using `Ollama` with `Gemma2` (2B parameter model) and `Moondream` vision models.
*   **Predictive Navigation:** Uses reinforcement learning (PPO) to learn user patterns and optimize paths.
*   **Text-to-Speech:** Delivers real-time audio feedback for obstacle alerts and navigation instructions.
*   **Non-Blocking Voice Input:** Supports voice commands without interrupting camera feed processing.
*   **Privacy-First Design:** Ensures 100% local processing with no cloud dependencies.

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
