# OrbyGlasses

AI-powered navigation assistant for blind and visually impaired users.

## What It Does

OrbyGlasses uses a camera to detect objects, estimate distances, and provide audio guidance for safe navigation. It works entirely on your computer without sending data to the cloud.

**Key Features:**
- Real-time object detection
- Distance estimation
- Audio directions and warnings
- Indoor mapping and navigation
- Works without GPS

## Quick Start

### Choose Your Mode:

**VO Mode (Trajectory Only):**
```bash
./run_vo_mode.sh
```

**Full SLAM Mode (With Mapping):**
```bash
./run_slam_mode.sh
```

Or just: `./run_orby.sh` (uses current config)

Press `q` to stop.

See `MODE_GUIDE.md` for details.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4/M5)
- Python 3.10, 3.11, or 3.12
- Built-in camera or webcam
- Speakers or headphones

## SLAM Systems

OrbyGlasses includes multiple SLAM options optimized for different needs:

### ✅ pySLAM (RECOMMENDED - Default)
- **Professional-grade** monocular SLAM
- **Real-time 3D visualization** with map viewer
- **Advanced feature tracking** with ORB/SIFT/SuperPoint
- **Loop closure detection** and bundle adjustment
- **Already enabled** in your config

### 🗺️ RTAB-Map (Optional - Appearance-Based Mapping)
- **Real-time appearance-based mapping** with loop closure detection
- **Graph-based SLAM optimization** for robust mapping
- **Multi-session mapping** support for long-term operation
- **Memory management** for large-scale environments
- **Excellent for long-term localization** and re-localization
- **Documentation**: [https://introlab.github.io/rtabmap/](https://introlab.github.io/rtabmap/)

### 🤖 DROID-SLAM (Optional - Deep Learning)
- **State-of-the-art** deep learning SLAM
- **Excellent accuracy** and robustness
- **Works with Apple Silicon** (PyTorch MPS)
- **Requires installation** (see below)

### 🏆 ORB-SLAM3 (Optional - Industry Standard)
- **Most accurate** traditional SLAM
- **Requires building from source** on macOS
- **Best for research** and maximum accuracy
- **Complex installation** process

### Quick Setup
```bash
# pySLAM is already ready to use!
./run.sh

# To install RTAB-Map (optional):
./install_rtabmap.sh

# To install DROID-SLAM (optional):
./install_slam.sh
```

### Configuration
Edit `config/config.yaml` to choose your SLAM system:
```yaml
slam:
  use_pyslam: true     # pySLAM with 3D visualization (recommended)
  use_rtabmap: false    # Appearance-based mapping
  use_droid: false      # Deep learning SLAM
  use_orbslam3: false   # Industry standard
```

## How It Works

1. **Camera** captures video
2. **Object Detection** identifies objects (YOLOv11)
3. **Depth Estimation** measures distances (Depth Anything V2)
4. **SLAM** tracks your position indoors
5. **Audio Guidance** speaks directions and warnings

## Running Modes

**Standard Mode** (15-25 FPS):
```bash
./run.sh
```
Full features with detailed navigation.

**Fast Mode** (20-30 FPS):
```bash
./run.sh --fast
```
Optimized for speed with core features only.

**With Navigation Panel** (overhead compass view):
```bash
./run.sh --nav              # standard mode with nav panel
./run.sh --fast --nav       # fast mode with nav panel
```
Shows overhead map, compass, and real-time position tracking.

## Audio Feedback

The system provides clear spoken directions:
- **Danger** (<1m): "Stop. Car ahead. Go left."
- **Caution** (1-2.5m): "Chair on your right. Two meters away."
- **Safe** (>2.5m): "Path is clear."

## Configuration

Edit `config/config.yaml` or `config/config_fast.yaml`:

```yaml
camera:
  source: 0        # 0 = built-in camera
  width: 320
  height: 240

models:
  yolo:
    confidence: 0.55   # Higher = fewer false positives

safety:
  danger_distance: 1.0      # meters
  min_safe_distance: 1.5    # meters
```

## Project Structure

```
OrbyGlasses/
├── src/
│   ├── main.py                    # Main program
│   ├── core/                      # Core modules
│   │   ├── detection.py           # Object detection
│   │   ├── depth_anything_v2.py   # Depth estimation
│   │   ├── utils.py               # Configuration and audio
│   │   └── safety_system.py       # Danger detection
│   ├── navigation/                # Navigation modules
│   │   ├── slam_system.py         # Indoor mapping
│   │   └── indoor_navigation.py   # Path planning
│   └── visualization/             # Display modules
├── config/
│   ├── config.yaml                # Standard settings
│   └── config_fast.yaml           # Fast mode settings
├── tests/                         # Test suite
├── models/                        # AI models
└── data/                          # Logs and maps
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run a specific test:

```bash
pytest tests/test_detection.py -v
```

## Features in Detail

### Object Detection
Uses YOLOv11n to identify objects like cars, people, chairs, and doors.

### Depth Estimation
Calculates distances to objects using Depth Anything V2 Small model.

### Visual SLAM
Tracks your position and builds a map of indoor spaces without GPS.

### Indoor Navigation
Plans paths to saved locations (e.g., "kitchen", "desk") and provides turn-by-turn directions.

### Safety System
Monitors your surroundings and issues warnings at three levels:
- Danger zone (<1m): Immediate audio alert
- Caution zone (1-2.5m): Regular updates
- Safe zone (>2.5m): Quiet operation

## Privacy

All processing happens locally on your computer. No data is sent to the internet.

## Safety Note

OrbyGlasses is a navigation aid, not a replacement for traditional tools like white canes or guide dogs. Always use it alongside your existing mobility aids.

## Technical Details

**AI Models:**
- YOLOv11n for object detection
- Depth Anything V2 Small for depth estimation
- Moondream for scene understanding
- Gemma 3 4B for language processing

**Performance:**
- Standard mode: 15-25 FPS
- Fast mode: 20-30 FPS
- Depth processing: 70-200ms per frame
- Audio latency: 0.8-2s

**Hardware Acceleration:**
Uses Apple Metal Performance Shaders (MPS) for fast AI inference on M-series chips.

## Troubleshooting

**Camera not working:**
```bash
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"
```

**Ollama not running:**
```bash
ollama serve
```

**Poor performance:**
Use fast mode or reduce camera resolution in the config file.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## Production Status

**Current Status:** Beta (v0.9.0)

OrbyGlasses is under active development. While core features are functional, this software is:
- ✅ Suitable for testing and development
- ⚠️ **Not yet recommended for independent use by blind users without supervision**
- 🚧 Requires further real-world validation and safety testing

**Safety Notice:**
OrbyGlasses is a navigation aid, not a replacement for traditional mobility tools like white canes or guide dogs. Always use it alongside your existing mobility aids and with appropriate caution.

## Installation

See [SETUP.md](SETUP.md) for complete installation instructions.

**Quick Install:**
```bash
git clone https://github.com/yourusername/OrbyGlasses.git
cd OrbyGlasses
./install_pyslam.sh
pip install -r requirements.txt
./run_orby.sh
```

## Contributing

We welcome contributions! Please read:
- [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) - Community guidelines
- [ARCHITECTURE.md](ARCHITECTURE.md) - System design

## Support & Documentation

- **Setup Guide:** [SETUP.md](SETUP.md)
- **Troubleshooting:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Issues:** [GitHub Issues](https://github.com/yourusername/OrbyGlasses/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/OrbyGlasses/discussions)

## License

GNU General Public License v3.0 or later - see [LICENSE](LICENSE) file for details.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

## Credits

Built with:
- Ultralytics YOLOv11 for object detection
- Depth-Anything-V2 for depth estimation
- Ollama for local AI inference
- OpenCV for computer vision
- PyTorch for deep learning

---

For detailed setup instructions, see the comments in `setup.sh` and `run.sh`.
