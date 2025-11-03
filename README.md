# OrbyGlasses

**AI-Powered Navigation Assistant for Blind and Visually Impaired Users**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform: macOS](https://img.shields.io/badge/platform-macOS-lightgrey.svg)](https://www.apple.com/macos/)

OrbyGlasses combines state-of-the-art computer vision with real-time audio guidance to help blind and visually impaired people navigate safely and independently.

## Graphics Video

[![OrbyGlasses Graphics](test_videos/main.MP4)](test_videos/main.MP4)

**Graphics Video:** See `test_videos/main.MP4` for a visual overview of OrbyGlasses features and capabilities.

## Features

### Core Navigation
- **🎯 Real-time Object Detection** - Detects 80 object types (people, cars, furniture, obstacles) using YOLOv11n
- **📏 Distance Measurement** - Accurate depth estimation (0-10m) using Depth Anything V2
- **🗺️ Indoor Position Tracking** - SLAM-based localization tracks your position indoors
- **🔊 Audio Guidance** - Clear, prioritized voice warnings through text-to-speech

### Safety Features
- **⚠️ Stair & Curb Detection** - Prevents falls by detecting vertical drops >15cm (NEW!)
- **🚨 Emergency Stop** - Instant stop with spacebar or 'q' key
- **📊 Distance-Based Warnings** - Danger (<1m), Caution (1-2.5m), Safe (>2.5m)
- **🔍 Uncertainty Handling** - Warns when distance cannot be measured

### Advanced Features
- **🗣️ Voice Control** - Hands-free operation with "Hey Orby" wake word (NEW!)
- **📍 Location Saving** - Save and navigate to named locations ("Kitchen", "Bedroom")
- **🧭 Turn-by-Turn Navigation** - A* path planning to saved waypoints
- **💾 Map Persistence** - Save/load indoor maps between sessions (NEW!)
- **🎵 Spatial Audio Beaconing** - Optional echolocation-style audio cues

## Quick Start

### Production Mode (Recommended for Blind Users)

```bash
# Use production config with all critical features enabled
./run_orby.sh --config config/config_production.yaml
```

**Production config includes:**
- ✅ Depth estimation (distance measurement)
- ✅ Stair/curb detection (safety)
- ✅ SLAM (position tracking)
- ✅ Voice control (hands-free operation)
- ✅ Indoor navigation (path planning)

### Development Mode (Fast, Features Disabled)

```bash
# Development config (many features disabled for speed)
./run_orby.sh
```

Press **SPACEBAR** or **Q** to stop.

### Testing with Videos

You can test OrbyGlasses with sample videos:

```bash
# Run with a test video
./run_orby.sh --video test_videos/your_test_video.mp4 --config config/config_production.yaml
```

**Test Videos Location:** `test_videos/`
- **Graphics Video:** `test_videos/main.MP4` - Visual overview of OrbyGlasses features
- **Test Videos:** `.mov` files available for testing object detection, depth estimation, and SLAM
- Supported formats: `.mp4`, `.MP4`, `.mov`, `.avi`
- Recommended: 640x480 or 1280x720 resolution, 30fps
- Example: `./run_orby.sh --video test_videos/Screen\ Recording\ 2025-11-03\ at\ 1.01.11\ PM.mov`

### Documentation Images

**Images Location:** `images/`
- Screenshots and diagrams for documentation
- UI screenshots, feature visualizations, system architecture diagrams
- Add images here and reference them in documentation using: `![Description](images/your_image.png)`

## Requirements

### Hardware
- **Mac with Apple Silicon** (M1/M2/M3/M4) or Intel Mac
- **Camera** (built-in or USB webcam)
- **Headphones/Speakers** for audio guidance

### Software
- **macOS** 11.0 or later
- **Python** 3.10-3.12
- **8GB RAM** minimum (16GB recommended)

## Installation

### Option 1: Quick Install (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/OrbyGlasses.git
cd OrbyGlasses

# Install dependencies
pip install -r requirements.txt

# Install pySLAM for indoor tracking (optional but recommended)
./install_pyslam.sh

# Run OrbyGlasses
./run_orby.sh
```

### Option 2: Development Install
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest
```

See [SETUP.md](SETUP.md) for detailed installation instructions.

## How It Works

```
Camera (640x480 @ 30fps)
    ↓
┌─────────────────────────────────────┐
│   Object Detection (YOLOv11n)       │ → 80 COCO classes @ 55% confidence
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Depth Estimation (Depth Anything) │ → Monocular depth 0-10m
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Stair Detection (NEW!)            │ → Prevents falls, detects >15cm drops
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   SLAM (pySLAM)                     │ → Indoor position tracking
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│   Audio Guidance (macOS TTS)        │ → Clear voice warnings
└─────────────────────────────────────┘
```

**Visual Examples:**
- See `images/` folder for screenshots, architecture diagrams, and feature visualizations
- See `test_videos/` folder for sample videos for testing

## Usage Examples

### Basic Navigation
```bash
# Start navigation
./run_orby.sh

# Audio output examples:
# "Path is clear. You're 3 meters forward from start."
# "STOP! Stairs going down ahead! 1.2 meters."
# "Person on your left. 2 meters."
```

### Voice Commands (NEW!)
```bash
# Activate with wake word
"Hey Orby"

# Ask about surroundings
"What's around me?"
"Is the path clear?"

# Save and navigate to locations
"Save this location as Kitchen"
"Take me to Kitchen"
"Where am I?"
```

### Map Management (NEW!)
```python
# Save current map
# (Press 's' during navigation, or programmatically)
slam.save_map("my_house")

# List saved maps
maps = slam.list_saved_maps()
# Returns: [{'map_name': 'my_house', 'timestamp': '20250102_143052', ...}]

# Load previous map
slam.load_map("data/maps/my_house_20250102_143052.pkl")
```

## Audio Warning System

OrbyGlasses uses a **priority-based audio system**:

### Priority Levels
1. **HIGHEST: Stair/Curb Warnings** (300ms interval)
   - `"STOP! Stairs going down ahead!"`
   - `"STOP! Curb ahead! 0.8 meters"`

2. **HIGH: Uncertain Depth** (400ms interval)
   - `"Caution! Person ahead. Distance unknown, use care"`

3. **MEDIUM: Danger Zone <1m** (400ms interval)
   - `"Stop now! Car very close on your left. Move right"`

4. **LOW: General Guidance** (1500ms interval)
   - `"Path clear. You're 5m forward, 2m left from start"`
   - `"Turn right in 3 meters to reach Kitchen"`

### Distance Terms
- **<0.3m**: "very close" → "Stop now!"
- **0.3-0.5m**: "close" → "Caution!"
- **0.5-1.0m**: "ahead" → "Watch out!"
- **1.0-2.5m**: "nearby" → "Be aware"
- **>2.5m**: "clear" → No urgent warning

## Configuration

### Production vs Development

**Production Config** (`config/config_production.yaml`):
- All critical features enabled for blind user navigation
- Optimized for safety and functionality
- Use this for actual navigation assistance

**Development Config** (`config/config.yaml`):
- Many features disabled for performance
- Optimized for development/debugging
- Faster, but less functional

### Customizing Configuration

Edit `config/config.yaml` (development) or `config/config_production.yaml` (production) to customize:

```yaml
camera:
  source: 0          # 0=built-in, 1=USB camera
  width: 640
  height: 480
  fps: 30

safety:
  danger_distance: 1.0       # Critical distance (meters)
  caution_distance: 2.5      # Warning distance (meters)

stair_detection:
  enabled: true              # Critical safety feature
  min_drop_height: 0.15      # Detect >15cm drops
  detection_distance: 2.5    # Scan up to 2.5m ahead

conversation:
  enabled: true              # Voice control
  activation_phrase: hey orby
  voice_input: true

slam:
  enabled: true
  loop_closure: true         # Relocalization support
  max_trajectory_length: 1000  # Memory management
  cleanup_interval: 500      # Prevent memory leaks
```

## Performance

### Typical Performance (M1/M2 Mac)
- **FPS**: 15-25 fps (real-time)
- **Latency**:
  - Detection: ~50-80ms
  - Depth: ~40-60ms
  - SLAM: ~80-120ms
  - Total: ~200-300ms
- **Memory**: ~500MB (with memory management)

### Optimization Tips
```yaml
# For better performance:
performance:
  depth_skip_frames: 2        # Skip depth every N frames
  enable_multithreading: true

# For lower latency:
slam:
  enabled: false              # Disable SLAM if not needed
```

## Safety Notice

**⚠️ IMPORTANT: This is an assistive technology, not a replacement for traditional mobility aids.**

- **Always use** with your white cane or guide dog
- **Recommended**: Have sighted assistance when first learning the system
- **Beta status**: This is version 0.9 - supervised testing recommended
- **Falls prevention**: Stair detection is active but not 100% foolproof

**Never rely solely on OrbyGlasses for critical safety decisions.**

## Current Status

### Production Configuration

**For blind users, use the production config** (`config/config_production.yaml`) which includes all critical features:

- ✅ **Depth Estimation** - Enabled for accurate distance measurement
- ✅ **Stair Detection** - Enabled for fall prevention
- ✅ **SLAM** - Enabled for indoor position tracking
- ✅ **Voice Control** - Enabled for hands-free operation
- ✅ **Indoor Navigation** - Enabled for path planning to saved locations

**Default config** (`config/config.yaml`) is optimized for development/debugging with many features disabled for performance.

### What Works ✅

**Core Features:**
- ✅ Real-time object detection (80 COCO classes)
- ✅ Depth estimation (0-10m range, monocular)
- ✅ Stair and curb detection (prevents falls)
- ✅ Voice control with wake word ("Hey Orby")
- ✅ Indoor position tracking (pySLAM)
- ✅ Map save/load functionality
- ✅ Turn-by-turn navigation (A* path planning)
- ✅ Priority-based audio warnings
- ✅ Emergency stop (keyboard or voice)
- ✅ Crash recovery (auto-disables loop closure on crashes)

**Technical:**
- ✅ pySLAM integration (real-time SLAM)
- ✅ Loop closure and relocalization
- ✅ Map persistence
- ✅ Memory management (prevents leaks)

### Known Limitations ⚠️

**Audio:**
- ⚠️ Audio latency: 1500-2000ms (target: <500ms for danger warnings)
- ⚠️ macOS TTS engine limitations

**Accuracy:**
- ⚠️ Monocular depth has ±25-40% error (inherent limitation)
- ⚠️ No glass door detection (transparent surfaces)
- ⚠️ No head-level hazard detection (focuses on ground-level)

**Platform:**
- ⚠️ macOS only (Linux/Windows support planned)

### In Development 🚧

**Priority 1:**
- Reduced audio latency (<500ms)
- Redundant audio (beeps + speech for urgent warnings)

**Priority 2:**
- Stereo spatial audio positioning
- Haptic feedback integration
- Head-level hazard detection

**Future:**
- Mobile app (iOS/Android)
- Multi-language support
- Glass door detection

## Troubleshooting

### Camera Issues
```bash
# Test camera
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'Failed')"

# Try different camera
# Edit config.yaml: camera.source: 1
```

### No Audio
```bash
# Test macOS speech
say "Testing audio"

# Check volume
# System Preferences → Sound → Output
```

### SLAM Crashes
```bash
# System auto-recovers, but if persistent:
# Edit config.yaml: slam.loop_closure: false

# Or use simple mode
./run_simple.sh
```

### Performance Issues
```bash
# Use fast mode
./run_orby.sh --fast

# Reduce camera resolution
# config.yaml: camera.width: 320, camera.height: 240

# Disable features
# config.yaml: slam.enabled: false
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for more solutions.

## Testing

```bash
# Run all tests
pytest

# Test specific modules
pytest tests/test_detection.py
pytest tests/test_slam.py

# Test with coverage
pytest --cov=src --cov-report=html

# Performance benchmarking
python tools/quick_validate.py
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Priority Areas
1. **Audio latency reduction** - Critical for safety (<500ms target)
2. **Stair detection improvements** - Increase accuracy (>90% target)
3. **Testing coverage** - Increase to 70%+ (currently ~30%)
4. **Accessibility features** - Screen reader support, voice-only setup
5. **Documentation** - API docs, tutorials, developer guide

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html
```

### Developer Documentation

See [docs/DEVELOPER.md](docs/DEVELOPER.md) for:
- Architecture overview
- pySLAM integration details
- SLAM tuning and relocalization
- Dense reconstruction
- Performance optimization
- Testing guidelines

## Community & Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/OrbyGlasses/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/OrbyGlasses/discussions)
- **Documentation**: [Full Docs](docs/)

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). We are committed to providing a welcoming, inclusive environment for all contributors and users.

## Privacy

**All processing happens locally on your device. Nothing is sent to the internet.**

- No cloud services
- No data collection
- No tracking
- Fully offline operation

## License

This project is licensed under the **GPL-3.0 License** - see [LICENSE](LICENSE) file.

You are free to:
- ✅ Use for personal navigation
- ✅ Modify and improve
- ✅ Distribute to others
- ✅ Use commercially (with attribution)

## Acknowledgments

### Technology Stack
- **Object Detection**: [YOLOv11](https://github.com/ultralytics/ultralytics) by Ultralytics
- **Depth Estimation**: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- **SLAM**: [pySLAM](https://github.com/luigifreda/pyslam) by Luigi Freda
- **Audio**: macOS built-in Text-to-Speech

### Inspiration
This project is dedicated to improving mobility and independence for blind and visually impaired people worldwide.

## Citation

If you use OrbyGlasses in research or publications, please cite:

```bibtex
@software{orbyglasses2025,
  title = {OrbyGlasses: AI-Powered Navigation for Blind Users},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/OrbyGlasses}
}
```

---

**Built with ❤️ for accessibility**

For detailed setup instructions, see [SETUP.md](SETUP.md)
For architecture details, see [docs/DEVELOPER.md](docs/DEVELOPER.md)
