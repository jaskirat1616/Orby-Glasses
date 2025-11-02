# OrbyGlasses Setup Guide

Complete installation and setup instructions for OrbyGlasses.

## System Requirements

### Hardware
- **Computer**: Mac with Apple Silicon (M1/M2/M3/M4/M5)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 5GB free space for models and dependencies
- **Camera**: Built-in webcam or external USB camera
- **Audio**: Speakers or headphones for audio guidance

### Software
- **OS**: macOS 12.0 (Monterey) or later
- **Python**: 3.10, 3.11, or 3.12
- **Homebrew**: For system dependencies

## Installation

### 1. Clone the Repository

```bash
cd ~/Desktop
git clone https://github.com/yourusername/OrbyGlasses.git
cd OrbyGlasses
```

### 2. Install System Dependencies

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install python@3.11 opencv ffmpeg portaudio
```

### 3. Install Python Dependencies

```bash
# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 4. Install pySLAM (SLAM System)

pySLAM requires a separate virtual environment:

```bash
# Run the installation script
./install_pyslam.sh
```

This will:
- Create a pySLAM virtual environment at `~/.python/venvs/pyslam`
- Clone and install the pySLAM library
- Install required dependencies (OpenCV, NumPy, etc.)
- Compile C++ extensions

**Note**: Installation takes 10-15 minutes on first run.

### 5. Install Ollama (Local LLM)

Ollama provides local language model support:

```bash
# Install Ollama
brew install ollama

# Start Ollama service
ollama serve &

# Pull required models
ollama pull moondream
ollama pull gemma2:4b
```

### 6. Download AI Models

The first time you run OrbyGlasses, it will automatically download:
- **YOLOv11n**: Object detection (~6MB)
- **Depth Anything V2 Small**: Depth estimation (~100MB)

Models are cached in `~/.cache/huggingface/` and `~/.cache/ultralytics/`.

### 7. Configure Camera

Edit `config/config.yaml` to set your camera source:

```yaml
camera:
  source: 0        # 0 = built-in camera, 1 = external USB camera
  width: 320       # Resolution (320x240 recommended for performance)
  height: 240
  fps: 30
```

To find your camera index:

```bash
python3 -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: Available')
        cap.release()
    else:
        print(f'Camera {i}: Not found')
"
```

## Running OrbyGlasses

### Basic Usage

```bash
# Standard mode (15-25 FPS)
./run_orby.sh

# Fast mode (20-30 FPS)
./run_orby.sh --fast

# With navigation panel
./run_orby.sh --nav
```

Press `q` to quit.

### Running Modes

**VO Mode (Visual Odometry - Trajectory Only):**
```bash
# Tracks camera movement without building a map
./run_vo_mode.sh
```

**Full SLAM Mode (With Mapping):**
```bash
# Builds a 3D map and tracks position
./run_slam_mode.sh
```

**Switch Between Modes:**
```bash
# Interactive mode switcher
./switch_mode.sh
```

### Verifying Installation

Test each component individually:

```bash
# Test object detection
python3 -c "from src.core.detection import YOLODetector; print('✓ Detection OK')"

# Test depth estimation
python3 -c "from src.core.depth_anything_v2 import DepthAnythingV2; print('✓ Depth OK')"

# Test pySLAM
source ~/.python/venvs/pyslam/bin/activate
python3 -c "from pyslam.slam.slam import Slam; print('✓ pySLAM OK')"

# Test audio
python3 -c "import pyttsx3; engine = pyttsx3.init(); engine.say('Test'); engine.runAndWait(); print('✓ Audio OK')"
```

## Configuration

### Main Configuration File

Edit `config/config.yaml` to customize behavior:

```yaml
# Camera settings
camera:
  source: 0
  width: 320
  height: 240
  fps: 30

# Object detection
models:
  yolo:
    model: yolov11n.pt
    confidence: 0.55      # 0-1, higher = fewer false positives
    device: mps           # mps (Apple Silicon) or cpu

# Depth estimation
  depth:
    model: depth-anything-v2-small
    max_depth: 10.0       # meters
    use_half_precision: true

# SLAM configuration
slam:
  enabled: true
  use_pyslam: true        # Recommended
  orb_features: 5000      # More = better tracking, slower
  loop_closure: true
  visualization: true

# Safety zones
safety:
  danger_distance: 1.0    # meters - immediate warning
  min_safe_distance: 1.5  # meters - caution zone
  max_detection_range: 10.0

# Audio feedback
audio:
  enabled: true
  rate: 175               # words per minute
  volume: 0.9             # 0-1
  min_time_between_warnings: 2.0  # seconds
```

### Fast Mode Configuration

For better performance on lower-end hardware:

```yaml
# Use config/config_fast.yaml or run with --fast flag
camera:
  width: 320
  height: 240

models:
  yolo:
    confidence: 0.60      # Higher threshold
  depth:
    skip_frames: 2        # Process every 2nd frame

slam:
  orb_features: 2000      # Fewer features
  visualization: false    # Disable 3D viewer
```

## Troubleshooting

### Camera Issues

**Problem**: Camera not detected
```bash
# Check camera availability
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"

# Try different camera indices (0, 1, 2)
# Update config.yaml with working index
```

**Problem**: Permission denied
```bash
# Grant camera access in System Settings > Privacy & Security > Camera
```

### pySLAM Issues

**Problem**: pySLAM import fails
```bash
# Verify pySLAM environment
source ~/.python/venvs/pyslam/bin/activate
python3 -c "import pyslam; print(pyslam.__file__)"

# Reinstall if needed
./install_pyslam.sh
```

**Problem**: Tracking loss
- Ensure adequate lighting
- Move camera slowly
- Avoid blank walls (add visual features)
- See TROUBLESHOOTING.md for SLAM tuning

### Audio Issues

**Problem**: No audio output
```bash
# Test system audio
say "This is a test"

# Test pyttsx3
python3 -c "import pyttsx3; e = pyttsx3.init(); e.say('Test'); e.runAndWait()"

# Check audio output device in System Settings
```

**Problem**: Audio is slow/laggy
- Use fast mode (`--fast`)
- Reduce `audio.rate` in config.yaml
- Disable non-essential features

### Performance Issues

**Problem**: Low FPS (<10)
```bash
# Use fast mode
./run_orby.sh --fast

# Reduce camera resolution in config.yaml
camera:
  width: 320
  height: 240

# Disable SLAM visualization
slam:
  visualization: false
```

**Problem**: High CPU usage
- Enable GPU acceleration (should be automatic on Apple Silicon)
- Reduce `slam.orb_features` in config.yaml
- Increase `depth.skip_frames`

### Model Download Issues

**Problem**: Models fail to download
```bash
# Set proxy if needed
export HF_ENDPOINT=https://huggingface.co
export HTTP_PROXY=your_proxy

# Pre-download models manually
python3 -c "
from ultralytics import YOLO
from transformers import AutoModel
YOLO('yolov11n.pt')
AutoModel.from_pretrained('depth-anything/Depth-Anything-V2-Small')
"
```

## Advanced Setup

### Using External Camera

For better quality:

```yaml
camera:
  source: 1           # External USB camera
  width: 640          # Higher resolution
  height: 480
  fps: 30
```

Recommended cameras:
- Logitech C920 (1080p)
- Logitech C270 (720p, budget)
- Any UVC-compatible webcam

### Custom SLAM Configuration

For challenging environments:

```yaml
slam:
  orb_features: 8000           # More features for low-texture
  scale_factor: 1.2            # Feature pyramid scale
  n_levels: 8                  # Pyramid levels
  loop_closure: true
  relocalization_threshold: 0.75
```

### Multi-Session Mapping

To save and load maps:

```yaml
slam:
  save_map: true
  map_path: data/maps/
  load_map_on_start: false
```

### Development Mode

For debugging:

```yaml
debug:
  enabled: true
  log_level: DEBUG
  save_frames: true
  frame_output_dir: data/debug_frames/
```

## Uninstallation

```bash
# Remove virtual environments
rm -rf venv
rm -rf ~/.python/venvs/pyslam

# Remove downloaded models (optional)
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/ultralytics/

# Remove Ollama (optional)
brew uninstall ollama

# Remove OrbyGlasses directory
cd ~
rm -rf Desktop/OrbyGlasses
```

## Next Steps

- Read [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- Review [ARCHITECTURE.md](ARCHITECTURE.md) to understand the system
- Check [README.md](README.md) for feature overview
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Getting Help

- **Issues**: https://github.com/yourusername/OrbyGlasses/issues
- **Discussions**: https://github.com/yourusername/OrbyGlasses/discussions
- **Documentation**: https://github.com/yourusername/OrbyGlasses/wiki
