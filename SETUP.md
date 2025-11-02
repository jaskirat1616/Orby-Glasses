# How to Install OrbyGlasses

Step-by-step guide to get OrbyGlasses running on your Mac.

## What You Need

### Computer
- Mac with M1, M2, M3, or M4 chip
- 8GB memory (16GB better)
- 5GB free space
- macOS 12.0 or newer

### Other
- Camera (built-in or USB)
- Headphones or speakers

## Installation Steps

### Step 1: Get OrbyGlasses

```bash
cd ~/Desktop
git clone https://github.com/jaskirat1616/Orby-Glasses.git
cd OrbyGlasses
```

### Step 2: Install Python Tools

```bash
# Install Homebrew (if you don't have it)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and other tools
brew install python@3.11 opencv ffmpeg portaudio
```

### Step 3: Install OrbyGlasses

```bash
# Install main programs
pip install -r requirements.txt

# Install navigation system
./install_pyslam.sh
```

This takes 10-15 minutes the first time.

### Step 4: Set Up Camera

Edit `config/config.yaml` and set your camera:

```yaml
camera:
  source: 0        # 0 = built-in, 1 = USB camera
  width: 320
  height: 240
```

Not sure which number? Try this:

```bash
python3 -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: Works')
        cap.release()
    else:
        print(f'Camera {i}: Not found')
"
```

### Step 5: Test It

```bash
./run_orby.sh
```

Press `q` to stop.

## Checking If It Works

Test each part:

```bash
# Test object detection
python3 -c "from src.core.detection import YOLODetector; print('✓ Object detection works')"

# Test distance measurement
python3 -c "from src.core.depth_anything_v2 import DepthAnythingV2; print('✓ Distance measurement works')"

# Test audio
python3 -c "import pyttsx3; e = pyttsx3.init(); e.say('Test'); e.runAndWait(); print('✓ Audio works')"
```

## Common Problems

### Camera Not Found

**Problem:** Can't find camera

**Fix:**
1. Go to System Settings > Privacy & Security > Camera
2. Allow Terminal to use camera
3. Try different camera numbers (0, 1, 2) in config.yaml

### Python Not Found

**Problem:** "python3: command not found"

**Fix:**
```bash
brew install python@3.11
```

### Installation Fails

**Problem:** Install script stops with errors

**Fix:**
```bash
# Install build tools
xcode-select --install

# Try again
./install_pyslam.sh
```

### No Audio

**Problem:** Can't hear anything

**Fix:**
```bash
# Test system audio
say "Testing audio"

# Check output device in System Settings > Sound
```

## Changing Settings

Edit `config/config.yaml`:

```yaml
# Camera settings
camera:
  source: 0           # Which camera to use
  width: 320          # Lower = faster, higher = better quality
  height: 240

# How close is too close
safety:
  danger_distance: 1.0        # Stop if closer (meters)
  min_safe_distance: 1.5      # Warning if closer (meters)

# Audio settings
audio:
  tts_rate: 220       # How fast to speak (words per minute)
  tts_volume: 0.9     # How loud (0-1)

# Indoor tracking
slam:
  enabled: true       # Turn tracking on/off
```

## Running OrbyGlasses

```bash
# Normal mode (15-25 FPS)
./run_orby.sh

# Fast mode (20-30 FPS)
./run_orby.sh --fast

# With navigation view
./run_orby.sh --nav
```

## Removing OrbyGlasses

```bash
# Remove programs
rm -rf ~/Desktop/OrbyGlasses

# Remove downloaded models (optional)
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/ultralytics/
```

## Getting Help

- Can't install? https://github.com/jaskirat1616/Orby-Glasses/issues
- Have questions? https://github.com/jaskirat1616/Orby-Glasses/discussions

## Next Steps

- Read [README.md](README.md) for how to use it
- Read [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for fixing problems
