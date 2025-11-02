# OrbyGlasses Troubleshooting Guide

Solutions to common issues and problems.

## Table of Contents

- [Camera Issues](#camera-issues)
- [SLAM Tracking Problems](#slam-tracking-problems)
- [Performance Issues](#performance-issues)
- [Audio Problems](#audio-problems)
- [Installation Errors](#installation-errors)
- [Model Loading Failures](#model-loading-failures)
- [macOS-Specific Issues](#macos-specific-issues)

---

## Camera Issues

### Camera Not Detected

**Symptoms**: Error "Could not open camera" or black screen

**Solutions**:

1. **Check camera availability**:
   ```bash
   python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL'); cap.release()"
   ```

2. **Try different camera indices**:
   ```bash
   # Test indices 0-4
   for i in {0..4}; do
     python3 -c "import cv2; cap = cv2.VideoCapture($i); print('Camera $i:', 'OK' if cap.isOpened() else 'FAIL'); cap.release()"
   done
   ```

3. **Update config.yaml** with working camera index:
   ```yaml
   camera:
     source: 1  # Change from 0 to 1, 2, etc.
   ```

4. **Grant camera permissions**:
   - Go to System Settings > Privacy & Security > Camera
   - Enable access for Terminal or your Python app

5. **Check if camera is in use**:
   ```bash
   # Close other apps using the camera (Zoom, FaceTime, etc.)
   lsof | grep "AppleCamera"
   ```

### Camera Shows But No Frames

**Symptoms**: Window opens but shows black or frozen image

**Solutions**:

1. **Check camera resolution**:
   ```yaml
   camera:
     width: 640   # Try standard resolutions
     height: 480
   ```

2. **Test camera directly**:
   ```bash
   python3 -c "
   import cv2
   cap = cv2.VideoCapture(0)
   ret, frame = cap.read()
   print('Frame captured:', ret)
   print('Frame shape:', frame.shape if ret else 'None')
   cap.release()
   "
   ```

3. **Update camera drivers**: Ensure macOS is up to date

### Poor Image Quality

**Solutions**:

1. **Increase resolution**:
   ```yaml
   camera:
     width: 640
     height: 480
   ```

2. **Adjust lighting**: Ensure adequate lighting in the environment

3. **Clean camera lens**: Physical cleaning may improve quality

---

## SLAM Tracking Problems

### Tracking Loss / "Lost Tracking" Warnings

**Symptoms**: SLAM loses position, map disappears, frequent "tracking lost" messages

**Causes**:
- Low-texture environments (blank walls)
- Fast camera movement
- Poor lighting
- Motion blur

**Solutions**:

1. **Tune SLAM parameters** (config.yaml):
   ```yaml
   slam:
     orb_features: 8000              # Increase from 5000
     scale_factor: 1.2
     n_levels: 8
     loop_closure: true
     relocalization_threshold: 0.75  # Lower for easier relocalization
   ```

2. **Environmental improvements**:
   - Add visual features (posters, furniture) to blank walls
   - Improve lighting (avoid darkness and backlighting)
   - Move camera slowly and smoothly

3. **Enable Visual Odometry fallback**:
   ```yaml
   slam:
     use_vo_fallback: true  # Continue tracking even if SLAM fails
   ```

4. **Switch to VO-only mode**:
   ```bash
   ./run_vo_mode.sh  # Trajectory tracking without mapping
   ```

### Frequent Relocalization Attempts

**Symptoms**: System constantly trying to relocalize, jerky position updates

**Solutions**:

1. **Aggressive relocalization tuning**:
   ```yaml
   slam:
     relocalization_threshold: 0.6   # Lower = more aggressive
     min_inliers: 15                 # Lower = more permissive
     max_reprojection_error: 3.0     # Higher = more tolerant
   ```

2. **Enable loop closure**:
   ```yaml
   slam:
     loop_closure: true
     loop_closure_threshold: 0.7
   ```

3. **Reset map and restart**:
   ```bash
   rm -rf data/maps/*
   ./run_orby.sh
   ```

### Map Drift Over Time

**Symptoms**: Mapped positions don't match reality after several minutes

**Solutions**:

1. **Enable bundle adjustment**:
   ```yaml
   slam:
     bundle_adjustment: true
     ba_local_window: 20
   ```

2. **Improve loop closure**:
   ```yaml
   slam:
     loop_closure: true
     loop_closure_frequency: 10  # Check every 10 keyframes
   ```

3. **Use more features**:
   ```yaml
   slam:
     orb_features: 10000  # More features = better accuracy
   ```

---

## Performance Issues

### Low FPS (<10 FPS)

**Symptoms**: Choppy video, slow response, lag

**Solutions**:

1. **Use fast mode**:
   ```bash
   ./run_orby.sh --fast
   ```

2. **Reduce camera resolution**:
   ```yaml
   camera:
     width: 320
     height: 240
   ```

3. **Disable expensive features**:
   ```yaml
   slam:
     visualization: false  # Disable 3D viewer
     orb_features: 2000    # Reduce features

   depth:
     skip_frames: 2        # Process every 2nd frame

   features:
     mapping3d: false
     occupancy_grid_3d: false
     trajectory_prediction: false
   ```

4. **Check CPU usage**:
   ```bash
   top -pid $(pgrep -f main.py)
   ```

### High Memory Usage

**Symptoms**: System slows down, "Memory pressure is high" warning

**Solutions**:

1. **Limit map size**:
   ```yaml
   slam:
     max_keyframes: 100      # Limit stored keyframes
     max_map_points: 5000    # Limit map points
   ```

2. **Enable memory cleanup**:
   ```yaml
   slam:
     cull_redundant_keyframes: true
     culling_frequency: 50
   ```

3. **Disable caching**:
   ```yaml
   depth:
     cache_size: 0  # Disable depth cache
   ```

### GPU/MPS Not Being Used

**Symptoms**: High CPU usage, low GPU usage in Activity Monitor

**Solutions**:

1. **Verify MPS availability**:
   ```bash
   python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
   ```

2. **Force MPS in config**:
   ```yaml
   models:
     yolo:
       device: mps
     depth:
       device: mps
   ```

3. **Check PyTorch version**:
   ```bash
   pip install --upgrade torch torchvision
   ```

---

## Audio Problems

### No Audio Output

**Symptoms**: No spoken warnings, silence

**Solutions**:

1. **Test system audio**:
   ```bash
   say "This is a test"
   ```

2. **Test pyttsx3**:
   ```bash
   python3 -c "import pyttsx3; e = pyttsx3.init(); e.say('Test'); e.runAndWait()"
   ```

3. **Check audio enabled in config**:
   ```yaml
   audio:
     enabled: true
   ```

4. **Select audio output device**: System Settings > Sound > Output

5. **Reinstall pyttsx3**:
   ```bash
   pip uninstall pyttsx3 pyobjc
   pip install pyttsx3 pyobjc
   ```

### Audio Is Slow/Laggy (>2 seconds delay)

**Symptoms**: Warnings come too late to be useful

**Solutions**:

1. **Increase speech rate**:
   ```yaml
   audio:
     rate: 200  # Increase from 175 (words per minute)
   ```

2. **Reduce processing time**:
   - Use fast mode
   - Disable expensive features
   - Reduce camera resolution

3. **Pre-generate common phrases**: (Future enhancement)

### Audio Cuts Out or Stutters

**Solutions**:

1. **Increase minimum time between warnings**:
   ```yaml
   audio:
     min_time_between_warnings: 3.0  # Increase from 2.0
   ```

2. **Check audio queue**:
   ```yaml
   audio:
     queue_size: 5  # Limit queued messages
   ```

---

## Installation Errors

### pySLAM Installation Fails

**Symptoms**: `./install_pyslam.sh` fails with compiler errors

**Solutions**:

1. **Install Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

2. **Install required dependencies**:
   ```bash
   brew install cmake opencv eigen boost
   ```

3. **Check Python version**:
   ```bash
   python3 --version  # Should be 3.10, 3.11, or 3.12
   ```

4. **Manual installation**:
   ```bash
   cd third_party/pyslam
   python3 -m venv ~/.python/venvs/pyslam
   source ~/.python/venvs/pyslam/bin/activate
   pip install -r requirements.txt
   ./install_dependencies.sh
   ```

### "No module named 'pyslam'" Error

**Solutions**:

1. **Check PYTHONPATH** (should be set by run_orby.sh):
   ```bash
   echo $PYTHONPATH
   # Should include: /path/to/OrbyGlasses/third_party/pyslam
   ```

2. **Use the correct launch script**:
   ```bash
   ./run_orby.sh  # NOT: python3 src/main.py
   ```

3. **Activate pySLAM venv**:
   ```bash
   source ~/.python/venvs/pyslam/bin/activate
   ```

### Pip Install Fails with "externally-managed-environment"

**Solution**:

```bash
# Use virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Model Loading Failures

### YOLOv11 Download Fails

**Solutions**:

1. **Check internet connection**

2. **Manual download**:
   ```bash
   mkdir -p models
   cd models
   wget https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov11n.pt
   ```

3. **Update config to use local model**:
   ```yaml
   models:
     yolo:
       model: models/yolov11n.pt
   ```

### Depth Model Download Fails

**Solutions**:

1. **Pre-download from Hugging Face**:
   ```bash
   python3 -c "
   from transformers import AutoModel
   model = AutoModel.from_pretrained('depth-anything/Depth-Anything-V2-Small')
   "
   ```

2. **Set Hugging Face cache**:
   ```bash
   export HF_HOME=~/.cache/huggingface
   ```

3. **Use proxy if needed**:
   ```bash
   export HTTP_PROXY=http://your-proxy:port
   export HTTPS_PROXY=http://your-proxy:port
   ```

### Ollama Models Not Found

**Solutions**:

1. **Start Ollama service**:
   ```bash
   ollama serve &
   ```

2. **Pull required models**:
   ```bash
   ollama pull moondream
   ollama pull gemma2:4b
   ```

3. **Check Ollama status**:
   ```bash
   ollama list
   ```

---

## macOS-Specific Issues

### "Operation not permitted" Error

**Solution**: Grant Full Disk Access
- System Settings > Privacy & Security > Full Disk Access
- Add Terminal or your IDE

### MPS Backend Errors

**Symptoms**: "MPS backend out of memory" or similar

**Solutions**:

1. **Reduce batch size**:
   ```yaml
   models:
     yolo:
       batch_size: 1
   ```

2. **Use half precision**:
   ```yaml
   models:
     depth:
       use_half_precision: true
   ```

3. **Fallback to CPU**:
   ```yaml
   models:
     yolo:
       device: cpu
     depth:
       device: cpu
   ```

### Gatekeeper Blocks Execution

**Symptoms**: "cannot be opened because the developer cannot be verified"

**Solution**:
```bash
# Remove quarantine attribute
xattr -d com.apple.quarantine run_orby.sh
chmod +x run_orby.sh
```

---

## Getting Help

If none of these solutions work:

1. **Enable debug logging**:
   ```yaml
   debug:
     enabled: true
     log_level: DEBUG
   ```

2. **Check logs**:
   ```bash
   tail -f data/logs/orbyglasses.log
   ```

3. **Report an issue**:
   - https://github.com/yourusername/OrbyGlasses/issues
   - Include: OS version, Python version, error message, logs

4. **Join discussions**:
   - https://github.com/yourusername/OrbyGlasses/discussions

---

## Quick Reference

### Emergency Fixes

```bash
# Reset everything
rm -rf data/maps/* data/logs/*
./run_orby.sh

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Reinstall pySLAM
rm -rf ~/.python/venvs/pyslam
./install_pyslam.sh

# Reset configuration
git checkout config/config.yaml
```

### Performance Tweaks (Copy-Paste)

```yaml
# Fast mode configuration
camera:
  width: 320
  height: 240

slam:
  orb_features: 2000
  visualization: false

depth:
  skip_frames: 2

models:
  yolo:
    confidence: 0.60

features:
  mapping3d: false
  occupancy_grid_3d: false
  trajectory_prediction: false
```

### Debugging Commands

```bash
# Test individual components
python3 -c "from src.core.detection import YOLODetector; print('Detection OK')"
python3 -c "from src.core.depth_anything_v2 import DepthAnythingV2; print('Depth OK')"
python3 -c "import pyttsx3; e=pyttsx3.init(); e.say('Audio OK'); e.runAndWait()"

# Check system resources
top -l 1 | grep -E "^CPU|^PhysMem"
system_profiler SPCameraDataType  # List cameras

# Monitor logs in real-time
tail -f data/logs/orbyglasses.log
```
