# OrbyGlasses - Quick Start Guide

## How to Run OrbyGlasses

### Option 1: Standard Run (Recommended)

```bash
./run.sh
```

This will:
- ✅ Start the camera
- ✅ Load AI models
- ✅ Begin object detection
- ✅ Provide audio guidance
- ✅ Show depth visualization

**Press 'q' to quit**

---

### Option 2: Test Individual Modules

#### Test Depth Estimation
```bash
python3 src/core/depth_anything_v2.py
```
Shows your camera feed with depth colors.

#### Test SLAM Tracking
```bash
python3 src/navigation/simple_slam.py
```
Tracks your position as you move.

#### Test Dark Depth Visualizer
```bash
python3 src/visualization/depth_visualizer_2025.py
```
Shows dark-themed depth map.

---

### Option 3: Run Tests

```bash
# Test all modules
python3 -m pytest tests/ -v

# Test specific module
python3 -m pytest tests/test_depth_visualizer_2025.py -v
```

---

## What You'll See

When running `./run.sh`:

1. **Camera Window**: Shows detected objects with boxes
2. **Depth Map Window**: Shows distance in colors
   - Dark red = Very close (danger!)
   - Orange = Close (caution)
   - Green = Safe distance
   - Blue = Far away
3. **SLAM Map Window**: Shows your position as you move

## What You'll Hear

- **"Navigation system ready"** - System started
- **"Path clear"** - No obstacles ahead
- **"Chair at 2 meters"** - Object detected with distance
- **"Stop! Person ahead. Go left"** - Danger warning with direction

---

## System Requirements

### Already Installed
- ✅ Python 3.11
- ✅ PyTorch
- ✅ OpenCV
- ✅ All core libraries

### Optional (for full features)
```bash
# For YOLO-World text prompts
pip install git+https://github.com/ultralytics/CLIP.git

# For haptic feedback (if you have hardware)
pip install pyserial hidapi
```

---

## Troubleshooting

### "Camera not found"
Make sure your webcam is connected and not being used by another app.

### "No audio"
Check your speaker volume. The system uses macOS 'say' command.

### "Slow performance"
Lower the resolution in `config/config.yaml`:
```yaml
camera:
  width: 320
  height: 240
```

---

## Current Test Results

✅ **All Core Tests Passing:**
- Depth Visualizer: 11/11 passed
- SLAM Tracking: 4/4 passed
- Depth Estimation: Working
- Haptic Patterns: Working
- Audio Sonification: Working

---

## Features Available Now

### Working Features
- ✅ Real-time object detection (YOLOv11)
- ✅ Depth estimation (Depth Anything V2)
- ✅ Indoor SLAM tracking
- ✅ Dark-themed depth visualization
- ✅ Audio guidance with TTS
- ✅ Haptic pattern generation
- ✅ Audio sonification

### In Development
- ⏳ YOLO-World (needs CLIP library)
- ⏳ Bio-adaptive feedback
- ⏳ VLC beacon navigation

---

## Quick Commands

```bash
# Start system
./run.sh

# Stop system
Press 'q' in the video window

# Run all tests
pytest tests/ -v

# Check status
git status

# Update code
git pull
```

---

## File Structure

```
OrbyGlasses/
├── run.sh                          # Main launcher
├── src/
│   ├── main.py                     # Main application
│   ├── core/
│   │   ├── depth_anything_v2.py    # NEW: Better depth
│   │   ├── yolo_world_detector.py  # NEW: Open-vocab detection
│   │   └── detection.py            # Object detection
│   ├── navigation/
│   │   ├── simple_slam.py          # NEW: Indoor tracking
│   │   └── slam_system.py          # SLAM
│   └── visualization/
│       └── depth_visualizer_2025.py # NEW: Dark theme depth
├── tests/                          # All tests
└── docs/                           # Documentation
```

---

## Performance

**Current Performance (on your Mac):**
- FPS: ~15-20
- Latency: ~100ms
- Memory: ~2GB
- Detection: Working
- Depth: Working
- SLAM: Working

---

## Getting Help

1. Check `docs/USER_GUIDE_2025.md` for detailed guide
2. Run tests to verify: `pytest tests/ -v`
3. Check GitHub issues: https://github.com/jaskirat1616/Orby-Glasses/issues

---

## Next Steps

1. **Try it now**: `./run.sh`
2. **Walk around**: Watch the SLAM map track your position
3. **Test depth**: Move closer/farther from objects
4. **Listen**: Hear audio guidance as you navigate

**Enjoy your AI-powered navigation system!** 🚀
