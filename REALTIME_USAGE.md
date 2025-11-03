# Real-time Webcam Usage with Feature Matching

## Quick Start - Live Webcam

### With Feature Matching Visualization (Recommended)

```bash
# Default webcam (camera 0) with feature matching
./run_orby.sh --show-features

# Or directly:
python3 src/main.py --show-features
```

### What You'll See

**Window 1: OrbyGlasses** (Main Camera View)
- Live camera feed with object detection boxes
- FPS counter
- Tracking status
- Navigation overlays

**Window 2: Feature Matching** (Real SLAM Visualization)
- **Left side**: Reference keyframe (grayscale)
- **Right side**: Current frame (grayscale)
- **Green lines**: Successfully matched ORB features between frames
- **Text overlay**: Frame labels and match count

This is the actual SLAM feature tracking - you'll see lines connecting features as you move the camera!

### Different Webcam Indices

```bash
# Camera 0 (default, usually built-in webcam)
./run_orby.sh --show-features

# Camera 1 (external USB webcam)
./run_orby.sh --video 1 --show-features

# Camera 2
./run_orby.sh --video 2 --show-features
```

## Display Modes

### 1. Basic Mode (Fastest - No Feature Visualization)
```bash
./run_orby.sh
```
- Only OrbyGlasses window
- Maximum FPS (~60-80)
- Good for navigation only

### 2. Feature Matching Mode (Recommended)
```bash
./run_orby.sh --show-features
```
- OrbyGlasses + Feature Matching windows
- Shows actual SLAM tracking visually
- **~40-60 FPS** (slight overhead from drawing)
- **Best for understanding SLAM performance**

### 3. Full SLAM Mode (All Visualizations)
```bash
./run_orby.sh --show-features --separate-slam
```
- OrbyGlasses window
- Feature Matching window
- pySLAM 3D map viewer (Pangolin)
- pySLAM trajectory plot
- ~20-30 FPS (heavy overhead)
- Use when you need complete SLAM debugging

### 4. Headless Mode (No Display)
```bash
./run_orby.sh --no-display
```
- No windows at all
- Maximum speed
- Logs only
- Good for background processing/testing

## Configuration for Real-time

The default `config/config.yaml` is already optimized for real-time webcam usage:

```yaml
slam:
  enabled: true
  orb_features: 1200  # Good balance for real-time

models:
  depth:
    enabled: false  # Disabled for speed (2-3x faster)

performance:
  stats_interval: 50  # Frequent FPS updates

logging:
  level: WARNING  # Minimal logging overhead
```

## Expected Performance

| Configuration | FPS | Use Case |
|--------------|-----|----------|
| Webcam + Feature Matching | 40-60 | **Recommended for SLAM visualization** |
| Webcam Only | 60-80 | Maximum speed, navigation only |
| Webcam + Full SLAM | 20-30 | Complete debugging |

## Tips for Best Real-time Performance

### 1. Good Lighting
- Ensure adequate lighting in the room
- Avoid backlighting (camera facing bright windows)
- Consistent lighting helps feature detection

### 2. Textured Environment
- Plain white walls = few features
- Posters, furniture, patterns = many features
- SLAM works best with textured environments

### 3. Smooth Movement
- Move camera slowly and smoothly
- Fast jerky movements can lose tracking
- Let SLAM initialize for 2-3 seconds before moving

### 4. Monitor Feature Count
In the Feature Matching window, you should see:
- **Good**: 100-200 matched features (green lines)
- **OK**: 50-100 matched features
- **Poor**: <50 matched features (may lose tracking)

### 5. Watch for Warnings
If you see many of these in the terminal:
```
ORBextractor::DistributeOctTree() - warning - nIni = 0
```
**Solution**: Increase features or improve lighting/texture

## Troubleshooting Real-time Issues

### Problem: Low FPS (<30)
**Solutions**:
1. Disable depth model (already disabled in default config)
2. Reduce features: `slam.orb_features: 800`
3. Lower camera resolution in config:
   ```yaml
   camera:
     width: 640
     height: 480
   ```

### Problem: SLAM Not Initializing
**Symptoms**: "NOT_INITIALIZED" message stays for >5 seconds

**Solutions**:
1. Move camera slowly side-to-side
2. Point camera at textured area (not plain wall)
3. Check for sufficient lighting
4. Ensure camera is working: `ls /dev/video*`

### Problem: Tracking Lost Frequently
**Symptoms**: "Tracking: LOST" messages

**Solutions**:
1. Slow down camera movement
2. Add more texture to environment
3. Increase features:
   ```yaml
   slam:
     orb_features: 1500
   ```
4. Check lighting consistency

### Problem: Feature Matching Window Not Showing
**Causes**:
- SLAM not initialized yet (wait 2-3 seconds)
- No reference keyframe selected
- `--show-features` flag not provided

**Solutions**:
1. Ensure you're using `--show-features` flag
2. Wait for SLAM to initialize
3. Check logs for errors
4. Verify SLAM is enabled in config

### Problem: No Features Detected (Empty Matching Window)
**Symptoms**: Feature Matching window shows frames but no green lines

**Solutions**:
1. Move camera to textured area
2. Improve lighting
3. Increase ORB features to 2000
4. Check camera is not out of focus

## Camera Selection

### Find Available Cameras

```bash
# macOS
ls /dev/video*

# List camera devices
system_profiler SPCameraDataType
```

### Test Different Cameras

```bash
# Try each camera index until you find the right one
./run_orby.sh --video 0 --show-features  # Usually built-in
./run_orby.sh --video 1 --show-features  # Usually external USB
./run_orby.sh --video 2 --show-features  # Additional cameras
```

## Advanced: Optimize for Your Webcam

### High-Resolution Webcam (1080p)
```yaml
# config/config.yaml
camera:
  width: 1280  # Higher resolution
  height: 720
  fps: 30

slam:
  orb_features: 1500  # More features for higher res
```

### Low-End Webcam (480p)
```yaml
camera:
  width: 640
  height: 480
  fps: 30

slam:
  orb_features: 800  # Fewer features for speed
```

## Keyboard Controls (Real-time)

While running:
- **`q`** - Quit application
- **Ctrl+C** - Force quit (terminal)

## Example Sessions

### Session 1: Basic Navigation Testing
```bash
./run_orby.sh --show-features
# Move camera around room
# Observe feature matching
# Check FPS stays >40
# Press 'q' to quit
```

### Session 2: SLAM Quality Check
```bash
./run_orby.sh --show-features --separate-slam
# Initialize SLAM (wait 2-3 seconds)
# Move camera in loop around room
# Watch 3D map build in Pangolin viewer
# Check feature matching stays consistent
# Verify trajectory is smooth
```

### Session 3: Performance Benchmarking
```bash
./run_orby.sh --show-features
# Keep camera still
# Note baseline FPS
# Move camera slowly
# Note FPS during movement
# Should stay >40 FPS consistently
```

## Summary

**For Most Users (Recommended):**
```bash
./run_orby.sh --show-features
```

This gives you:
✅ Real-time camera view with detection
✅ Visual SLAM feature matching
✅ Good performance (40-60 FPS)
✅ Easy to understand what SLAM is doing
✅ No map viewer clutter

**The feature matching window shows you exactly what ORB-SLAM2 is doing** - tracking features between frames with green lines connecting matched points!

## Quick Test Script

Created for you at: `test_webcam_features.sh`

```bash
./test_webcam_features.sh
```

Runs a quick test with your default webcam + feature visualization.
