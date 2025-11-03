# OrbyGlasses Visualization Modes

## Overview

OrbyGlasses now supports **4 distinct visualization modes** for different use cases:

| Mode | Windows | Performance | Use Case |
|------|---------|-------------|----------|
| **`features`** | 2 | 40-60 FPS | **Default** - Feature matching visualization |
| `basic` | 1 | 60-80 FPS | Maximum speed, navigation only |
| `full_slam` | 4+ | 20-30 FPS | Complete SLAM debugging |
| `vo` | 2 | 50-70 FPS | Visual odometry only (no map) |

## Usage

### 1. Feature Matching Mode (Default, Recommended)

```bash
# Using --mode flag
./run_orby.sh --mode features

# Or set in config (default)
# visualization_mode:
#   mode: 'features'

# Webcam
python3 src/main.py --mode features

# Video file
./run_orby.sh --video file.mp4 --mode features
```

**Windows:**
- OrbyGlasses (main camera view)
- Feature Matching (side-by-side with green lines)

**Best for:**
- Understanding SLAM tracking
- Debugging feature detection
- Real-time performance with visualization
- Most users

### 2. Basic Mode (Fastest)

```bash
./run_orby.sh --mode basic

# Or
python3 src/main.py --mode basic
```

**Windows:**
- OrbyGlasses only

**Best for:**
- Maximum FPS
- Navigation only (no SLAM visualization)
- Production use
- Low-power devices

### 3. Full SLAM Mode (Complete Debugging)

```bash
./run_orby.sh --mode full_slam

# Or
python3 src/main.py --mode full_slam
```

**Windows:**
- OrbyGlasses (main camera view)
- Feature Matching (our visualization)
- pySLAM 3D Viewer (Pangolin - rotating 3D map)
- pySLAM Trajectory Plot
- pySLAM Feature Viewer

**Best for:**
- Complete SLAM debugging
- Map quality analysis
- Research and development
- When FPS doesn't matter

### 4. Visual Odometry Mode (Lightweight Tracking)

```bash
./run_orby.sh --mode vo

# Or
python3 src/main.py --mode vo
```

**Windows:**
- OrbyGlasses (main camera view)
- VO Tracking visualization

**Best for:**
- Position tracking without map building
- Lower memory usage
- Faster than full SLAM
- When you don't need loop closure

## Configuration File

Set default mode in `config/config.yaml`:

```yaml
# Visualization modes
visualization_mode:
  # Options: 'full_slam', 'vo', 'features', 'basic'
  mode: 'features'  # Default mode
```

## Mode Comparison

### Feature Matching Mode (features)
```
✅ SLAM enabled
✅ Feature matching visualization
❌ No 3D map viewer
❌ No trajectory plot
🎯 Performance: 40-60 FPS
💡 Best for most users
```

### Basic Mode (basic)
```
✅ SLAM enabled
❌ No visualizations
🎯 Performance: 60-80 FPS
💡 Fastest, production use
```

### Full SLAM Mode (full_slam)
```
✅ SLAM enabled
✅ Feature matching
✅ 3D map viewer
✅ Trajectory plot
✅ All debug windows
🎯 Performance: 20-30 FPS
💡 Complete debugging
```

### Visual Odometry Mode (vo)
```
✅ VO enabled
✅ Position tracking
❌ No map building
❌ No loop closure
🎯 Performance: 50-70 FPS
💡 Lightweight tracking
```

## Examples

### Quick Start (Default)
```bash
# Webcam with feature matching
./run_orby.sh

# Same as:
./run_orby.sh --mode features
```

### Performance Testing
```bash
# Test all modes with same video
VIDEO=/Users/jaskiratsingh/Downloads/recone.mp4

./run_orby.sh --video $VIDEO --mode basic      # Baseline FPS
./run_orby.sh --video $VIDEO --mode features   # With feature viz
./run_orby.sh --video $VIDEO --mode full_slam  # Full debugging
./run_orby.sh --video $VIDEO --mode vo         # VO only
```

### Different Cameras
```bash
# Built-in webcam, feature mode
./run_orby.sh --mode features

# External USB camera, basic mode (fastest)
./run_orby.sh --video 1 --mode basic

# External camera, full SLAM
./run_orby.sh --video 1 --mode full_slam
```

## Switching Modes

### At Runtime
Simply restart with different `--mode` flag:
```bash
# Currently running in features mode
# Ctrl+C to quit

# Restart in full SLAM mode
./run_orby.sh --mode full_slam
```

### In Config
Edit `config/config.yaml`:
```yaml
visualization_mode:
  mode: 'basic'  # Change to desired mode
```

Then run without `--mode` flag to use config default:
```bash
./run_orby.sh  # Uses mode from config
```

## Deprecated Flags

Old flags still work but show deprecation warning:

```bash
# OLD (deprecated)
./run_orby.sh --show-features
./run_orby.sh --separate-slam

# NEW (recommended)
./run_orby.sh --mode features
./run_orby.sh --mode full_slam
```

## Performance Optimization by Mode

### For Maximum FPS (basic mode)
```yaml
# config/config.yaml
visualization_mode:
  mode: 'basic'

slam:
  orb_features: 800  # Fewer features

models:
  depth:
    enabled: false  # No depth

logging:
  level: WARNING  # Minimal logging
```
**Expected: 60-80 FPS**

### For Best Visualization (features mode)
```yaml
visualization_mode:
  mode: 'features'

slam:
  orb_features: 1200  # Balanced

models:
  depth:
    enabled: false  # Keep disabled for speed
```
**Expected: 40-60 FPS**

### For Complete Debugging (full_slam mode)
```yaml
visualization_mode:
  mode: 'full_slam'

slam:
  orb_features: 1500  # More features
  loop_closure: true  # Enable loop closure
```
**Expected: 20-30 FPS**

## Troubleshooting

### Mode Not Recognized
```
Error: argument --mode: invalid choice: 'xyz'
```
**Solution**: Use one of: `features`, `basic`, `full_slam`, `vo`

### Feature Matching Window Not Showing
**Check:**
1. Mode is set to `features` or `full_slam`
2. SLAM initialized (wait 2-3 seconds)
3. No errors in logs

### Too Slow in Full SLAM Mode
**Solution**: Switch to `features` mode for better FPS:
```bash
./run_orby.sh --mode features
```

### Need Even More Speed
**Solution**: Use `basic` mode:
```bash
./run_orby.sh --mode basic
```

## Quick Reference

```bash
# Most users (default)
./run_orby.sh                    # Uses config default (features)
./run_orby.sh --mode features    # Explicitly set features mode

# Maximum speed
./run_orby.sh --mode basic

# Complete debugging
./run_orby.sh --mode full_slam

# Visual odometry only
./run_orby.sh --mode vo

# With video file
./run_orby.sh --video file.mp4 --mode features

# With webcam index
./run_orby.sh --video 1 --mode basic
```

## Summary

**Recommended for daily use:** `features` mode
- Good balance of visualization and performance
- Shows what SLAM is doing
- 40-60 FPS
- Only 2 windows (not cluttered)

**Set as default in config:**
```yaml
visualization_mode:
  mode: 'features'
```

Then simply run:
```bash
./run_orby.sh  # Always uses your preferred mode
```
