# OrbyGlasses Quick Start Guide

## Run with Feature Visualization (Recommended)

Shows the main OrbyGlasses window + a lightweight feature tracking window:

```bash
./run_orby.sh --video /path/to/video.mp4 --show-features
```

**What you'll see:**
- **OrbyGlasses Window**: Main camera view with object detection
- **Feature Tracking Window**: SLAM statistics and tracking quality

**Performance**: ~40-60 FPS (very lightweight)

## Alternative Display Modes

### 1. Basic Mode (Fastest)
```bash
./run_orby.sh --video /path/to/video.mp4
```
- Only shows OrbyGlasses window
- No SLAM visualization
- Best for maximum speed

### 2. Full SLAM Mode (Slowest)
```bash
./run_orby.sh --video /path/to/video.mp4 --separate-slam
```
- Shows OrbyGlasses + full pySLAM 3D viewer
- Complete map visualization
- ~20-30 FPS (heavier overhead)

### 3. Headless Mode (No Display)
```bash
./run_orby.sh --video /path/to/video.mp4 --no-display
```
- No windows (for servers/testing)
- Logs only

## Configuration Presets

### For Speed (Current Default)
```yaml
# config/config.yaml
slam:
  enabled: true
  orb_features: 1200
models:
  depth:
    enabled: false  # Major speedup
```
**Performance**: 40-60 FPS

### For Accuracy (with Depth)
```yaml
# config/config.yaml
slam:
  enabled: true
  orb_features: 2000
models:
  depth:
    enabled: true  # Better obstacle distances
```
**Performance**: 15-20 FPS

### For Maximum Speed (Minimal)
```yaml
# Use config_fast_slam.yaml
./run_orby.sh --config config/config_fast_slam.yaml --video file.mp4
```
**Performance**: 60-80 FPS

## Common Commands

```bash
# Basic run with feature view
./run_orby.sh --video video.mp4 --show-features

# Save output video
./run_orby.sh --video video.mp4 --show-features --save-video

# Use different config
./run_orby.sh --config config/config_fast.yaml --video video.mp4

# Live camera (camera index 0)
./run_orby.sh

# Specific camera
./run_orby.sh --video 1  # Camera index 1
```

## Keyboard Controls

While running:
- `q` - Quit
- `Ctrl+C` - Force quit

## Troubleshooting

### Slow Performance
1. Disable depth model (major speedup)
2. Reduce ORB features: `orb_features: 800`
3. Lower resolution in config

### No Features Detected
1. Increase features: `orb_features: 2000`
2. Check video has texture/contrast
3. Ensure good lighting

### SLAM Not Working
1. Check config: `slam.enabled: true`
2. Verify ORB2 is built: `cd third_party/pyslam && ./install_all.sh`
3. Check logs for errors

## File Structure

```
OrbyGlasses/
├── run_orby.sh              # Main launcher
├── src/main.py              # Main application
├── config/
│   ├── config.yaml          # Main config (optimized for speed)
│   ├── config_fast.yaml     # Previous fast config
│   └── config_fast_slam.yaml # Ultra-fast config
└── docs/
    ├── PERFORMANCE_FIXES.md          # Initial fixes
    ├── SPEED_OPTIMIZATION_GUIDE.md   # Performance tuning
    └── FEATURE_VISUALIZATION_GUIDE.md # Feature view guide
```

## Performance Summary

| Mode | FPS | Windows | Best For |
|------|-----|---------|----------|
| `--show-features` | 40-60 | 2 | **Most users** - Good balance |
| Basic | 50-70 | 1 | Maximum speed |
| `--separate-slam` | 20-30 | 3+ | Full SLAM visualization |
| Headless | 60-80 | 0 | Background processing |

## Getting Help

- Check `FEATURE_VISUALIZATION_GUIDE.md` for detailed feature view docs
- Check `SPEED_OPTIMIZATION_GUIDE.md` for performance tuning
- Check `PERFORMANCE_FIXES.md` for initial setup fixes
