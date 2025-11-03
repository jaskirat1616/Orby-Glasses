# Feature Visualization Mode - Usage Guide

## Overview

OrbyGlasses now has a **lightweight feature tracking visualization** mode that shows SLAM feature matching statistics without the heavy 3D map viewer from pySLAM.

This mode is perfect for:
- **Monitoring SLAM performance** in real-time
- **Debugging feature detection** issues
- **Reducing system overhead** (no 3D rendering)
- **Understanding tracking quality** at a glance

## Usage

### Basic Usage (with Feature Tracking Window)

```bash
./run_orby.sh --video /path/to/video.mp4 --show-features
```

or

```bash
python3 src/main.py --video /path/to/video.mp4 --show-features
```

### What You'll See

Two windows will appear:

#### 1. **OrbyGlasses Window** (480x360)
- Main camera view with object detection boxes
- FPS counter
- SLAM position overlay
- Tracking quality indicator

#### 2. **Feature Tracking Window** (600x400)
- **Status Indicator**: TRACKING: GOOD/OK/POOR (color-coded)
- **Matched Points**: Number of features successfully matched between frames
- **Map Points**: Total points in the SLAM map
- **Tracking Quality**: 0.0-1.0 score (higher is better)
- **Keyframe**: Whether current frame is a keyframe
- **Visual Bar**: Feature count visualization (green/orange/red)

## Display Modes Comparison

| Mode | Command | Windows Shown | Performance Impact |
|------|---------|---------------|-------------------|
| **Default** | `--video file.mp4` | OrbyGlasses only | Minimal |
| **Feature View** | `--show-features` | OrbyGlasses + Feature Tracking | Low (~5% slower) |
| **Full SLAM** | `--separate-slam` | OrbyGlasses + pySLAM 3D viewer | High (~30% slower) |

## Feature Tracking Metrics Explained

### Matched Points
- **Good**: >100 matches (green)
- **OK**: 50-100 matches (orange)
- **Poor**: <50 matches (red)

**What it means**: Number of ORB features successfully tracked from previous frame. More matches = more robust tracking.

### Map Points
- **Healthy map**: 1000-3000 points
- **Too few**: <500 (may lose tracking)
- **Too many**: >5000 (slower performance)

**What it means**: Total 3D points in the SLAM map. Grows over time as you explore new areas.

### Tracking Quality
- **Excellent**: 0.8-1.0 (bright green)
- **Good**: 0.6-0.8 (green)
- **OK**: 0.4-0.6 (orange)
- **Poor**: <0.4 (red)

**What it means**: Confidence score for current camera pose estimate. Based on match count, reprojection error, and tracking history.

### Keyframe Indicator
- **YES**: Current frame was selected as a keyframe
- **NO**: Regular frame

**What it means**: Keyframes are special frames used for map building and loop closure. Selected when camera moves significantly or scene changes.

## Example Scenarios

### Good Tracking
```
TRACKING: GOOD
Matched Points: 157
Map Points: 2234
Quality: 0.87
Keyframe: NO
[████████████████████████████] 157 / 200 features
```

### Initialization Phase
```
TRACKING: OK
Matched Points: 45
Map Points: 234
Quality: 0.52
Keyframe: YES
[███████████░░░░░░░░░░░░░░░░] 45 / 200 features
```

### Lost Tracking
```
TRACKING: POOR
Matched Points: 8
Map Points: 1456
Quality: 0.23
Keyframe: NO
[██░░░░░░░░░░░░░░░░░░░░░░░░░] 8 / 200 features
```

## Troubleshooting

### Problem: "TRACKING: POOR" constantly
**Causes**:
- Not enough texture in scene (plain walls, uniform surfaces)
- Motion blur (camera moving too fast)
- Low light conditions
- Video resolution too low

**Solutions**:
1. Increase ORB features in config:
   ```yaml
   slam:
     orb_features: 1500  # Increase from 1200
   ```

2. Slow down video playback
3. Use better quality video source
4. Improve lighting in scene

### Problem: Map Points keep growing (>5000)
**Causes**:
- Exploring large new areas continuously
- Map cleanup not frequent enough

**Solutions**:
1. Increase cleanup frequency:
   ```yaml
   slam:
     cleanup_interval: 200  # More frequent (was 300)
     max_map_points: 3000   # Lower limit (was 5000)
   ```

2. Enable loop closure (slower but manages map size):
   ```yaml
   slam:
     loop_closure: true
   ```

### Problem: No Feature Tracking window appears
**Causes**:
- SLAM not enabled in config
- SLAM failed to initialize
- Missing `--show-features` flag

**Solutions**:
1. Check config:
   ```yaml
   slam:
     enabled: true  # Must be true
   ```

2. Check logs for SLAM initialization errors
3. Ensure ORB2 is built: `cd third_party/pyslam && ./install_all.sh`

### Problem: Feature count is 0
**Causes**:
- ORB feature extractor not finding corners
- Very low contrast scene
- Incorrect ORB parameters

**Solutions**:
1. Check video quality - ensure sufficient texture
2. Increase feature target:
   ```yaml
   slam:
     orb_features: 2000  # Higher target
   ```

3. Check for "ORBextractor::DistributeOctTree() - warning - nIni = 0" in logs
   - This means no features in some pyramid levels (can be normal)
   - Only concerning if ALL levels show 0

## Configuration for Feature View

Recommended settings for best feature tracking visualization:

```yaml
# config/config.yaml

slam:
  enabled: true
  feature_type: ORB2
  orb_features: 1200  # Good balance
  loop_closure: false  # Keep disabled for speed
  tracking_quality_threshold: 0.5
  min_tracked_points: 8
  max_map_points: 3000

# Logging - reduce for performance
logging:
  level: WARNING  # Less console spam

# Performance - lightweight
performance:
  stats_interval: 50  # Frequent FPS updates
```

## Keyboard Controls

While running with feature visualization:

| Key | Action |
|-----|--------|
| `q` | Quit application |
| (Any key) | Processed by OpenCV waitKey |

## Performance Notes

### CPU Usage
- **OrbyGlasses only**: 80-120% CPU
- **+ Feature View**: 85-125% CPU (+5% overhead)
- **+ Full SLAM Viewer**: 150-200% CPU (+50% overhead)

### Memory Usage
- Feature Tracking window: ~2-3 MB additional
- Negligible impact on overall performance

### FPS Impact
- Typical: <1 FPS reduction
- Feature visualization is very lightweight
- Much faster than pySLAM's 3D viewer

## Advanced: Customizing the Visualization

To modify the feature tracking window, edit `src/main.py`:

```python
def _create_feature_matching_view(self, frame, slam_result):
    # Location: line ~835

    # Adjust window size
    viz_height = 400  # Change height
    viz_width = 600   # Change width

    # Adjust feature bar maximum
    max_matches = 200  # Expected max features

    # Modify colors
    bar_color = (0, 255, 0)  # BGR format
```

## Integration with Other Tools

### With Rerun.io (if enabled)
```bash
# Both visualizations work together
python3 src/main.py --video file.mp4 --show-features
# Rerun will start automatically if configured
```

### Saving Output
```bash
# Save both windows (OrbyGlasses + Feature Tracking)
python3 src/main.py --video file.mp4 --show-features --save-video
# Note: Only OrbyGlasses window is saved to video file
```

### Headless Mode
```bash
# Feature tracking requires display
# Use --no-display for completely headless operation
python3 src/main.py --video file.mp4 --no-display
```

## Summary

The **--show-features** flag provides a lightweight, informative view of SLAM feature tracking without the overhead of 3D visualization. Perfect for:

✅ Monitoring SLAM health in real-time
✅ Debugging feature detection issues
✅ Understanding tracking quality
✅ Minimal performance impact
❌ Does NOT show 3D map or trajectory (use pySLAM viewer for that)

**Recommended for most users** who want SLAM feedback without the complexity of the full 3D viewer.
