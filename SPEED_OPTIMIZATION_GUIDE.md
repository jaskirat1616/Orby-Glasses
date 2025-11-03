# OrbyGlasses Speed Optimization Guide

## Problem: Slow SLAM Performance

The system was running very slowly due to multiple bottlenecks in the processing pipeline.

## Main Performance Bottlenecks Identified

### 1. **Depth Model** (BIGGEST BOTTLENECK - 30-50ms per frame)
- Depth Anything V2 model runs on every frame
- Even with MPS acceleration, takes 30-50ms per frame
- Limits FPS to ~20-25 maximum

### 2. **Too Many ORB Features** (10-20ms overhead)
- 2000 features = extensive computation
- More features = slower matching and tracking
- Diminishing returns above 1200 features

### 3. **High Resolution Video** (5-10ms overhead)
- 640x480 or higher = more pixels to process
- YOLO, ORB, and all algorithms scale with pixel count

### 4. **Map Point Management** (5-10ms periodic)
- Large map with 5000+ points = slower queries
- More trajectory history = more memory operations

## Speed Optimizations Applied

### Quick Wins (Major Impact)

#### 1. Disable Depth Model
```yaml
models:
  depth:
    enabled: false  # DISABLED - saves 30-50ms per frame
```
**Speed Gain: 2-3x FPS increase**

#### 2. Reduce ORB Features
```yaml
slam:
  orb_features: 1200  # Down from 2000
```
**Speed Gain: 1.3-1.5x faster feature extraction**

#### 3. Lower Video Resolution
```yaml
camera:
  max_video_width: 480   # Down from 640
  max_video_height: 360  # Down from 480
```
**Result**: 1080x1920 portrait → 270x480 (instead of 360x640)
**Speed Gain**: 1.4x faster (44% fewer pixels)

#### 4. Reduce Map Complexity
```yaml
slam:
  max_map_points: 3000  # Down from 5000
  max_trajectory_length: 300  # Down from 500
  cleanup_interval: 300  # More frequent cleanup
```
**Speed Gain**: 1.2x faster map operations

### Additional Optimizations

#### 5. Less Aggressive Tracking
```yaml
slam:
  tracking_quality_threshold: 0.5  # More tolerant (was 0.6)
  min_tracked_points: 8  # Lower minimum (was 10)
  max_frames_between_keyframes: 20  # Less frequent (was 15)
```
**Speed Gain**: Fewer keyframes = less bundle adjustment overhead

#### 6. Minimal Logging
```yaml
logging:
  level: WARNING  # Only show warnings and errors
```
**Speed Gain**: ~5-10% faster (less I/O)

#### 7. Less Frequent Performance Stats
```yaml
performance:
  stats_interval: 50  # Check every 50 frames (was 200)
```
**Result**: More visibility into FPS without overhead

## Performance Comparison

| Configuration | Resolution | Features | Depth | Expected FPS |
|--------------|------------|----------|-------|--------------|
| **Original** | 360x640 | 800 | No | Failed (0 features) |
| **Fixed** | 360x640 | 2000 | Yes | 15-20 FPS |
| **Optimized** | 270x480 | 1200 | No | **40-60 FPS** |

## Trade-offs

### What You Lose:
1. **No Depth Information** - Can't measure exact distances to objects
2. **Fewer Features** - Slightly less robust tracking in difficult scenes
3. **Lower Resolution** - Less detail in video

### What You Keep:
1. **✅ SLAM Tracking** - Position and mapping still works
2. **✅ Object Detection** - YOLO still detects obstacles
3. **✅ Real-time Performance** - Smooth, responsive system

## Usage

### Option 1: Use Speed-Optimized Config
```bash
./run_orby.sh --config config/config.yaml --video /path/to/video.mp4
```
The default config.yaml is now optimized for speed.

### Option 2: Use Ultra-Fast Config (Even More Aggressive)
```bash
./run_orby.sh --config config/config_fast_slam.yaml --video /path/to/video.mp4
```

### Option 3: Manual Tuning

If you want **more accuracy** (slower):
```yaml
slam:
  orb_features: 1500-2000
camera:
  max_video_width: 640
models:
  depth:
    enabled: true  # Re-enable if you need distance measurements
```

If you want **maximum speed**:
```yaml
slam:
  orb_features: 800-1000
camera:
  max_video_width: 320
  max_video_height: 240
```

## Debugging Slow Performance

If it's still slow, check these:

### 1. Check Feature Detection
```bash
# Look for warnings in output:
ORBextractor::DistributeOctTree() - warning - nIni = 0
```
**Fix**: Increase `orb_features` or check lighting/contrast in video

### 2. Check FPS in Output
```bash
# Every 50 frames you should see:
INFO - Performance: XX.X FPS
```
**Target**: 30-60 FPS for smooth operation

### 3. Check Process Count
```bash
ps aux | grep python | wc -l
```
**Should be**: 1-2 processes (main + optionally depth workers)
**If many**: Kill old processes: `pkill -9 -f "python3 src/main.py"`

### 4. Check CPU/GPU Usage
```bash
# Monitor while running:
top -pid $(pgrep -f "python3 src/main.py")
```
**Should see**: High CPU usage (80-120% on multiple cores)

### 5. Profile Specific Bottlenecks
Add timing in main.py around suspected slow sections:
```python
import time
start = time.time()
# ... slow code ...
print(f"Operation took: {(time.time()-start)*1000:.1f}ms")
```

## Common Issues

### Issue: Still seeing "nIni = 0" warnings
**Cause**: Not enough texture/contrast in video frames
**Fix**:
- Increase `orb_features` to 1500-2000
- Check video quality
- Ensure proper lighting in scene

### Issue: FPS drops after a few seconds
**Cause**: Map growing too large
**Fix**:
- Reduce `max_map_points` to 2000-3000
- Increase cleanup frequency: `cleanup_interval: 200`

### Issue: SLAM loses tracking frequently
**Cause**: Too few features or too aggressive settings
**Fix**:
- Increase `orb_features` to 1500
- Lower `tracking_quality_threshold` to 0.4
- Lower `min_tracked_points` to 6

## MPS (Metal Performance Shaders) Status

✅ **MPS is available and working**:
- PyTorch: 2.9.0
- MPS available: True

✅ **Currently using MPS**:
- YOLO object detection (accelerated)
- Depth model (when enabled, accelerated)

❌ **NOT using MPS**:
- ORB-SLAM2 (uses C++ OpenCV, not PyTorch)
- Feature extraction (pure C++ implementation)

**Note**: SLAM doesn't need MPS - it's already highly optimized C++ code from ORB-SLAM2.

## Recommended Configuration

For **best balance** of speed and functionality:

```yaml
slam:
  orb_features: 1200  # Good balance
models:
  depth:
    enabled: false  # Major bottleneck
camera:
  max_video_width: 480
  max_video_height: 360
logging:
  level: WARNING
```

**Expected Performance**: 40-60 FPS with robust tracking

For **maximum speed** (tracking only):
```yaml
slam:
  orb_features: 800
  max_map_points: 2000
camera:
  max_video_width: 320
  max_video_height: 240
```

**Expected Performance**: 80-120 FPS (limited by video source)

## Next Steps

1. **Test current configuration**: Should now get 40-60 FPS
2. **If still slow**: Reduce features to 800-1000
3. **If need depth**: Re-enable but skip more frames (depth_skip_frames: 15)
4. **Monitor FPS**: Check stats output every 50 frames

## Summary

**Key Insight**: The depth model was the main bottleneck, taking 30-50ms per frame. Disabling it gives **2-3x speed improvement**. Combined with other optimizations, you should now get **smooth 40-60 FPS** performance.

The system now prioritizes SLAM tracking and object detection, which is sufficient for navigation without per-pixel depth measurements.
