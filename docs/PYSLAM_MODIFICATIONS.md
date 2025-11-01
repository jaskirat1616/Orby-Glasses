# pySLAM Modifications for OrbyGlasses

This document tracks modifications made to the pySLAM third-party library for OrbyGlasses compatibility.

## Overview

OrbyGlasses uses a modified version of pySLAM to support live camera operation and handle real-world scenarios where frames may be dropped or processed at variable rates.

## Modifications

### 1. Non-Consecutive Frame Handling

**File**: `third_party/pyslam/pyslam/slam/tracking.py`
**Line**: ~1342
**Status**: Required for live camera support

#### Original Code:
```python
# get previous frame in map as reference
f_ref = self.map.get_frame(-1)
self.f_ref = f_ref

assert f_ref.img_id == f_cur.img_id - 1  # FAILS in live scenarios

# add current frame f_cur to map
```

#### Modified Code:
```python
# get previous frame in map as reference
f_ref = self.map.get_frame(-1)
self.f_ref = f_ref

# Note: Removed strict consecutive frame assertion for live camera support
# In live scenarios, frames may be dropped due to processing delays
if f_ref.img_id != f_cur.img_id - 1:
    Printer.orange(f"Warning: Non-consecutive frames detected: {f_ref.img_id} -> {f_cur.img_id}")

# add current frame f_cur to map
```

#### Reason:
The original pySLAM code assumes frames are always processed consecutively (frame N+1 immediately follows frame N). This is true for:
- Offline datasets
- Video files
- Perfectly synchronized systems

However, in **live camera scenarios**:
- Processing may be slower than camera FPS
- Frames can be dropped to maintain real-time performance
- System load can cause variable frame processing rates
- Loop closing operations may pause frame processing

The strict assertion `assert f_ref.img_id == f_cur.img_id - 1` would crash the system whenever frames are non-consecutive.

#### Impact:
- **Before**: AssertionError crash when frames are dropped
- **After**: Warning logged, system continues tracking
- **Safety**: The warning alerts users to frame drops without crashing
- **Performance**: No negative impact on tracking quality

### 2. Loop Closing Configuration

**File**: `src/navigation/pyslam_wrapper.py`
**Lines**: 30, 119-122
**Status**: Enhancement for relocalization

#### Addition:
```python
from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs

# Configure loop closing/relocalization
# Using IBOW (Incremental Bag of Words) - builds vocabulary incrementally
loop_detector_config = LoopDetectorConfigs.IBOW

# Initialize SLAM with loop closing enabled
self.slam = Slam(self.camera, feature_tracker_config, loop_detector_config=loop_detector_config)
```

#### Reason:
Enables relocalization when tracking is lost. IBOW was chosen because:
- No pre-built vocabulary files required
- Builds vocabulary incrementally during operation
- Works well with ORB features
- Lower memory footprint than DBOW3

## Applying Modifications

If you reinstall or update pySLAM, reapply these modifications:

### Relocalization Tuning

**File:** `third_party/pyslam/pyslam/slam/relocalizer.py` (line ~189)

```python
# Find this line:
solver.set_ransac_parameters(0.99, 10, 300, 6, 0.5, 5.991)

# Replace with:
solver.set_ransac_parameters(0.99, 6, 300, 4, 0.6, 7.815)
```

### Frame Tracking Fix

**File:** `third_party/pyslam/pyslam/slam/tracking.py` (line ~1342)

### Manual Edit:
1. Open `third_party/pyslam/pyslam/slam/tracking.py`
2. Find line ~1342: `assert f_ref.img_id == f_cur.img_id - 1`
3. Replace with:
```python
# Note: Removed strict consecutive frame assertion for live camera support
# In live scenarios, frames may be dropped due to processing delays
if f_ref.img_id != f_cur.img_id - 1:
    Printer.orange(f"Warning: Non-consecutive frames detected: {f_ref.img_id} -> {f_cur.img_id}")
```

### Using Patch File:
```bash
cd third_party/pyslam
patch -p1 < ../../patches/pyslam-live-camera.patch
```

## Testing Modifications

### Test 1: Live Camera with Frame Drops
```bash
./switch_mode.sh slam
./run_orby.sh
# Move camera rapidly - should see warnings but no crashes
```

Expected output:
```
Warning: Non-consecutive frames detected: 150 -> 152
```

### Test 2: Relocalization
```bash
# Run SLAM, then cover camera lens for 2-3 seconds
# Uncover - system should relocalize without crash
```

Expected: `Relocalization successful` message (may take 5-10 frames)

## Upstream Contribution

These modifications address real-world live camera use cases. Consider contributing:

1. **Frame Drop Tolerance**: Submit PR to pySLAM to add configurable frame drop handling
2. **Live Camera Mode**: Propose a "live mode" flag that relaxes sequential frame requirements

## Version Compatibility

Modifications tested with:
- **pySLAM**: Latest commit as of project date
- **Python**: 3.11+
- **OpenCV**: 4.10.0
- **Platform**: macOS (Apple Silicon), Linux (x86_64)

## Troubleshooting

### "AssertionError" on frame tracking
- pySLAM modification not applied
- Reapply the tracking.py modification above

### "WARNING: you did not set any loop closing method"
- Loop detector config not passed to SLAM constructor
- Check pyslam_wrapper.py or pyslam_live.py initialization

### Excessive "Non-consecutive frames" warnings
- System is dropping many frames (performance issue)
- Reduce processing load:
  - Disable Rerun visualization
  - Reduce orb_features count
  - Close other applications
  - Check CPU/GPU usage

## References

- [pySLAM GitHub](https://github.com/luigifreda/pyslam)
- [SLAM Troubleshooting](SLAM_TRACKING_TROUBLESHOOTING.md)
- [Dense Reconstruction](DENSE_RECONSTRUCTION.md)
