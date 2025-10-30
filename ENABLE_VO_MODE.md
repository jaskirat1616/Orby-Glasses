# How to Enable Visual Odometry Mode in OrbyGlasses

## Quick Enable VO Mode

Edit `config/config.yaml`:

```yaml
# Option 1: VO only (disable SLAM)
slam:
  enabled: false

visual_odometry:
  enabled: true
  use_pyslam_vo: true
  use_rerun: true

# Option 2: Both SLAM and VO together
slam:
  enabled: true
  use_pyslam: true
  use_rerun: true

visual_odometry:
  enabled: true
  use_pyslam_vo: true
  use_rerun: true
```

Then run:
```bash
./run_orby.sh
```

## Current Status

The isinstance error is **FIXED** ✅

### What's Working Now:
- ✅ **SLAM Mode** - Full pySLAM SLAM with all windows
- ✅ **MockG2O** - Properly handles isinstance checks
- ✅ **Depth disabled** - No unnecessary depth window
- ✅ **All pySLAM windows** - Viewer3D, SlamPlotDrawer, Rerun

### What Needs Testing:
- ⏳ **VO Mode** - Need to verify with live camera

## VO Mode Windows (like main_vo.py)

When VO is enabled, you should see:
1. **pySLAM VO - Camera** - Feature tracking visualization
2. **pySLAM VO - Trajectory** - 2D top-down path view
3. **Rerun Viewer** - 3D trajectory (browser)
4. **OrbyGlasses** - Main camera with YOLO detections

## Testing Instructions

### Test SLAM Mode (Currently Working)
```bash
# config.yaml should have:
# slam.enabled: true
# slam.use_pyslam: true

./run_orby.sh
```

**Expected**: No isinstance errors, all pySLAM windows appear

### Test VO Mode
```bash
# Edit config.yaml:
# slam.enabled: false
# visual_odometry.enabled: true
# visual_odometry.use_pyslam_vo: true

./run_orby.sh
```

**Expected**: VO windows appear like main_vo.py

## Key Differences from pySLAM Examples

### main_slam.py vs OrbyGlasses SLAM
| Feature | main_slam.py | OrbyGlasses |
|---------|-------------|-------------|
| Input | Pre-recorded video (KITTI) | Live camera |
| Ground truth | Required (from dataset) | None (not needed) |
| Calibration | From settings file | From config.yaml |
| Viewers | Viewer3D + Rerun | Same + OrbyGlasses UI |

### main_vo.py vs OrbyGlasses VO
| Feature | main_vo.py | OrbyGlasses |
|---------|-----------|-------------|
| Input | Pre-recorded video | Live camera |
| Ground truth | Optional (for error plots) | None |
| VO Class | VisualOdometryEducational | Same (via pyslam_vo_integration) |
| Viewers | Trajectory + Rerun | Same + OrbyGlasses UI |

## Configuration Reference

### SLAM Config
```yaml
slam:
  enabled: true
  use_pyslam: true
  feature_type: ORB  # or SIFT, SUPERPOINT
  loop_closure: true
  use_rerun: true
  orb_features: 2000
```

### VO Config
```yaml
visual_odometry:
  enabled: true
  use_pyslam_vo: true
  feature_type: ORB
  num_features: 2000
  use_rerun: true
  trajectory_length: 1000
  visualization_scale: 50
```

## Troubleshooting

### If you see isinstance errors:
- ✅ **FIXED** - MockG2O now handles isinstance properly
- The fix is in `src/main.py` lines 23-41

### If VO windows don't appear:
- Check `visual_odometry.use_pyslam_vo: true`
- Check logs for VO initialization messages
- Verify pySLAM venv is activated

### If SLAM windows don't appear:
- Check `slam.use_pyslam: true`
- Check `slam.use_rerun: true`
- Verify Viewer3D and SlamPlotDrawer are initialized

## Next Steps

1. Test SLAM mode - should work now with no errors ✅
2. Test VO mode - enable in config and test
3. Test both together - both modes simultaneously
4. Report any remaining issues with full error logs

## Files Modified for isinstance Fix

1. **src/main.py** - Fixed MockG2O to support isinstance
2. **src/navigation/pyslam_live.py** - Added error logging
3. **config/config.yaml** - Added use_rerun flag

All changes committed and pushed to GitHub ✅
