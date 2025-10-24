# Performance Fixes

## Problem: Low FPS (9 FPS)

### Quick Fix: Use Fast Config

```bash
# Use the optimized config
python3 src/main.py --config config/config_fast.yaml
```

This config has:
- Lower resolution (320x240 instead of 640x480) = **4x faster**
- Fewer SLAM features (500 instead of 2000) = **2x faster**
- Skip more depth frames = **2x faster**
- Disabled heavy features (VLM, occupancy grid, etc.)

**Expected Result: 20-30 FPS**

---

## Manual Config Changes

Edit `config/config.yaml`:

### 1. Lower Camera Resolution (Biggest Impact)
```yaml
camera:
  width: 320   # Was 640 (4x faster)
  height: 240  # Was 480
```

### 2. Skip More Depth Frames
```yaml
performance:
  depth_skip_frames: 3  # Was 1 (process every 4th frame)
  max_detections: 5     # Was 8 (track fewer objects)
```

### 3. Simplify SLAM
```yaml
slam:
  orb_features: 500     # Was 2000 (4x fewer features)
  visualize: false      # Disable SLAM window
  loop_closure: false   # Disable slow feature
  bundle_adjustment: false
```

### 4. Disable Heavy Features
```yaml
models:
  llm:
    vlm_enabled: false  # Disable vision-language model

conversation:
  enabled: false  # Disable voice commands

trajectory_prediction:
  enabled: false

occupancy_grid_3d:
  enabled: false
```

---

## SLAM Overflow Fix

The SLAM overflow warning is now fixed:
- Added depth value validation (0.1m - 10m)
- Added scale clamping (0.01 - 0.5)
- Added movement magnitude limits (max 1m per frame)
- Added position bounds (-100m to +100m)

---

## FPS Comparison

| Config | Resolution | Features | Expected FPS |
|--------|------------|----------|--------------|
| **Original** | 640x480 | All | 9-12 FPS |
| **Fast** | 320x240 | Core only | 20-30 FPS |
| **Minimum** | 160x120 | Detection only | 40+ FPS |

---

## Test Performance

```bash
# Test with fast config
python3 src/main.py --config config/config_fast.yaml

# Or temporarily lower resolution
python3 -c "
from core.utils import ConfigManager
config = ConfigManager('config/config.yaml')
config.set('camera.width', 320)
config.set('camera.height', 240)
config.save()
"
```

---

## What Slows Things Down

1. **High Resolution** (640x480) - Most impact
2. **SLAM with many features** (2000 ORB points)
3. **Vision-Language Model** (Moondream inference)
4. **Depth estimation every frame** (should skip frames)
5. **Occupancy grid updates** (3D voxel processing)

---

## Recommended Settings for Speed

```yaml
# Fastest possible while maintaining functionality
camera:
  width: 320
  height: 240

performance:
  depth_skip_frames: 3
  max_detections: 5

slam:
  enabled: true  # Keep SLAM
  orb_features: 500
  visualize: false

models:
  llm:
    vlm_enabled: false
```

**This should give you 20-25 FPS consistently.**
