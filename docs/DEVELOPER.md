# Developer Documentation

This document provides technical details for developers working on OrbyGlasses.

## Table of Contents

1. [Architecture](#architecture)
2. [pySLAM Integration](#pyslam-integration)
3. [SLAM Tuning & Relocalization](#slam-tuning--relocalization)
4. [Dense Reconstruction](#dense-reconstruction)
5. [Configuration](#configuration)
6. [Performance Optimization](#performance-optimization)
7. [Testing](#testing)

---

## Architecture

### System Overview

OrbyGlasses uses a modular architecture:

```
src/
├── main.py              # Main entry point
├── core/                # Core functionality
│   ├── detection.py     # Object detection (YOLO)
│   ├── depth_anything_v2.py  # Depth estimation
│   ├── stair_detection.py     # Stair/curb detection
│   └── utils.py         # Utilities (audio, config, etc.)
├── features/            # Advanced features
│   ├── conversation.py  # Voice control
│   └── trajectory_prediction.py
├── navigation/          # Navigation & SLAM
│   ├── pyslam_live.py   # pySLAM integration
│   └── indoor_navigation.py  # Path planning
└── visualization/      # Visualization (optional)
```

### Data Flow

```
Camera Frame
    ↓
Object Detection (YOLOv11n) → 80 COCO classes
    ↓
Depth Estimation (Depth Anything V2) → Distance measurements
    ↓
Stair Detection → Safety warnings
    ↓
SLAM (pySLAM) → Position tracking
    ↓
Audio Guidance (TTS) → User warnings
```

---

## pySLAM Integration

### Overview

OrbyGlasses uses the [pySLAM](https://github.com/luigifreda/pyslam) library for real-time SLAM (Simultaneous Localization and Mapping). The integration is in `src/navigation/pyslam_live.py`.

### Key Features

- **Real-time SLAM tracking** with ORB features
- **Loop closure** for relocalization
- **Map persistence** (save/load maps)
- **Crash recovery** (auto-disables loop closure on crashes)
- **3D visualization** (Pangolin viewer)

### Live Camera Support

OrbyGlasses includes modifications for live camera operation:

**File**: `third_party/pyslam/pyslam/slam/tracking.py`

**Change**: Removed strict consecutive frame assertion for live camera support. In live scenarios, frames may be dropped due to processing delays.

**Original Code**:
```python
assert f_ref.img_id == f_cur.img_id - 1  # FAILS in live scenarios
```

**Modified Code**:
```python
# Note: Removed strict consecutive frame assertion for live camera support
if f_ref.img_id != f_cur.img_id - 1:
    Printer.orange(f"Warning: Non-consecutive frames detected: {f_ref.img_id} -> {f_cur.img_id}")
```

### Usage

```python
from navigation.pyslam_live import LivePySLAM

# Initialize SLAM
slam = LivePySLAM(config)

# Process frame
result = slam.process_frame(frame)
# Returns: {
#     'pose': 4x4 transformation matrix,
#     'position': [x, y, z],
#     'tracking_quality': 0.0-1.0,
#     'num_map_points': int,
#     ...
# }

# Save map
slam.save_map("my_house")

# Load map
slam.load_map("data/maps/my_house_20250102_143052.pkl")
```

---

## SLAM Tuning & Relocalization

### Overview

When SLAM loses tracking, relocalization tries to find where the camera is by matching current features against the map.

### Auto-Tuning

OrbyGlasses automatically applies aggressive relocalization tuning for real-world success:

- **20 inlier threshold** (down from 50) - More realistic for monocular SLAM
- **Relaxed PnP solver** - Uses chi-square threshold of 7.815 (98% confidence)
- **Larger search windows** - 15px coarse, 5px fine
- **Lenient matching** - Ratio test 0.85-0.95

**Configuration** (in `pyslam_live.py`):
```python
Parameters.kRelocalizationDoPoseOpt2NumInliers = 20  # Reduced from 50
Parameters.kRelocalizationFeatureMatchRatioTest = 0.85  # Relaxed from 0.75
Parameters.kRelocalizationMaxReprojectionDistanceMapSearchCoarse = 15  # Increased from 10
```

### Relocalization Failures

**Common causes**:
- Not enough features detected (<2000)
- Matches aren't geometrically consistent
- Camera moved too far from original viewpoint
- Poor lighting conditions

**Debugging**:
1. Check feature count in logs: `detector: ORB , #features: 1313`
2. If below 2000, increase `slam.orb_features` in config
3. Check tracking quality: `tracking_quality > 0.7` is good

### Crash Recovery

OrbyGlasses includes automatic crash recovery:

- Detects loop closure crashes (MLPnPsolver, Bus error)
- Auto-disables loop closure after 3 crashes
- Gracefully falls back to visual odometry
- System stays alive instead of crashing

**Configuration**:
```yaml
slam:
  loop_closure: true  # Will auto-disable on crashes
```

---

## Dense Reconstruction

### Overview

OrbyGlasses supports dense 3D reconstruction from saved SLAM maps. This generates detailed point clouds or meshes.

### Two Approaches

#### 1. Post-Processing (Recommended)

**Best for**: Monocular cameras (OrbyGlasses default)

**Steps**:
1. Run SLAM and save map:
   ```bash
   ./run_orby.sh --mode full_slam
   # Press 's' in Pangolin viewer to save map
   ```

2. Generate dense reconstruction:
   ```bash
   ./dense_reconstruction.sh
   ```

**Output**: Dense point cloud or mesh in `data/maps/`

#### 2. Real-Time (Experimental)

**Best for**: Stereo cameras, RGBD sensors

**Limitations**:
- Requires depth data (monocular needs depth estimation)
- Depth estimation is slow (1-2 seconds/frame)
- Not suitable for real-time use with monocular cameras

**Configuration**:
```yaml
slam:
  dense_reconstruction:
    enabled: true  # Enable for real-time dense mapping
    type: TSDF     # TSDF or GAUSSIAN_SPLATTING
    extract_mesh: true
    voxel_length: 0.04  # 4cm voxels
    depth_trunc: 4.0    # Max depth (meters)
```

### TSDF Volume Integration

**How it works**:
1. Loads saved SLAM map with keyframes
2. For each keyframe:
   - Projects 3D points onto camera view
   - Estimates depth (if not available)
   - Integrates into TSDF volume
3. Extracts mesh or point cloud

**Output formats**:
- `.ply` - Point cloud or mesh
- `.obj` - Mesh (alternative format)

---

## Configuration

### Production vs Development

**Production Config** (`config/config_production.yaml`):
- All critical features enabled
- Optimized for blind user navigation
- Safety features enabled (stair detection, depth estimation)
- SLAM enabled for position tracking
- Voice control enabled

**Development Config** (`config/config.yaml`):
- Many features disabled for performance
- Optimized for development/debugging
- Lower resource usage

### Key Settings

**Critical for Blind Users**:
```yaml
models:
  depth:
    enabled: true  # Essential for distance measurement

stair_detection:
  enabled: true  # Critical safety feature

slam:
  enabled: true  # Essential for position tracking

conversation:
  enabled: true  # Essential for hands-free operation
```

**Performance Tuning**:
```yaml
performance:
  danger_audio_interval: 0.3  # Faster danger warnings
  depth_skip_frames: 1  # Process depth every 2nd frame

slam:
  orb_features: 2000  # Higher for better tracking
  loop_closure: true  # Enable for relocalization
```

---

## Performance Optimization

### Bottlenecks

1. **Depth Estimation**: 30-50ms per frame
   - Solution: Skip frames (`depth_skip_frames: 1`)
   - Or: Use faster depth model (Depth Anything V2 Small)

2. **SLAM Tracking**: 80-120ms per frame
   - Solution: Reduce `orb_features` (but affects accuracy)
   - Or: Disable loop closure (but loses relocalization)

3. **Audio TTS**: 1500-2000ms latency
   - Solution: Use faster TTS engine
   - Or: Use beep sounds for urgent warnings

### Optimization Tips

1. **For Development**: Use `config/config.yaml` (fast mode)
2. **For Production**: Use `config/config_production.yaml` (full features)
3. **For Speed**: Disable visualization (`--no-display`)
4. **For Accuracy**: Increase `orb_features` and enable loop closure

### Memory Management

SLAM can use a lot of memory over time:

```yaml
slam:
  max_trajectory_length: 1000  # Keep last 1000 poses
  max_map_points: 5000  # Limit map points
  cleanup_interval: 500  # Cleanup every N frames
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_detection.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Coverage

Current test files:
- `tests/test_detection.py` - Object detection tests
- `tests/test_slam.py` - SLAM integration tests
- `tests/test_integration.py` - End-to-end tests

### Adding Tests

When adding new features:
1. Create test file in `tests/`
2. Add unit tests for core functionality
3. Add integration tests for end-to-end flows
4. Target 70%+ coverage

---

## Troubleshooting

### SLAM Tracking Issues

**Symptoms**: Tracking quality < 0.5, frequent tracking loss

**Solutions**:
1. Increase `slam.orb_features` to 2000+
2. Ensure good lighting conditions
3. Point camera at textured surfaces
4. Check camera calibration (`camera.fx`, `camera.fy`)

### Audio Latency

**Symptoms**: Warnings come too late (1-2 seconds)

**Solutions**:
1. Reduce `performance.audio_update_interval`
2. Use faster TTS engine
3. Add beep sounds for urgent warnings

### Memory Leaks

**Symptoms**: System slows down over time

**Solutions**:
1. Reduce `slam.max_trajectory_length`
2. Reduce `slam.max_map_points`
3. Decrease `slam.cleanup_interval`

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on:
- Code style
- Pull request process
- Testing requirements
- Documentation standards

---

## References

- [pySLAM GitHub](https://github.com/luigifreda/pyslam)
- [ORB-SLAM2 Paper](https://ieeexplore.ieee.org/document/7946260)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [YOLOv11](https://github.com/ultralytics/ultralytics)

