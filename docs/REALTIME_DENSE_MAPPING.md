# Real-time Dense 3D Mapping

OrbyGlasses supports **real-time dense 3D reconstruction** that builds a complete 3D model as you run SLAM, not just sparse feature points.

## ⚠️ Important Limitations

**Real-time dense mapping requires depth data:**

- ✅ **Works with**: Stereo cameras, RGBD sensors
- ❌ **Monocular cameras** (OrbyGlasses default): Requires depth estimation
  - Depth estimation is **very slow** (1-2 seconds/frame)
  - Not suitable for real-time use
  - **Recommended**: Use [post-processing](DENSE_RECONSTRUCTION.md) instead

**For OrbyGlasses with single camera:**
```bash
# ❌ Real-time dense won't work well (frames skipped, no depth)
./switch_mode.sh slam_dense

# ✅ Better: Run SLAM, then post-process for dense reconstruction
./switch_mode.sh slam && ./run_orby.sh  # Capture
./dense_reconstruction.sh                # Generate dense map
```

## Overview

While standard SLAM tracks camera pose and sparse 3D points, dense mapping creates a full volumetric representation:

- **Standard SLAM**: 2000-5000 sparse feature points
- **Dense Mapping**: Millions of voxels forming complete surfaces

## Quick Start

### Enable Real-time Dense Mapping

```bash
# Switch to slam_dense mode
./switch_mode.sh slam_dense

# Run OrbyGlasses
./run_orby.sh
```

You'll see the dense 3D model building in real-time in the Pangolin 3D viewer!

## What Happens

When you enable slam_dense mode:

1. **SLAM runs normally** - Tracks camera and builds sparse map
2. **Volumetric integration** - Each keyframe adds to TSDF volume
3. **Dense visualization** - Point cloud or mesh updates live
4. **Memory accumulation** - Volume grows as you explore

## Performance Impact

Real-time dense mapping is **computationally intensive**:

| Setting | FPS | CPU | Memory | Quality |
|---------|-----|-----|--------|---------|
| SLAM only | 25-30 | ~40% | ~200MB | Sparse |
| SLAM + Dense | 5-15 | ~80-100% | ~1-2GB | Dense |

### Performance Tips

**1. Adjust voxel size** (in config.yaml):
```yaml
dense_reconstruction:
  voxel_length: 0.02  # Larger = faster (default: 0.015)
```

**2. Reduce feature count**:
```yaml
slam:
  orb_features: 3000  # Lower = faster (default: 5000)
```

**3. Disable loop closure**:
```yaml
slam:
  loop_closure: false  # Saves ~15% CPU
```

**4. Point cloud instead of mesh**:
```yaml
dense_reconstruction:
  extract_mesh: false  # Faster than mesh extraction
```

## When to Use Real-time Dense Mapping

### ✅ Good Use Cases

- **Live room scanning** - See the 3D model as you scan
- **Interactive exploration** - Immediate feedback while mapping
- **Research/development** - Test dense reconstruction parameters
- **Short sessions** - 2-5 minutes of mapping

### ❌ Not Recommended For

- **Long sessions** - Memory will fill up (>10 minutes)
- **Low-power devices** - Too CPU/GPU intensive
- **High-quality results** - Post-processing gives better quality
- **Large environments** - Volume size limits apply

## Comparison: Real-time vs Post-processing

| Feature | Real-time | Post-processing |
|---------|-----------|-----------------|
| **Speed** | Slow (5-15 FPS) | Fast (20-30 FPS SLAM) |
| **Memory** | High (1-2GB+) | Low (200MB SLAM) |
| **Quality** | Good | Better |
| **Editing** | No | Yes (reprocess with different settings) |
| **Workflow** | One-step | Two-step (SLAM → Dense) |

## Configuration Options

Edit `config/config.yaml`:

```yaml
slam:
  dense_reconstruction:
    enabled: true  # Enable for real-time dense mapping
    type: TSDF  # or GAUSSIAN_SPLATTING
    extract_mesh: true  # true = mesh, false = point cloud
    voxel_length: 0.015  # meters - resolution
    depth_trunc: 4.0  # meters - max depth
```

### Parameter Guide

**voxel_length** (meters):
- `0.01` - Very high detail, slow, memory intensive
- `0.015` - High detail (default), good balance
- `0.02` - Medium detail, faster
- `0.03` - Low detail, fastest

**depth_trunc** (meters):
- `3.0` - Small rooms
- `4.0` - Normal rooms (default)
- `6.0` - Large rooms
- `10.0` - Outdoor scenes

**extract_mesh**:
- `true` - Generate triangle mesh (slower, looks better)
- `false` - Point cloud only (faster)

**type**:
- `TSDF` - Classic volumetric fusion (recommended)
- `GAUSSIAN_SPLATTING` - Neural rendering (experimental)

## Viewing the Dense Map

The Pangolin 3D viewer shows both:
- **Sparse map** - Green/red feature points, yellow camera trajectory
- **Dense map** - Colored surface point cloud or mesh

### Controls

- **Mouse drag**: Rotate view
- **Scroll**: Zoom
- **Right drag**: Pan
- **'s'**: Save map (includes dense volume)
- **'r'**: Reset view
- **'q'**: Quit

## Saving Dense Maps

When you press 's' in the Pangolin viewer:
- Sparse SLAM map saved to `results/slam_state/`
- Dense volume saved to `results/slam_state/volumetric_*.ply`

You can later reload and re-export with different settings:
```bash
./dense_reconstruction.sh results/slam_state results/my_dense_map
```

## Troubleshooting

### "Out of memory" Error

The TSDF volume has finite size. Either:
- Increase voxel_length (lower resolution)
- Increase depth_trunc limits (smaller volume)
- Map smaller area
- Use post-processing instead

### Very Slow Performance

- Reduce voxel resolution (increase voxel_length)
- Disable mesh extraction (extract_mesh: false)
- Reduce ORB features
- Check CPU/GPU usage (should be close to 100%)

### Dense Map Not Showing

Check logs for:
```
🗺️  Real-time Dense Reconstruction ENABLED
```

If not shown:
```bash
# Verify mode is set
./switch_mode.sh slam_dense

# Check config
grep -A5 "dense_reconstruction:" config/config.yaml
```

### Poor Quality Reconstruction

For better quality, use **post-processing** instead:
1. Run regular SLAM: `./switch_mode.sh slam && ./run_orby.sh`
2. Save map (press 's')
3. Generate dense map: `./dense_reconstruction.sh`

Post-processing allows:
- Higher resolution without real-time constraints
- Multiple passes with different parameters
- Better depth estimation
- No performance impact during SLAM

## Technical Details

### TSDF Volume Integration

Truncated Signed Distance Function (TSDF) stores distance to nearest surface:

1. **Each voxel** stores signed distance + weight
2. **Each keyframe** projects points into volume
3. **Weighted averaging** fuses multiple views
4. **Marching cubes** extracts mesh from zero-crossing

### Memory Requirements

Approximate memory usage:

| Voxel Size | Volume (5m × 5m × 3m) | Memory |
|-----------|----------------------|--------|
| 0.01m     | 375 million voxels   | ~3-4GB |
| 0.015m    | 111 million voxels   | ~1-2GB |
| 0.02m     | 47 million voxels    | ~500MB-1GB |
| 0.03m     | 14 million voxels    | ~200-300MB |

### Performance Bottlenecks

1. **Voxel integration** - Most expensive, scales with keyframe rate
2. **Mesh extraction** - Runs periodically, can cause FPS drops
3. **Visualization** - Drawing millions of points/triangles

## Best Practices

### For Real-time Scanning

1. **Start with low resolution**:
   - voxel_length: 0.02 or 0.03
   - extract_mesh: false

2. **Move slowly**:
   - 30-50cm/second maximum
   - Smooth, steady motion
   - Overlap views 60%+

3. **Good lighting**:
   - Well-lit environment
   - Avoid shadows/reflections
   - Consistent lighting

4. **Limit duration**:
   - 2-5 minute sessions
   - Reset if memory fills
   - Save frequently

### For Best Quality

Use **post-processing** workflow:

1. **Capture with regular SLAM**:
   ```bash
   ./switch_mode.sh slam
   ./run_orby.sh
   ```
   - Faster, more stable
   - Better tracking
   - More keyframes

2. **Save the map** (press 's')

3. **Generate dense reconstruction**:
   ```bash
   ./dense_reconstruction.sh
   ```
   - No real-time constraints
   - Can adjust parameters
   - Better results

## Advanced Usage

### Custom Volume Size

Edit `pyslam/config_parameters.py`:
```python
Parameters.kVolumetricIntegrationDepthTruncIndoor = 6.0  # Larger volume
```

### Depth Estimation (Experimental)

For stereo cameras, estimate depth:
```python
Parameters.kVolumetricIntegrationUseDepthEstimator = True
Parameters.kVolumetricIntegrationDepthEstimatorType = "DEPTH_RAFT_STEREO"
```

Note: Very slow, 1-2 seconds per frame.

## See Also

- [Dense Reconstruction (Post-processing)](DENSE_RECONSTRUCTION.md)
- [SLAM Guide](SLAM_TRACKING_TROUBLESHOOTING.md)
- [Configuration](../config/config.yaml)
