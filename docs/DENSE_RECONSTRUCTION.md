# Dense 3D Map Reconstruction

OrbyGlasses supports dense 3D reconstruction from saved SLAM maps. This generates detailed point clouds or meshes from your SLAM sessions.

## Overview

Dense reconstruction takes a sparse SLAM map (feature points and camera poses) and creates a complete 3D model using:
- **TSDF (Truncated Signed Distance Function)**: Volumetric fusion for smooth, accurate reconstructions
- **Gaussian Splatting**: Neural rendering approach (experimental)

## Quick Start

### 1. Run SLAM and Save Map

First, run SLAM to create a map:

```bash
./switch_mode.sh slam
./run_orby.sh
```

While SLAM is running:
- Move the camera to explore the environment
- Ensure good tracking (watch feature matches)
- **Press 's' in the Pangolin 3D viewer to save the map**

The map will be saved to `results/slam_state/`

### 2. Generate Dense Reconstruction

After saving your SLAM map, run dense reconstruction:

```bash
./dense_reconstruction.sh
```

Or specify custom paths:

```bash
./dense_reconstruction.sh path/to/slam_state path/to/output
```

## How It Works

### TSDF Volume Integration

The system:
1. Loads saved SLAM map with keyframes
2. For each keyframe:
   - Projects 3D points onto camera view
   - Integrates depth information into TSDF volume
   - Fuses multiple views for accuracy
3. Extracts final point cloud or mesh
4. Displays in 3D viewer

### Configuration

Edit `config/config.yaml`:

```yaml
slam:
  dense_reconstruction:
    enabled: false      # Only for post-processing
    type: TSDF          # TSDF or GAUSSIAN_SPLATTING
    extract_mesh: true  # true = mesh, false = point cloud
    voxel_length: 0.015 # meters - smaller = more detail
    depth_trunc: 4.0    # meters - max depth (indoor)
```

## Parameters

### Voxel Length
- **Default**: 0.015m (1.5cm)
- **Smaller**: More detail, more memory
- **Larger**: Faster, less detail
- **Indoor**: 0.01-0.02m
- **Outdoor**: 0.05-0.1m

### Depth Truncation
- **Default**: 4.0m (indoor)
- **Purpose**: Ignore depth beyond this distance
- **Indoor**: 3-5m
- **Outdoor**: 8-15m

### SDF Truncation
- **Default**: 0.04m
- **Purpose**: Band width around surfaces
- **Rule**: Usually 2-3× voxel length

## 3D Viewer Controls

Once reconstruction starts:

- **Mouse drag**: Rotate view
- **Scroll wheel**: Zoom in/out
- **Right drag**: Pan view
- **'s'**: Save dense map to output folder
- **'q'**: Quit viewer

## Output Files

Dense reconstruction saves:

```
results/dense_reconstruction/
├── point_cloud.ply     # Point cloud (if extract_mesh: false)
├── mesh.ply            # Mesh (if extract_mesh: true)
├── slam_map.json       # Original sparse map
└── metadata.json       # Reconstruction parameters
```

## Tips for Best Results

### 1. Good SLAM Map
- Collect 50+ keyframes
- Cover the environment from multiple angles
- Ensure stable tracking throughout
- Avoid rapid camera motion

### 2. Dense Coverage
- Move slowly and smoothly
- Overlap views significantly (60%+)
- Maintain consistent distance to surfaces
- Re-visit areas from different viewpoints

### 3. Lighting
- Well-lit environment (for ORB features)
- Avoid direct sunlight/shadows
- Consistent lighting throughout session

### 4. Texture
- Rich visual texture helps SLAM
- Avoid blank walls/floors
- Add temporary markers if needed

## Troubleshooting

### "SLAM map not found"
- Run SLAM first: `./switch_mode.sh slam && ./run_orby.sh`
- Save map by pressing 's' in Pangolin viewer
- Check `results/slam_state/` exists

### Reconstruction is noisy
- Increase voxel_length (faster, smoother)
- Improve SLAM map quality (more keyframes)
- Ensure stable tracking during capture

### Memory issues
- Reduce reconstruction volume
- Increase voxel_length
- Increase depth_trunc (limit volume size)

### Incomplete reconstruction
- Collect more keyframes
- Improve view coverage
- Check tracking quality during SLAM

## Technical Details

### TSDF Integration

The TSDF volume stores signed distance to nearest surface:
- **Negative**: Inside surface
- **Zero**: On surface
- **Positive**: Outside surface

Integration formula:
```
TSDF(x) = truncate(depth(x) - measured_depth, sdf_trunc)
```

Weighted averaging across views:
```
TSDF_new = (TSDF_old * W_old + TSDF_obs * W_obs) / (W_old + W_obs)
```

### Mesh Extraction

Uses marching cubes algorithm:
1. Extract zero-crossing isosurface
2. Generate triangular mesh
3. Optional: Smooth and decimate

## Advanced Usage

### Custom Depth Estimator (Stereo/RGBD)

For stereo cameras, enable depth estimation:

```python
# In dense_reconstruction.sh or custom script
Parameters.kVolumetricIntegrationUseDepthEstimator = True
Parameters.kVolumetricIntegrationDepthEstimatorType = "DEPTH_RAFT_STEREO"
```

### Semantic Reconstruction

Enable semantic mapping during SLAM:

```yaml
slam:
  semantic_mapping:
    enabled: true
```

Reconstructed mesh will include semantic labels.

## Performance

### Typical Processing Time

- **50 keyframes**: ~2-5 minutes
- **100 keyframes**: ~5-10 minutes
- **200 keyframes**: ~10-20 minutes

Depends on:
- Number of keyframes
- Voxel resolution
- Depth estimator (if enabled)
- CPU/GPU performance

### Memory Usage

Approximate VRAM/RAM requirements:

| Voxel Size | Volume Size | Memory  |
|-----------|-------------|---------|
| 0.01m     | 5m × 5m × 3m | ~2-4GB |
| 0.015m    | 5m × 5m × 3m | ~1-2GB |
| 0.02m     | 5m × 5m × 3m | ~500MB-1GB |

## See Also

- [SLAM Guide](SLAM_TRACKING_TROUBLESHOOTING.md)
- [Feature Tracking](NORMAL_ENVIRONMENT_GUIDE.md)
- [Configuration Guide](../config/config.yaml)
