# OrbyGlasses: SLAM and Voxel Mapping System

## Overview
This project implements a SLAM (Simultaneous Localization and Mapping) system and voxel mapping system for the OrbyGlasses navigation system. The system works exclusively with camera input (no IMU) and provides positioning and 3D environmental mapping.

## Key Improvements

### 1. SLAM System (`slam_system.py`)

#### Feature Detection:
- **Increased ORB features**: Now uses 3000 features (up from 2000) for better tracking
- **More precise scale factor**: 1.1 (down from 1.2) for finer feature matching
- **More levels**: 16 levels (up from 8) for better scale invariance
- **Lower FAST threshold**: 10 (down from 20) to detect more feature points
- **Stricter RANSAC threshold**: 0.5 (down from 1.0) for more accurate pose estimation

#### Temporal Consistency:
- Temporal consistency checks prevent drift and jumping
- More aggressive pose smoothing (0.8 vs 0.7) reduces jitter
- Better motion model with velocity estimation
- Improved keyframe insertion logic based on displacement and tracking quality

#### Tracking Improvements:
- Better scale estimation from depth maps
- More robust essential matrix computation
- Rotation smoothing using axis-angle representation
- Improved outlier rejection in feature matching

#### Visualization Support:
- Added `visualize_tracking()` method compatible with main application
- Feature visualization with quality indicators
- Real-time status overlay on camera feed

### 2. Voxel Mapping System (`voxel_map.py`)

#### Multi-Resolution Mapping:
- **Near resolution**: 0.05m for high detail close to camera
- **Far resolution**: 0.2m for efficient mapping at distance
- Adaptive resolution based on distance from camera

#### Uncertainty Modeling:
- Sensor-specific uncertainty calculations
- Pixel location-based uncertainty (corners less accurate)
- Depth uncertainty propagation through ray casting
- Probabilistic modeling

#### Temporal Filtering:
- Observation history tracking for each voxel
- Weighted averaging of recent observations
- Temporal decay for dynamic scene handling
- Confirmation thresholds

#### Performance:
- Sparse voxel grid storage for memory efficiency
- Optimized 3D Bresenham algorithm for ray casting
- Configurable update intervals
- Efficient neighbor voxel processing

#### Visualization and Interaction Support:
- **`visualize_3d_interactive()`**: Full 3D visualization with camera position tracking
- **`visualize_2d_slice()`**: 2D slice visualization at specified height
- **`handle_mouse_events()`**: Interactive mouse controls for zoom/pan/rotate
- **`update_view_controls()`**: Keyboard controls for navigation
- **`handle_mouse_wheel()`**: Zoom functionality via mouse wheel

### 3. Rich Terminal Output

#### UI with Rich:
- Table layouts for organized information
- Color-coded status indicators
- Real-time performance metrics
- Structured data presentation
- Visual panels for different information categories

#### Information Hierarchy:
- Primary metrics in main table
- Performance stats in separate panel
- System summary with quick status
- Configurable refresh rate to prevent flickering

## System Features

### Error Handling:
- Graceful degradation when tracking fails
- Motion prediction when visual features are insufficient
- Logging for debugging
- Safe fallback mechanisms

### Performance:
- Configurable tracking vs performance trade-offs
- Memory-efficient sparse data structures
- Algorithms for real-time processing
- Adjustable update intervals

### Configurable Parameters:
- All parameters accessible through config file
- Easy tuning for different environments
- Separation of tracking and performance settings
- Runtime parameter adjustments

## Camera-Only Operation (No IMU)

The system works using only visual features by:

1. **Feature Tracking**: Robust feature detection and matching
2. **Temporal Consistency**: Checks to prevent unrealistic pose jumps
3. **Depth Integration**: Using depth maps for scale estimation
4. **Motion Prediction**: Predictive models when visual tracking is weak
5. **Multi-Resolution Mapping**: Adaptive resolution

## Configuration

The system is configured through `config/config.yaml` with specific sections for:
- SLAM parameters (`slam` section)
- Voxel mapping parameters (`occupancy_grid_3d` section)
- Performance parameters
- Feature detection parameters

## Integration

The new systems replace the old ones:
- `SLAMSystem` replaces `MonocularSLAM` in main.py
- `VoxelMap` replaces `OccupancyGrid3D` in main.py
- Rich terminal output updates the existing `_display_terminal_info` function

## Compatibility

All visualization and interaction methods required by the main application have been implemented:
- `SLAMSystem` includes `visualize_tracking()` method
- `VoxelMap` includes:
  - `visualize_3d_interactive()` for 3D visualization
  - `visualize_2d_slice()` for 2D slices
  - `handle_mouse_events()` for mouse interaction
  - `update_view_controls()` for keyboard navigation
  - `handle_mouse_wheel()` for zooming

## Validation

The system has been validated through:
- Unit testing of individual components
- Integration testing with the main application
- Performance benchmarking

This system provides improved tracking and mapping compared to the original implementation while maintaining real-time performance and full compatibility with the existing application architecture.