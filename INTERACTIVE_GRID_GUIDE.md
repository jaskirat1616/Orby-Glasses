# Interactive 3D Occupancy Grid - Quick Start Guide

## Overview

The 3D Occupancy Grid now features a **fully interactive visualization** with mouse and keyboard controls, making it easy to explore and inspect the mapped environment.

## Features

✅ **Mouse Wheel Zoom** - Scroll to zoom in/out smoothly
✅ **Keyboard Pan** - Arrow keys to move view around
✅ **3D Rotation** - Rotate view on multiple axes
✅ **Large Visible Voxels** - Voxels are 4+ pixels for easy visibility
✅ **800x800 Display** - Large window for better viewing
✅ **Real-time Updates** - All controls update instantly
✅ **Camera Position** - Shows your current location in the map

## How to Use

### 1. Enable the Feature

Edit `config/config.yaml`:

```yaml
occupancy_grid_3d:
  enabled: true
  visualize: true
  resolution: 0.1  # Smaller = more detail, larger voxels = better visibility
```

### 2. Run OrbyGlasses

```bash
python3 src/main.py
```

You'll see a new window titled **"3D Occupancy Grid"** appear.

### 3. Interactive Controls

#### Mouse Controls
- **Scroll Up/Down** - Zoom in/out

#### Keyboard Controls
- **Arrow Keys** - Pan the view
  - ⬆️ Up: Move view up
  - ⬇️ Down: Move view down
  - ⬅️ Left: Move view left
  - ➡️ Right: Move view right

- **Rotation**
  - **Q/E** - Rotate around Z-axis (spin left/right)
  - **W/S** - Tilt view (rotate around X-axis)

- **Zoom**
  - **+/=** - Zoom in (alternative to mouse wheel)
  - **-/_** - Zoom out

- **Reset**
  - **R** - Reset view to default (30° tilt, 45° rotation, 20x zoom)

- **Quit**
  - **Q** - Quit OrbyGlasses (when main window is focused)
  - **ESC** - Quit test demo

## Visual Guide

### What You'll See

```
┌─────────────────────────────────────────────┐
│ 3D Occupancy Grid                          │
│ Occupied: 245                              │
│ Free: 1,023                                │
│ Resolution: 10cm                           │
│ Zoom: 20.0x                                │
│                                            │
│         ┌──┐ ┌──┐                         │
│         │██│ │██│  ← Red voxels (occupied)│
│    YOU→ ●    ┌──┐                         │
│         │██│ │██│                         │
│         └──┘ └──┘                         │
│                                            │
│ Mouse Wheel: Zoom In/Out                   │
│ Arrow Keys: Pan View                       │
│ R: Reset   Q/E: Rotate Z   W/S: Rotate X  │
└─────────────────────────────────────────────┘
```

### Color Coding

- 🔴 **Red voxels** - Occupied space (obstacles detected)
- 🟢 **Green voxels** - Free space (confirmed empty)
- ⬜ **White borders** - Voxel boundaries for clarity
- 🔵 **Cyan circle "YOU"** - Your current camera position

### Voxel Sizes

The voxel size adapts to zoom level:
- **Default zoom (20x)**: ~2-6 pixels per voxel
- **Zoomed in (100x)**: ~10-20 pixels per voxel
- **Zoomed out (5x)**: ~1-2 pixels per voxel

Voxels are drawn as **rectangles with borders** for maximum visibility.

## Testing Without Running Full System

Use the included test script to explore controls:

```bash
python3 test_interactive_grid.py
```

This creates a sample environment with:
- Walls on 3 sides
- Random obstacles in the middle
- Your camera position at the center

**Perfect for learning the controls!**

## Tips for Best Experience

### For Better Visibility

1. **Increase voxel size** in config:
   ```yaml
   resolution: 0.2  # 20cm voxels (easier to see)
   ```

2. **Zoom in** using mouse wheel or `+` key

3. **Rotate the view** to see from different angles:
   - Top-down: Press `S` several times
   - Isometric: Press `R` to reset
   - Side view: Press `W` several times

### For Better Performance

1. **Lower resolution** (larger voxels):
   ```yaml
   resolution: 0.15  # Fewer voxels to draw
   ```

2. **Reduce update frequency**:
   ```yaml
   update_interval: 1.0  # Update every second
   ```

3. **Increase subsampling**:
   ```yaml
   subsample_step: 16  # Process fewer pixels
   ```

## Typical Workflow

1. **Start OrbyGlasses** - Grid window appears
2. **Move camera around** - Map builds automatically
3. **Zoom in** with mouse wheel to inspect details
4. **Pan with arrows** to explore different areas
5. **Rotate with Q/E/W/S** to view from different angles
6. **Reset with R** if you get lost

## Advanced Features

### Show Both 2D and 3D Views

Enable the 2D top-down slice alongside 3D view:

```yaml
show_2d_slice: true
```

This opens a second window showing:
- 2D occupancy map at head height (1.5m)
- Useful for path planning
- Color-coded like the 3D view

### Adjust View Defaults

The view starts with:
- X-rotation: 30° (tilt)
- Z-rotation: 45° (isometric angle)
- Zoom: 20x
- Position: Centered

These reset when you press **R**.

## Troubleshooting

### Can't See Any Voxels

**Cause**: Map hasn't been built yet
**Solution**: Move the camera around for 10-20 seconds to build the map

### Voxels Too Small

**Cause**: Resolution too fine or zoom too low
**Solution**:
- Increase `resolution` in config (0.1 → 0.2)
- Zoom in with mouse wheel
- Press `+` several times

### View Is Upside Down or Strange

**Cause**: Accidentally rotated too much
**Solution**: Press **R** to reset view to defaults

### Controls Not Responding

**Cause**: Window not focused
**Solution**: Click on the "3D Occupancy Grid" window title bar

### Low Frame Rate

**Cause**: Too many voxels being rendered
**Solution**:
- Increase `resolution` (larger voxels = fewer to draw)
- Increase `subsample_step` for faster updates
- Close other windows

## Examples

### Example 1: Inspect a Corner

1. Move camera to build map
2. Press **E** 3-4 times to rotate view
3. Zoom in with mouse wheel
4. Use arrow keys to center the corner
5. Press **W** to tilt down for better angle

### Example 2: Top-Down View

1. Press **S** repeatedly until view is nearly flat
2. Zoom in for detail
3. Pan with arrows to explore
4. Press **R** when done to reset

### Example 3: Find Yourself

1. Look for cyan circle labeled "YOU"
2. If lost, press **R** to reset view
3. YOU marker is always at your camera position
4. Zoom in to see it more clearly

## Configuration Reference

Full configuration options:

```yaml
occupancy_grid_3d:
  enabled: true                # Turn on/off
  visualize: true              # Show window
  show_2d_slice: false         # Additional 2D window
  resolution: 0.1              # Voxel size (meters)
  grid_size: [20, 20, 3]       # Map bounds (meters)
  update_interval: 0.5         # Update frequency (seconds)
  subsample_step: 8            # Performance tuning
```

## Keyboard Cheat Sheet

| Key | Action |
|-----|--------|
| **Mouse Wheel** | Zoom in/out |
| **↑/↓/←/→** | Pan view |
| **Q** | Rotate left (Z-axis) |
| **E** | Rotate right (Z-axis) |
| **W** | Tilt up (X-axis) |
| **S** | Tilt down (X-axis) |
| **+** | Zoom in |
| **-** | Zoom out |
| **R** | Reset view |

## Summary

The interactive 3D occupancy grid gives you:

✅ Full control over view with mouse + keyboard
✅ Large, visible voxels you can actually see
✅ Real-time updates as you explore
✅ Professional visualization with clear labels
✅ Easy to learn, intuitive controls

**Try it now:** `python3 test_interactive_grid.py`

Enjoy exploring your 3D environment! 🚀
