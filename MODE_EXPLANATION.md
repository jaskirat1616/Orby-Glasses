# Visualization Modes - How They Work

## Mode System Logic

Each mode automatically configures SLAM/VO based on what makes sense:

### Features Mode (Default - RECOMMENDED)
```bash
./run_orby.sh --mode features
```

**What it does:**
- ✅ Enables **SLAM** (for feature matching data)
- ✅ Shows feature matching window ONLY
- ❌ No 3D map viewer (hidden in this mode)
- ❌ No trajectory plot (hidden in this mode)

**Why SLAM?**
- SLAM provides the feature matching data we need
- But we only show the feature matching window
- Map viewer and other SLAM windows are hidden
- Clean 2-window interface
- **40-60 FPS**

**Windows:** OrbyGlasses + Feature Matching (that's it!)

---

### Basic Mode (Fastest)
```bash
./run_orby.sh --mode basic
```

**What it does:**
- ✅ Uses whatever is in config (default: SLAM enabled)
- ❌ No visualizations
- 🎯 Maximum speed

**Why use this?**
- Production navigation
- When you don't need to see feature matching
- Want full SLAM tracking but no viz overhead

**Windows:** OrbyGlasses only
**FPS:** 60-80

---

### Full SLAM Mode (Complete)
```bash
./run_orby.sh --mode full_slam
```

**What it does:**
- ✅ Enables **Full SLAM** with map building
- ✅ Shows ALL windows (3D viewer, trajectory, features)
- ✅ Loop closure enabled
- ✅ Global map optimization

**Why use this?**
- Need global map
- Want loop closure
- Complete SLAM debugging
- Research/development

**Windows:** OrbyGlasses + Feature Matching + 3D Viewer + Trajectory
**FPS:** 20-30

---

### VO Mode (Lightweight Tracking)
```bash
./run_orby.sh --mode vo
```

**What it does:**
- ✅ Enables Visual Odometry only
- ✅ Shows VO tracking window
- ❌ No SLAM, no map building
- ❌ No feature matching visualization

**Why use this?**
- Want position tracking without map
- Lower memory than SLAM
- Don't need feature viz

**Windows:** OrbyGlasses + VO Tracking
**FPS:** 50-70

---

## Comparison Table

| Mode | Backend | Map Building | Windows Shown | FPS | Best For |
|------|---------|--------------|---------------|-----|----------|
| **features** | **SLAM** | **Yes** | **2 only** | **40-60** | **Daily use** |
| basic | SLAM | Yes | 1 | 60-80 | Production |
| full_slam | SLAM | Yes | 4+ | 20-30 | Debugging |
| vo | VO | No | 2 | 50-70 | Tracking only |

## Why Features Mode Uses SLAM

**Question:** Why use SLAM in features mode?

**Answer:**
1. **Feature matching data comes from SLAM tracking**
   - SLAM tracks features between current and reference frames
   - This is what we visualize with the green lines
   - Need SLAM's tracking engine to get this data

2. **But we hide other SLAM windows**
   - No 3D map viewer (Pangolin)
   - No trajectory plot
   - Only show: OrbyGlasses + Feature Matching

3. **Clean 2-window interface**
   - User sees: Main view + Feature matching
   - That's it - no clutter!

4. **User Goal**
   - Features mode = "show me feature matching"
   - Get the visualization without the complexity

## What if I Want Full SLAM + Features?

Use `full_slam` mode:
```bash
./run_orby.sh --mode full_slam
```

You get:
- ✅ Full SLAM with map building
- ✅ Feature matching window
- ✅ 3D map viewer
- ✅ Everything

Trade-off: Slower (20-30 FPS)

## Backend Configuration by Mode

### Automatic Configuration

| Mode | `slam.enabled` | `visual_odometry.enabled` |
|------|----------------|---------------------------|
| features | False | True (auto) |
| basic | True (from config) | False (from config) |
| full_slam | True (auto) | False (auto) |
| vo | False (auto) | True (auto) |

### Manual Override

If you want to force specific backend:

1. Edit `config/config.yaml`:
```yaml
slam:
  enabled: true  # or false
visual_odometry:
  enabled: false  # or true
```

2. Use `basic` mode (doesn't override):
```bash
./run_orby.sh --mode basic
```

## Examples

### Example 1: Feature Matching with VO (Recommended)
```bash
./run_orby.sh --mode features
```
- Shows: OrbyGlasses + Feature Matching
- Uses: Visual Odometry
- FPS: 50-70
- Perfect for seeing feature tracking!

### Example 2: Navigation Only (Fastest)
```bash
./run_orby.sh --mode basic
```
- Shows: OrbyGlasses only
- Uses: SLAM (from config)
- FPS: 60-80
- No visualization overhead

### Example 3: Complete SLAM Debugging
```bash
./run_orby.sh --mode full_slam
```
- Shows: Everything
- Uses: Full SLAM
- FPS: 20-30
- See map building in real-time

## Summary

**Features mode uses VO** because:
- ✅ Faster (50-70 FPS vs 40-60)
- ✅ Lighter (no map building)
- ✅ Sufficient for feature visualization
- ✅ Same beautiful feature matching output

**If you need full SLAM**, use `full_slam` mode which gives you everything at the cost of FPS.

**Recommended for most users:**
```bash
./run_orby.sh --mode features  # VO + feature matching, 50-70 FPS
```
