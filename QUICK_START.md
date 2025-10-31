# OrbyGlasses Quick Start Guide

## 🚀 Running OrbyGlasses

### Simple Method (Current Config)
```bash
./run_orby.sh
```

This runs with whatever mode is currently configured in `config/config.yaml`.

---

## 🔄 Switching Between Modes

Use the mode switcher script:

```bash
./switch_mode.sh [MODE]
```

### Available Modes

#### 1. **SLAM Mode** (Full SLAM with 3D Mapping)
```bash
./switch_mode.sh slam
./run_orby.sh
```

**What you get:**
- ✅ Full 3D point cloud mapping
- ✅ Bundle adjustment
- ✅ Map persistence
- ✅ Pangolin 3D viewer window
- ✅ Trajectory plots
- ✅ Feature tracking visualization
- ✅ Rerun.io visualization (optional)
- ⚠️  External depth estimation disabled (pySLAM uses monocular depth)
- ⚠️  Loop closure disabled (requires pyobindex2)

**Best for:** Indoor navigation, map building, accurate positioning

---

#### 2. **Visual Odometry Mode** (Fast, No Mapping)
```bash
./switch_mode.sh vo
./run_orby.sh
```

**What you get:**
- ✅ Fast motion tracking
- ✅ Real-time trajectory estimation
- ✅ 2D trajectory visualization
- ✅ Camera view with feature matches
- ✅ Lower CPU/memory usage

**Best for:** Quick testing, real-time motion tracking, limited resources

---

#### 3. **Both SLAM + VO** (Comparison Mode)
```bash
./switch_mode.sh both
./run_orby.sh
```

**What you get:**
- ✅ All SLAM windows
- ✅ All VO windows
- ⚠️  Resource intensive!

**Best for:** Research, comparison, debugging

---

#### 4. **Detection Only** (No SLAM/VO)
```bash
./switch_mode.sh off
./run_orby.sh
```

**What you get:**
- ✅ Object detection only
- ✅ Audio guidance
- ✅ Lowest resource usage

**Best for:** Testing object detection, running on low-end hardware

---

## 📊 Current Configuration

Check what's currently enabled:
```bash
grep -A 5 "^slam:" config/config.yaml
grep -A 5 "^visual_odometry:" config/config.yaml
```

---

## 🎨 What Windows to Expect

### SLAM Mode Windows:
1. **OrbyGlasses** - Main camera view with object detection overlay
2. **Pangolin 3D Viewer** - Interactive 3D point cloud and camera trajectory
3. **pySLAM Trajectory Plots** - 2D trajectory and error plots
4. **pySLAM Camera** - Camera view with feature tracking

### VO Mode Windows:
1. **OrbyGlasses** - Main camera view with object detection overlay
2. **pySLAM VO - Camera** - Camera view with feature matches
3. **pySLAM VO - Trajectory** - 2D trajectory accumulation

---

## 🛠️ Manual Configuration

Alternatively, edit `config/config.yaml` directly:

### For SLAM:
```yaml
slam:
  enabled: true
  feature_type: ORB
  orb_features: 2000
  loop_closure: true
  use_pyslam: true
  use_rerun: true
```

### For VO:
```yaml
visual_odometry:
  enabled: true
  feature_type: ORB
  num_features: 3000
  use_rerun: true
```

---

## 🐛 Troubleshooting

### "pySLAM not available"
```bash
cd third_party/pyslam
./install_all.sh
```

### Windows not showing
- Make sure you're not in headless mode
- Check that Pangolin and OpenCV are installed
- Try running with `./run_orby.sh --display` (though display is on by default)

### Too many windows
```bash
./switch_mode.sh slam    # Use SLAM only
# or
./switch_mode.sh vo      # Use VO only
```

### Performance issues
- Reduce features: Edit `config/config.yaml` → `orb_features: 1000`
- Use VO instead of SLAM: `./switch_mode.sh vo`
- Disable Rerun: Set `use_rerun: false` in config

---

## 💡 Tips

1. **Start with VO** if you're testing - it's faster and simpler
2. **Use SLAM** for actual navigation and mapping
3. **Both mode** is great for comparing accuracy
4. **Press 'q'** to quit the application
5. **Move the camera slowly** at startup to help initialization

---

## 📝 Examples

### Quick Test Run (VO):
```bash
./switch_mode.sh vo && ./run_orby.sh
```

### Full Navigation (SLAM):
```bash
./switch_mode.sh slam && ./run_orby.sh
```

### Debugging (Both):
```bash
./switch_mode.sh both && ./run_orby.sh
```
