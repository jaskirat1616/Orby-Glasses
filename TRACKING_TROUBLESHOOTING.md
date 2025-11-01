# SLAM Tracking Lost - Troubleshooting Guide

## 🔍 Problem Analysis

Your SLAM is in **RELOCALIZE** state because tracking was lost.

### What Happened
```
detector: ORB , #features: 133  ← Should be ~3000!
Relocalizer: num_matches: 0-1   ← Needs minimum 15
Relocalizer: failed
```

**Root Cause:** Only 133 features detected (should be 2500-3000)

---

## 🎯 Quick Fixes

### 1. **Environment Issues** (Most Common)
**Symptoms:** Low features detected

**Causes:**
- ❌ Pointing at blank wall
- ❌ Poor lighting
- ❌ Uniform surfaces (white walls, ceiling)
- ❌ Motion blur (moving too fast)

**Solutions:**
✅ **Point camera at textured surfaces**
- Books, posters, furniture
- Corners and edges
- Varied surfaces

✅ **Improve lighting**
- Turn on more lights
- Avoid direct sunlight (causes glare)
- Ensure even lighting

✅ **Move camera slowly**
- Especially during startup (first 20-30 frames)
- Smooth, steady movements
- No sudden rotations

---

### 2. **Camera Issues**
**Check camera quality:**

```bash
# Test camera 1
python3 -c "
import cv2
cap = cv2.VideoCapture(1)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print(f'✅ Camera 1 working')
        print(f'   Resolution: {frame.shape[1]}x{frame.shape[0]}')
        print(f'   Average brightness: {frame.mean():.1f} (target: 100-150)')
    else:
        print('❌ Cannot read from camera 1')
else:
    print('❌ Camera 1 not available - trying camera 0')
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print('✅ Camera 0 works - consider using that')
cap.release()
"
```

**If camera 1 doesn't work:**
```bash
# Switch back to camera 0
sed -i.bak 's/source: 1/source: 0/' config/config.yaml
sed -i.bak 's/VideoCapture(1)/VideoCapture(0)/' src/navigation/pyslam_live.py
```

---

### 3. **Recovery Tips**

When you see "Relocalization failed":

**Option A: Restart with better conditions**
1. Stop the program (press 'q')
2. Point camera at textured area (books, desk, etc.)
3. Ensure good lighting
4. Restart: `./run_orby.sh`

**Option B: Help system relocalize**
1. Keep moving slowly
2. Return to an area you've mapped before
3. Point at recognizable features
4. System will auto-recover when it finds matches

---

## 🔧 Advanced Fixes

### Increase Feature Detection Sensitivity

If you're in a low-texture environment, increase features:

**Edit config/config.yaml:**
```yaml
slam:
  orb_features: 4000  # ↑ from 3000 (more features)
```

**Pros:** Better tracking in challenging environments
**Cons:** Slightly higher CPU usage

---

### Adjust ORB Detection Parameters

For better feature detection in poor conditions:

**Create config override:**
```bash
# Add to config/config.yaml
slam:
  orb_features: 4000
  orb_scale_factor: 1.15  # More pyramid levels
  orb_edge_threshold: 15  # Lower = more features (default: 19)
```

---

### Disable Auto-exposure/Auto-focus

Camera auto-adjustments can cause tracking loss:

**The camera optimizations in pyslam_live.py already do this:**
```python
self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)         # Disable autofocus
```

If still issues, try:
```python
# In pyslam_live.py, add:
self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Manual exposure value
self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 120)  # Increase brightness
```

---

## 📊 Monitoring Feature Detection

### Good Feature Detection
```
detector: ORB , #features: 2500-3000
# matched points: >100
Tracking state: OK
```

### Poor Feature Detection (Problems!)
```
detector: ORB , #features: <500  ← Problem!
# matched points: <50             ← Problem!
Tracking state: RELOCALIZE        ← Lost!
```

---

## 🎯 Best Practices

### During Startup (Critical!)
**First 20-30 frames are crucial for initialization:**

1. ✅ **Point at textured area** (books, desk, posters)
2. ✅ **Stay still for 2-3 seconds**
3. ✅ **Move VERY slowly** at first
4. ✅ **Watch console for feature count**

### During Normal Operation
1. ✅ Move smoothly and steadily
2. ✅ Avoid blank walls and ceilings
3. ✅ Keep camera pointed at textured areas
4. ✅ If features drop below 500, slow down

### When Tracking is Lost
1. ⚠️ **Don't panic** - system will try to relocalize
2. ⚠️ Move slowly back to mapped area
3. ⚠️ Point at recognizable features
4. ⚠️ If it doesn't recover in 10-20 frames, restart

---

## 🔍 Debugging Commands

### Check Feature Detection in Real-time
```bash
# Monitor feature count
./run_orby.sh 2>&1 | grep "#features:"
```

### Check Tracking State
```bash
# Monitor tracking state
./run_orby.sh 2>&1 | grep "state:"
```

### Full Debug Log
```bash
# Save full log for analysis
./run_orby.sh 2>&1 | tee slam_debug.log
```

---

## 🎬 Step-by-Step Recovery

If SLAM keeps losing tracking:

### Step 1: Test Environment
```bash
# Quick feature test
python3 << 'EOF'
import cv2
cap = cv2.VideoCapture(1)
ret, frame = cap.read()
if ret:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=3000)
    kp = orb.detect(gray, None)
    print(f"Features detected: {len(kp)}")
    if len(kp) < 500:
        print("⚠️  LOW FEATURES - Point camera at textured area!")
    else:
        print("✅ Good features - environment OK")
cap.release()
EOF
```

### Step 2: Improve Conditions
- Turn on lights
- Point at books/desk/posters
- Avoid blank walls

### Step 3: Restart SLAM
```bash
./run_orby.sh
```

### Step 4: Move Carefully
- Start still for 2-3 seconds
- Move VERY slowly
- Watch feature count in console

---

## 🎛️ Parameter Tuning

### For Low-Light Environments
```yaml
slam:
  orb_features: 4000          # More features
  orb_edge_threshold: 10      # More sensitive
```

### For Fast Movement
```yaml
slam:
  orb_features: 5000          # Even more features
  max_frames_between_keyframes: 15  # More keyframes
```

### For Blank Environments
```yaml
slam:
  orb_features: 5000          # Maximum features
  orb_scale_factor: 1.1       # More scale levels
```

---

## ✅ Success Indicators

You have good tracking when you see:

```
detector: ORB , #features: 2500-3000 ✅
# matched points: 100-500          ✅
state: OK                           ✅
```

---

## 🆘 Emergency: Switch to Camera 0

If camera 1 is the problem:

```bash
# Quick switch to camera 0
./switch_mode.sh off  # Stop SLAM
sed -i '' 's/source: 1/source: 0/' config/config.yaml
./run_orby.sh
```

---

## 📞 Quick Reference

| Issue | Solution |
|-------|----------|
| Low features (<500) | Point at textured area, improve lighting |
| Relocalization failed | Move slowly back to mapped area |
| Tracking lost | Restart with better conditions |
| Camera 1 not working | Switch to camera 0 in config |
| Motion blur | Move slower, increase exposure |

---

**TL;DR:**
1. ✅ Point at **textured surfaces** (not blank walls)
2. ✅ **Good lighting** (not too dark, not too bright)
3. ✅ **Move slowly**, especially at startup
4. ✅ If tracking lost, return to mapped area slowly

**Most common cause: Pointing at blank wall or poor lighting!** 🎯
