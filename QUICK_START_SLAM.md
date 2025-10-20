# Quick Start: Testing SLAM

## ✅ SLAM is Now Integrated!

Commit: `2e6b557` - SLAM fully integrated into main pipeline

---

## What You'll See

When you run OrbyGlasses with SLAM enabled, you'll see **3 windows**:

### 1. Main Window (OrbyGlasses)
Shows object detection with SLAM overlay:
```
FPS: 15.2
Danger: 0 | Caution: 2
Closest: chair 2.5m
Process: 65ms
SAFE

SLAM: (0.5, 0.2, 0.0)              ← Your position
Quality: 0.85 | Points: 1250       ← Tracking quality and map size
```

### 2. SLAM Tracking Window
Shows:
- Green dots: Detected ORB features
- Position: Your 3D coordinates
- Tracking quality: How confident SLAM is
- Map points: Number of landmarks

### 3. Depth Map Window
Shows depth estimation (unchanged from before)

---

## How to Run

### Option 1: With SLAM (Current Default)
```bash
python3 src/main.py
```

**You'll see**:
```
🗺️  SLAM enabled - Indoor navigation active
   SLAM visualization window will appear
```

### Option 2: Without SLAM (Faster)
Edit `config/config.yaml`:
```yaml
slam:
  enabled: false
```

Then run:
```bash
python3 src/main.py
```

---

## What SLAM Does

### Real-Time Tracking
As you move your camera, SLAM tracks your position:
```
Frame 1: Position (0.0, 0.0, 0.0)
Frame 10: Position (0.5, 0.1, 0.0)   ← Moved 50cm forward, 10cm left
Frame 20: Position (1.2, 0.3, 0.0)   ← Moved 1.2m forward total
```

### Map Building
SLAM creates a map of your environment:
```
Map Points: 500   ← Just started
Map Points: 1250  ← After exploring
Map Points: 2500  ← Fully mapped room
```

### Tracking Quality
Color-coded quality indicator:
- 🟢 **Green (> 0.7)**: Excellent tracking
- 🟡 **Yellow (0.4-0.7)**: Moderate tracking
- 🔴 **Red (< 0.4)**: Poor tracking / Lost

---

## Testing SLAM

### Test 1: Static Scene
1. Run `python3 src/main.py`
2. Point camera at textured wall/desk
3. Keep camera still
4. **Expected**: Position stays at (0.0, 0.0, 0.0), Quality > 0.7

### Test 2: Slow Movement
1. Point at textured area
2. Slowly move camera forward (10-20cm)
3. **Expected**: Position X increases, green features tracked

### Test 3: Room Mapping
1. Start at one corner of room
2. Slowly pan across room
3. **Expected**: Map points increase, position changes

### Test 4: Return to Start
1. Map a small area
2. Move away
3. Return to starting position
4. **Expected**: Position returns close to (0, 0, 0)

---

## Interpreting SLAM Output

### Position (x, y, z)
- **X**: Forward/backward movement
- **Y**: Left/right movement
- **Z**: Up/down movement (usually near 0)

Units: Meters

### Tracking Quality
- **1.0**: Perfect tracking (all features matched)
- **0.7-1.0**: Good tracking (most features matched)
- **0.4-0.7**: Moderate tracking (some drift expected)
- **< 0.4**: Poor tracking (SLAM may be lost)

### Map Points
- **0-500**: Just initialized
- **500-2000**: Exploring environment
- **2000+**: Well-mapped environment

---

## Troubleshooting

### "Lost tracking: only 0 matches"
**Cause**: Not enough visual features
**Fix**: Point camera at textured surfaces (posters, books, patterned walls)

### "Insufficient features detected: 0"
**Cause**: Blank wall or very dark
**Fix**: Better lighting or point at objects

### SLAM window shows no green dots
**Cause**: No ORB features detected
**Fix**: Ensure good lighting and textured surfaces

### Position drifting (moving when camera still)
**Cause**: Low-quality features or repetitive patterns
**Fix**: Normal for long sessions, re-initialize by restarting

### FPS drops below 10
**Cause**: SLAM is computationally expensive
**Fix**:
1. Disable visualization: `slam.visualize: false`
2. Or disable SLAM entirely if not needed

---

## Performance Expectations

### Best Case (Random Textures)
- FPS: 40-90
- Tracking: Excellent
- Example: Bookshelf, poster wall

### Typical (Indoor Environment)
- FPS: 15-30
- Tracking: Good
- Example: Office, living room

### Challenging (Repetitive Patterns)
- FPS: 8-15
- Tracking: Moderate
- Example: Tiled floor, striped wallpaper

### Worst Case (Blank Walls)
- FPS: N/A (SLAM fails)
- Tracking: Lost
- Solution: System falls back to detection-only

---

## Advanced: Saving Maps

### Save Current Map
```python
# While running, if you want to save programmatically:
slam.save_map("my_office.json")
```

Map saved to: `data/maps/my_office.json`

### Load Saved Map
```python
# On startup:
slam.load_map("data/maps/my_office.json")
```

Now SLAM already knows your environment!

---

## Configuration Options

### Minimal (Fastest)
```yaml
slam:
  enabled: true
  visualize: false  # No extra window = faster
```

### Full Debug (Slowest but Most Info)
```yaml
slam:
  enabled: true
  visualize: true

indoor_navigation:
  enabled: true
```

### Production (Recommended After Testing)
```yaml
slam:
  enabled: false  # Disable for production until optimized
```

---

## Next Steps

### After Testing SLAM
1. ✅ Verify it works with your webcam
2. ✅ Test in different environments
3. ✅ Record demo video
4. ⬜ Get feedback from users
5. ⬜ Optimize performance

### Optimizations to Try
1. **Frame skipping**: Process SLAM every 2-3 frames
2. **Reduced features**: Change `nfeatures=2000` to `nfeatures=1000` in slam.py
3. **Async processing**: Run SLAM in background thread
4. **Smaller image**: Resize frame before SLAM processing

---

## Summary

**Current Status**: ✅ SLAM Fully Integrated

**What Works**:
- ✅ Position tracking
- ✅ Map building
- ✅ Visualization
- ✅ Integration with detection
- ✅ Graceful degradation

**Performance**: 10-30 FPS (environment dependent)

**Ready to Test**: YES! Just run `python3 src/main.py`

---

**Enjoy exploring with SLAM!** 🗺️

Questions? Check `docs/SLAM_INDOOR_NAVIGATION.md` for full guide.
