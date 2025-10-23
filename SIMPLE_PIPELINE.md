# OrbyGlasses - Simple Pipeline

## Overview
Fast, clean, and focused navigation system. No unnecessary complexity.

---

## Quick Start

```bash
# Run the system
./run.sh
```

That's it! The script handles everything.

---

## What It Does

### Core Loop (Simple and Fast)
1. **Capture frame** from camera
2. **Detect objects** with YOLO
3. **Estimate depth** (skips frames for speed)
4. **Check safety** (distance calibration + warnings)
5. **Track position** with SLAM (optional)
6. **Speak guidance** with smart priority
7. **Show results** on screen

### Processing Time
- **Target**: 15-20 FPS
- **Detection**: ~20-30ms
- **Depth**: ~30-40ms (every 3rd frame)
- **SLAM**: ~15-25ms
- **Total**: ~50-70ms per frame

---

## Files

### Main Code
- `src/main_simple.py` - Clean pipeline (300 lines vs 1100 lines)
- `src/detection.py` - Object detection + depth
- `src/safety_system.py` - Distance calibration + warnings
- `src/audio_priority.py` - Smart audio queue

### Configuration
- `config/fast.yaml` - Settings for speed and accuracy
- `run.sh` - Simple launcher script

---

## Key Features

### 1. Object Detection
- YOLOv12 Nano (fast and accurate)
- Confidence threshold: 0.7 (fewer false positives)
- Tracks top 5 most important objects
- Priority classes (people, cars, chairs, doors, etc.)

### 2. Depth Estimation
- Depth Anything V2 model
- Skips frames for speed (processes every 3rd frame)
- Caches last depth map
- Resolution: 320px for fast processing

### 3. Safety System
- Distance calibration using real object sizes
- Safety zones: Immediate (0.4m), Danger (1.0m), Caution (2.0m)
- Clear warnings with action guidance
- Health monitoring (auto safe mode if FPS drops)

### 4. Audio Priority
- Queues messages by priority
- Critical warnings bypass queue
- Avoids repetition
- Minimum 0.5s between messages

### 5. SLAM Navigation
- Tracks position indoors
- Builds maps of rooms
- Saves named locations
- Simplified for speed (no bundle adjustment)

---

## Configuration Explained

### Camera
```yaml
camera:
  source: 0      # 0 = built-in, 1 = external
  width: 640     # Good quality
  height: 480    # Standard aspect
  fps: 30        # Smooth video
```

### Performance
```yaml
performance:
  max_detections: 5         # Track top 5 objects
  depth_skip_frames: 2      # Depth every 3rd frame
  cache_depth_maps: true    # Reuse last depth
```

### Safety
```yaml
safety:
  danger_distance: 0.4      # Immediate stop
  caution_distance: 2.0     # Warning zone
  enable_calibration: true  # Use object sizes
```

### Audio
```yaml
audio:
  min_message_interval: 0.5  # Max 2 messages/second
  max_queue_size: 5          # Keep 5 messages max
```

---

## What's Disabled (For Speed)

### Not Needed for Core Navigation
- ❌ Trajectory prediction (GNN) - experimental
- ❌ 3D occupancy grid - visualization only
- ❌ Point cloud viewer - visualization only
- ❌ Movement visualizer - debugging only
- ❌ Conversation system - adds complexity
- ❌ Scene understanding VLM - slow
- ❌ Echolocation beeps - prefer voice
- ❌ SLAM loop closure - slow, not needed
- ❌ Bundle adjustment - slow, not needed

### Still Enabled
- ✓ Object detection
- ✓ Depth estimation
- ✓ Safety warnings
- ✓ Distance calibration
- ✓ Audio guidance
- ✓ SLAM tracking
- ✓ Indoor navigation
- ✓ Health monitoring

---

## How It Works

### Main Loop
```python
while running:
    # 1. Get frame
    frame = camera.read()

    # 2. Detect objects
    detections = detector.detect(frame)

    # 3. Estimate depth (smart caching)
    if frame_count % 3 == 0:
        depth_map = depth_estimator.estimate(frame)

    # 4. Add depth to detections
    for det in detections:
        det['depth'] = get_depth_at_bbox(depth_map, det['bbox'])

    # 5. Safety checks
    detections, warnings = safety_system.process(detections)

    # 6. SLAM (if enabled)
    slam_result = slam.process_frame(frame, depth_map)

    # 7. Generate audio message
    message = create_audio_message(detections, warnings)
    if message:
        speak(message)

    # 8. Display
    show(annotated_frame)
```

### Safety Processing
```python
# Calibrate using object size
for det in detections:
    size_depth = estimate_from_size(det['label'], det['bbox_height'])
    model_depth = det['depth']

    # Weighted average (60% size, 40% model)
    calibrated = 0.6 * size_depth + 0.4 * model_depth
    det['depth'] = calibrated

# Check safety zones
warnings = []
for det in detections:
    if det['depth'] < 0.4:
        warnings.append({'level': 'IMMEDIATE_DANGER', ...})
    elif det['depth'] < 1.0:
        warnings.append({'level': 'DANGER', ...})
    elif det['depth'] < 2.0:
        warnings.append({'level': 'CAUTION', ...})

return detections, warnings
```

### Audio Priority
```python
# Add messages to queue
if warnings:
    add_message("STOP! Person ahead", priority=10)
elif obstacles_close:
    add_message("Caution: Chair 1.5m ahead", priority=8)
else:
    add_message("Path clear", priority=3)

# Speak highest priority
message = get_next_message()  # Picks highest priority
speak(message)
```

---

## Comparison

### Old main.py
- 1100+ lines
- 10+ optional features
- Complex initialization
- Many conditionals
- Hard to understand
- Slower (12-15 FPS)

### New main_simple.py
- 300 lines
- Core features only
- Simple initialization
- Clear flow
- Easy to read
- Faster (15-20 FPS)

---

## Performance Tips

### For Maximum Speed
1. Disable SLAM: `slam.enabled: false`
2. Increase depth skip: `depth_skip_frames: 4`
3. Lower resolution: `camera.width: 416`
4. Reduce detections: `max_detections: 3`

### For Maximum Accuracy
1. Enable SLAM: `slam.enabled: true`
2. More frequent depth: `depth_skip_frames: 1`
3. Higher resolution: `camera.width: 640`
4. More detections: `max_detections: 7`

### Balanced (Default)
- Resolution: 640x480
- Depth skip: 2 (every 3rd frame)
- Max detections: 5
- SLAM: enabled
- Result: 15-20 FPS

---

## Troubleshooting

### Low FPS (< 10)
- Check: `depth_skip_frames` - increase to 4
- Check: `camera.width` - reduce to 416
- Check: `slam.enabled` - disable if not needed

### Inaccurate Distances
- Check: `safety.focal_length` - calibrate for your camera
- Check: `safety.enable_calibration` - must be true
- Check: Object is in known list (person, car, chair, etc.)

### No Audio
- Check: Speakers/headphones connected
- Check: `audio.tts_engine` is "pyttsx3"
- Check: Not speaking too fast (min_message_interval)

### Camera Not Found
- Check: `camera.source` - try 0 or 1
- Check: Camera permissions in System Settings
- Check: Camera not used by another app

---

## For Developers

### Adding New Object Sizes
Edit `src/safety_system.py`:
```python
OBJECT_HEIGHTS = {
    'person': 1.7,
    'your_object': height_in_meters,
    ...
}
```

### Changing Safety Zones
Edit `config/fast.yaml`:
```yaml
safety:
  danger_distance: 0.4   # Your value
  caution_distance: 2.0  # Your value
```

### Adding Audio Messages
Edit `src/audio_priority.py`:
```python
def create_custom_message(detections):
    return {
        'message': "Your message",
        'priority': 8,  # 0-10
        'category': 'warning'
    }
```

---

## Next Steps

1. **Test accuracy** - Measure actual vs estimated distances
2. **Calibrate camera** - Get exact focal length
3. **Add more objects** - Expand size database
4. **User testing** - Get feedback from blind users
5. **Optimize further** - Profile and improve bottlenecks

---

This simple pipeline is the core of OrbyGlasses. Fast, accurate, and focused on what matters: keeping blind users safe.
