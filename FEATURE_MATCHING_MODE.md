# Feature Matching Mode - Standalone Usage

## Overview

Feature Matching Mode is a lightweight visualization mode that shows SLAM/VO feature matching without requiring YOLO or other heavy dependencies.

## Usage

### Option 1: Using Main Script (Recommended)

```bash
# With SLAM (default)
python3 src/main.py --mode feature_matching

# With Visual Odometry
python3 src/main.py --mode feature_matching --vo
```

### Option 2: Standalone Script (No Dependencies)

```bash
# With SLAM
python3 src/feature_matching_standalone.py --camera 1 --mode slam

# With Visual Odometry
python3 src/feature_matching_standalone.py --camera 1 --mode vo

# With video file
python3 src/feature_matching_standalone.py --camera path/to/video.mp4 --mode slam
```

## What You'll See

The output matches the reference image (`main-feature-matching.png`):

- **Side-by-side frames**: Reference keyframe (left) and current frame (right)
- **Green circles**: Feature points detected in both frames
- **Colored lines**: Lines connecting matched features (green, red, blue, orange)
- **Feature sizes**: Green circles showing keypoint sizes (like reference image)

## Features

✅ **No YOLO dependency** - Works without object detection
✅ **Minimal dependencies** - Only requires pySLAM and OpenCV
✅ **SLAM and VO support** - Works with both SLAM and Visual Odometry
✅ **Real-time visualization** - Shows feature matching in real-time from webcam
✅ **Video file support** - Can process video files too

## Troubleshooting

### cv2.imshow Error

If you see `AttributeError: module 'cv2' has no attribute 'imshow'`:

1. **Check OpenCV installation**:
   ```bash
   python3 -c "import cv2; print(cv2.__version__)"
   ```

2. **Reinstall OpenCV**:
   ```bash
   pip3 install opencv-python --upgrade
   ```

3. **Use standalone script** (avoids YOLO import issues):
   ```bash
   python3 src/feature_matching_standalone.py --camera 1 --mode slam
   ```

### pySLAM Not Available

If pySLAM is not available, install it:

```bash
cd third_party/pyslam
# Follow installation instructions
```

## Configuration

You can customize camera settings:

```bash
python3 src/feature_matching_standalone.py \
  --camera 1 \
  --mode slam \
  --width 640 \
  --height 480 \
  --fx 525.0 \
  --fy 525.0
```

## Comparison with Other Modes

| Mode | Detection | SLAM | VO | Windows | Use Case |
|------|-----------|------|-----|---------|----------|
| `feature_matching` | ❌ | ✅ | ✅ | 1 | Feature matching only |
| `features` | ✅ | ✅ | ❌ | 2 | Full system + features |
| `vo` | ✅ | ❌ | ✅ | 2 | Visual odometry |
| `basic` | ✅ | ✅ | ❌ | 1 | Navigation only |
| `full_slam` | ✅ | ✅ | ❌ | 4+ | Complete SLAM |

## Performance

- **FPS**: 40-60 FPS (depends on camera and system)
- **CPU**: ~10-15% (much lighter than full system)
- **Memory**: ~50-100 MB (minimal overhead)

