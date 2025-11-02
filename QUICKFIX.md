# Quick Fix Guide

## Issue: OpenCV Import Error

If you see this error:
```
AttributeError: module 'cv2' has no attribute 'imshow'
```

This means OpenCV was installed in headless mode (no GUI support).

### Fix:

```bash
# Uninstall headless OpenCV
pip uninstall opencv-python-headless opencv-python -y

# Install full OpenCV with GUI support
pip install opencv-python

# Verify installation
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

## Issue: PathPlanner Import Error

**Status**: ✅ FIXED in commit 0f0a98d

If you see:
```
ModuleNotFoundError: No module named 'features.prediction'
```

This has been fixed. Run:
```bash
git pull
```

## Testing After Fix

```bash
# Test imports
python3 -c "import sys; sys.path.insert(0, 'src'); from main import OrbyGlasses; print('✅ All imports successful!')"

# Run the system
./run_orby.sh
```

## Other Common Issues

### Camera not found
```bash
# Check available cameras
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera 0:', cap.isOpened()); cap.release()"
python3 -c "import cv2; cap = cv2.VideoCapture(1); print('Camera 1:', cap.isOpened()); cap.release()"

# Update config.yaml with correct camera number
# camera:
#   source: 0  # or 1, 2, etc.
```

### pySLAM not found
```bash
# Install pySLAM
cd third_party/pyslam
./install.sh
cd ../..

# Or run simple mode (no SLAM)
./run_simple.sh
```

### Permission denied on scripts
```bash
chmod +x run_orby.sh
chmod +x run_simple.sh
chmod +x install_pyslam.sh
```

## Quick Start After Fixes

```bash
# 1. Fix OpenCV
pip uninstall opencv-python-headless opencv-python -y
pip install opencv-python

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make scripts executable
chmod +x *.sh

# 4. Run OrbyGlasses
./run_orby.sh
```

## If All Else Fails

Run the simple version (no SLAM, no advanced features):
```bash
./run_simple.sh
```

This bypasses SLAM and advanced features but core navigation still works.
