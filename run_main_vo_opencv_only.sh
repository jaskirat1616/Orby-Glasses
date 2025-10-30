#!/bin/bash
# Run pySLAM main_vo.py with OpenCV windows only (no Rerun)

echo "🚀 Starting pySLAM main_vo.py with OpenCV windows only..."

# Navigate to pySLAM directory
cd /Users/jaskiratsingh/Desktop/OrbyGlasses/third_party/pyslam

# Activate pySLAM virtual environment
source ~/.python/venvs/pyslam/bin/activate

# Copy config to pySLAM directory
cp ../../config_pyslam_live.yaml config_live.yaml

# Create a temporary modified main_vo.py to force OpenCV windows
MODIFIED_VO_SCRIPT="main_vo_opencv_only.py"
cp main_vo.py "$MODIFIED_VO_SCRIPT"

# Force OpenCV windows by disabling Rerun
sed -i '' 's/kUseRerun = True/kUseRerun = False/' "$MODIFIED_VO_SCRIPT"

echo "Starting pySLAM Visual Odometry with OpenCV windows only..."
python "$MODIFIED_VO_SCRIPT" --config_path=config_live.yaml

echo "pySLAM Visual Odometry finished."

# Clean up the modified script
rm "$MODIFIED_VO_SCRIPT"
