#!/bin/bash
# Run pySLAM main_vo.py with live camera and windows

echo "🚀 Starting pySLAM main_vo.py with live camera and windows..."

# Navigate to pySLAM directory
cd /Users/jaskiratsingh/Desktop/OrbyGlasses/third_party/pyslam

# Activate pySLAM virtual environment
source ~/.python/venvs/pyslam/bin/activate

# Copy config to pySLAM directory
cp ../../config_pyslam_live.yaml config_live.yaml

# Run main_vo.py (not headless - shows windows)
echo "Starting pySLAM Visual Odometry with live camera and windows..."
python main_vo.py --config_path=config_live.yaml

echo "pySLAM Visual Odometry finished."
