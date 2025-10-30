#!/bin/bash
# Run pySLAM main_slam.py with proper windows

echo "🚀 Starting pySLAM main_slam.py with windows..."

# Navigate to pySLAM directory
cd /Users/jaskiratsingh/Desktop/OrbyGlasses/third_party/pyslam

# Activate pySLAM virtual environment
source ~/.python/venvs/pyslam/bin/activate

# Copy config to pySLAM directory
cp ../../config_pyslam_live.yaml config_live.yaml

# Run main_slam.py (not headless - shows windows)
echo "Starting pySLAM with live camera and windows..."
python main_slam.py --config_path=config_live.yaml

echo "pySLAM finished."
