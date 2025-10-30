#!/bin/bash
# Run pySLAM main_slam.py with proper windows

echo "🚀 Starting pySLAM main_slam.py with windows..."

# Navigate to pySLAM directory
cd /Users/jaskiratsingh/Desktop/OrbyGlasses/third_party/pyslam

# Activate pySLAM virtual environment
source ~/.python/venvs/pyslam/bin/activate

# Run main_slam.py (not headless - shows windows)
echo "Starting pySLAM with windows..."
python main_slam.py

echo "pySLAM finished."
