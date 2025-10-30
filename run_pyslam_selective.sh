#!/bin/bash
# Run pySLAM main_slam.py with selective windows

echo "🚀 Starting pySLAM main_slam.py with selective windows..."
echo ""
echo "Available windows:"
echo "1. Camera - Main camera view with feature trails"
echo "2. Depth - Depth estimation window" 
echo "3. 3D Viewer - Interactive 3D point cloud and trajectory"
echo "4. Plots - 2D metrics and statistics plots"
echo ""

# Navigate to pySLAM directory
cd /Users/jaskiratsingh/Desktop/OrbyGlasses/third_party/pyslam

# Activate pySLAM virtual environment
source ~/.python/venvs/pyslam/bin/activate

echo "Starting pySLAM with all windows..."
echo "Press 'q' in any window to quit"
echo ""

# Run main_slam.py (shows all windows by default)
python main_slam.py

echo "pySLAM finished."
