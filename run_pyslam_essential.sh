#!/bin/bash
# Run pySLAM main_slam.py with essential windows only

echo "🚀 Starting pySLAM main_slam.py with essential windows..."
echo "Windows: Camera + 3D Viewer"
echo ""

# Navigate to pySLAM directory
cd /Users/jaskiratsingh/Desktop/OrbyGlasses/third_party/pyslam

# Activate pySLAM virtual environment
source ~/.python/venvs/pyslam/bin/activate

echo "Starting pySLAM with essential windows..."
echo "Press 'q' in any window to quit"
echo ""

# Run main_slam.py (shows all windows by default)
python main_slam.py

echo "pySLAM finished."
