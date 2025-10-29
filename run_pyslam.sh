#!/bin/bash
# Run OrbyGlasses with real pySLAM

echo "🚀 Starting OrbyGlasses with Real pySLAM..."

# Activate main virtual environment
echo "Activating main virtual environment..."
source venv/bin/activate

# Set pySLAM environment variables
export PYSLAM_PATH="/Users/jaskiratsingh/Desktop/OrbyGlasses/third_party/pyslam"
export PYTHONPATH="$PYSLAM_PATH:$PYTHONPATH"

# Run the main application with real pySLAM
echo "Starting OrbyGlasses with Real pySLAM..."
python src/main.py "$@"