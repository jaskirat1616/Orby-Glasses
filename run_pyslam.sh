#!/bin/bash
# Run OrbyGlasses with pySLAM properly activated

echo "🚀 Starting OrbyGlasses with pySLAM..."

# Activate main virtual environment
echo "Activating main virtual environment..."
source venv/bin/activate

# Activate pySLAM environment
echo "Activating pySLAM environment..."
cd third_party/pyslam
source pyenv-activate.sh
cd ../..

# Run the main application
echo "Starting OrbyGlasses with pySLAM..."
python src/main.py "$@"