#!/bin/bash
# Run OrbyGlasses with pySLAM (using main environment)

echo "🚀 Starting OrbyGlasses with pySLAM..."

# Activate main virtual environment
echo "Activating main virtual environment..."
source venv/bin/activate

# Run the main application (pySLAM will use simple implementation)
echo "Starting OrbyGlasses with pySLAM..."
python src/main.py "$@"