#!/bin/bash

# OrbyGlasses Fast Mode - Optimized for 20-30 FPS

echo "Starting OrbyGlasses in FAST MODE..."
echo "Target: 20-30 FPS with core features"
echo ""

# Check virtual environment
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment not found"
    echo "Run ./setup.sh first"
    exit 1
fi

# Activate
source venv/bin/activate

# Check Ollama
if ! pgrep -x "ollama" > /dev/null; then
    echo "Starting Ollama..."
    ollama serve > /dev/null 2>&1 &
    sleep 1
fi

# Run with fast config
echo "Using optimized config for speed..."
python3 src/main.py --config config/config_fast.yaml

deactivate
