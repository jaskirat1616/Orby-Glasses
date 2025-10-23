#!/bin/bash

# OrbyGlasses Launcher

echo "================================================"
echo "  OrbyGlasses - Navigation Assistant"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source venv/bin/activate

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if Ollama is running
echo -e "${GREEN}Checking Ollama service...${NC}"
if ! pgrep -x "ollama" > /dev/null; then
    echo -e "${YELLOW}Warning: Ollama is not running${NC}"
    echo "Starting Ollama in background..."
    ollama serve &
    sleep 3
fi

# Check if required models are available
echo -e "${GREEN}Checking AI models...${NC}"
if ! ollama list | grep -q "gemma3:4b"; then
    echo -e "${YELLOW}Downloading gemma3:4b model...${NC}"
    ollama pull gemma3:4b
fi

if ! ollama list | grep -q "moondream"; then
    echo -e "${YELLOW}Downloading moondream model...${NC}"
    ollama pull moondream
fi

# Check for camera
echo -e "${GREEN}Checking camera access...${NC}"
if ! python3 -c "import cv2; cap = cv2.VideoCapture(0); ret = cap.isOpened(); cap.release(); exit(0 if ret else 1)" 2>/dev/null; then
    echo -e "${YELLOW}Warning: Camera not accessible at index 0${NC}"
    echo "You may need to configure the camera source in config/production.yaml"
fi

# Create required directories
mkdir -p data/logs
mkdir -p data/maps
mkdir -p models/yolo
mkdir -p models/depth

# System health check
echo -e "${GREEN}Running system health check...${NC}"
python3 << EOF
import sys
try:
    import cv2
    import torch
    import numpy as np
    from ultralytics import YOLO
    print("✓ Core dependencies OK")

    # Check MPS availability
    if torch.backends.mps.is_available():
        print("✓ Apple Silicon (MPS) acceleration available")
    else:
        print("⚠ MPS not available, will use CPU")

    sys.exit(0)
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo -e "${RED}System health check failed${NC}"
    echo "Please run: pip install -r requirements.txt"
    exit 1
fi

echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Starting OrbyGlasses Navigation System${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Safety Features Enabled:"
echo "  ✓ Distance calibration"
echo "  ✓ Collision warnings"
echo "  ✓ System health monitoring"
echo "  ✓ Intelligent audio prioritization"
echo "  ✓ SLAM indoor navigation"
echo "  ✓ 3D spatial mapping"
echo ""
echo "Press 'q' to emergency stop"
echo ""

# Run with best config
python3 src/main.py --config config/best.yaml

# Cleanup
echo ""
echo -e "${GREEN}OrbyGlasses stopped${NC}"
deactivate
