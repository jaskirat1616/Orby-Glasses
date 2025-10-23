#!/bin/bash

# OrbyGlasses - Simple Launcher

echo "================================================"
echo "  OrbyGlasses - Navigation for Blind Users"
echo "================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check virtual environment
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Run ./setup.sh first"
    exit 1
fi

# Activate
echo -e "${GREEN}Activating environment...${NC}"
source venv/bin/activate

# Check Ollama
echo -e "${GREEN}Checking Ollama...${NC}"
if ! pgrep -x "ollama" > /dev/null; then
    echo -e "${YELLOW}Starting Ollama...${NC}"
    ollama serve &
    sleep 2
fi

# Check models
if ! ollama list | grep -q "gemma3:4b"; then
    echo -e "${YELLOW}Downloading AI model...${NC}"
    ollama pull gemma3:4b
fi

# Check camera
echo -e "${GREEN}Checking camera...${NC}"
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('✓ Camera OK' if cap.isOpened() else '✗ Camera failed'); cap.release()" 2>/dev/null

# Create directories
mkdir -p data/logs data/maps models/yolo models/depth

# Run
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Starting OrbyGlasses${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "Features:"
echo "  ✓ Object detection"
echo "  ✓ Depth estimation"
echo "  ✓ Safety warnings"
echo "  ✓ Audio guidance"
echo "  ✓ SLAM navigation"
echo ""
echo "Press 'q' to stop"
echo ""

# Run full pipeline with fast config (better accuracy)
python3 src/main.py --config config/fast.yaml

# Cleanup
echo ""
echo -e "${GREEN}OrbyGlasses stopped${NC}"
deactivate
