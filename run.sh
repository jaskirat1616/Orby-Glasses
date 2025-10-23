#!/bin/bash

# OrbyGlasses - Production Launcher

echo "================================================"
echo "  🚀 OrbyGlasses - Robot Navigation System"
echo "================================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check virtual environment
if [ ! -d "venv" ]; then
    echo -e "${RED}✗ Virtual environment not found${NC}"
    echo "Run ./setup.sh first"
    exit 1
fi

# Activate
echo -e "${GREEN}✓ Activating environment...${NC}"
source venv/bin/activate

# Check Ollama (optional for audio)
echo -e "${BLUE}Checking Ollama...${NC}"
if ! pgrep -x "ollama" > /dev/null; then
    echo -e "${YELLOW}Starting Ollama in background...${NC}"
    ollama serve > /dev/null 2>&1 &
    sleep 2
fi

# Check model (optional)
if ! ollama list 2>/dev/null | grep -q "gemma3:4b"; then
    echo -e "${YELLOW}AI model not found (audio will be simple)${NC}"
fi

# Check camera
echo -e "${BLUE}Checking camera...${NC}"
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('✓ Camera ready' if cap.isOpened() else '✗ Camera not available'); cap.release()" 2>/dev/null

# Create directories
mkdir -p data/logs data/maps models/yolo models/depth

# System info
echo ""
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  PRODUCTION FEATURES${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "✓ Smart motion-based caching (2-3x faster)"
echo "✓ Predictive collision avoidance"
echo "✓ Robot-style clean UI"
echo "✓ Simple audio guidance"
echo "✓ Production error handling"
echo ""
echo -e "${BLUE}Display:${NC}"
echo "  • Robot Vision (main camera)"
echo "  • Depth Sensor (heat map)"
echo "  • Navigation Map (SLAM)"
echo ""
echo -e "${BLUE}Audio:${NC}"
echo "  • Simple messages: 'Stop. Car ahead. Go left'"
echo "  • Clear directional guidance"
echo ""
echo "Target: 30+ FPS | Press 'q' to stop"
echo ""

# Run production system
echo -e "${GREEN}Starting...${NC}"
echo ""
python3 src/main.py

# Cleanup
echo ""
echo -e "${GREEN}✓ OrbyGlasses stopped${NC}"
deactivate
