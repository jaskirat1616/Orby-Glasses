#!/bin/bash

# OrbyGlasses Launcher

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

clear

echo ""
echo -e "${BOLD}${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║                                                           ║${NC}"
echo -e "${BOLD}${CYAN}║              OrbyGlasses Navigation System                ║${NC}"
echo -e "${BOLD}${CYAN}║                                                           ║${NC}"
echo -e "${BOLD}${CYAN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

sleep 0.3

echo -e "${YELLOW}Starting system...${NC}"
echo ""

# Check virtual environment
if [ ! -d "venv" ]; then
    echo -e "${RED}✗ Virtual environment not found${NC}"
    echo "Run ./setup.sh first"
    exit 1
fi

# Activate
echo -e "${GREEN}✓ Loading environment${NC}"
source venv/bin/activate

# Check Ollama
if ! pgrep -x "ollama" > /dev/null; then
    echo -e "${BLUE}Starting AI engine...${NC}"
    ollama serve > /dev/null 2>&1 &
    sleep 1
fi
echo -e "${GREEN}✓ AI engine ready${NC}"

# Check camera
python3 -c "import cv2; cap = cv2.VideoCapture(0); cap.release()" 2>/dev/null
echo -e "${GREEN}✓ Camera ready${NC}"

# Create directories
mkdir -p data/logs data/maps models/yolo models/depth

echo ""
echo -e "${BOLD}Features:${NC}"
echo -e "  • Smart caching for 30+ FPS"
echo -e "  • Predictive collision warnings"
echo -e "  • Large readable text"
echo -e "  • Simple audio guidance"
echo ""
echo -e "${BOLD}Controls:${NC} Press 'q' to stop"
echo ""
echo -e "${GREEN}Launching...${NC}"
echo ""

# Run system
python3 src/main.py

# Cleanup
echo ""
echo -e "${GREEN}System stopped${NC}"
echo ""

deactivate
