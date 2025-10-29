#!/bin/bash

# OrbyGlasses Launcher
# Usage: ./run.sh [--fast] [--nav]

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Parse arguments
FAST_MODE=false
NAV_MODE=false
for arg in "$@"; do
    if [ "$arg" = "--fast" ]; then
        FAST_MODE=true
    elif [ "$arg" = "--nav" ]; then
        NAV_MODE=true
    fi
done

clear

echo ""
echo -e "${BOLD}${CYAN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║                                                           ║${NC}"
echo -e "${BOLD}${CYAN}║              OrbyGlasses Navigation System                ║${NC}"
if [ "$FAST_MODE" = true ]; then
echo -e "${BOLD}${CYAN}║                    (Fast Mode)                            ║${NC}"
fi
if [ "$NAV_MODE" = true ]; then
echo -e "${BOLD}${CYAN}║              (Navigation Panel Enabled)                   ║${NC}"
fi
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

# Activate pySLAM virtual environment if it exists
# Note: This will work whether you're in a venv or not
if [ -d "$HOME/.python/venvs/pyslam" ]; then
    echo -e "${GREEN}✓ Activating pySLAM environment${NC}"
    # Use the pySLAM environment's Python directly
    export PYTHONPATH="$HOME/.python/venvs/pyslam/lib/python3.11/site-packages:$PYTHONPATH"
    # Add pySLAM to the path
    export PATH="$HOME/.python/venvs/pyslam/bin:$PATH"
fi

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

if [ "$FAST_MODE" = true ]; then
    echo -e "${BOLD}Fast Mode Features:${NC}"
    echo -e "  ${GREEN}✓${NC} Object detection: YOLOv11n"
    echo -e "  ${GREEN}✓${NC} Depth estimation: 320x240 resolution"
    echo -e "  ${GREEN}✓${NC} SLAM navigation"
    echo -e "  ${GREEN}✓${NC} Audio guidance"
    echo -e "  ${GREEN}✓${NC} Performance: 20-30 FPS"
else
    echo -e "${BOLD}System Features:${NC}"
    echo -e "  ${GREEN}✓${NC} YOLOv11n: Real-time object detection"
    echo -e "  ${GREEN}✓${NC} Apple Depth Pro: High-quality depth estimation"
    echo -e "  ${GREEN}✓${NC} Visual SLAM: Indoor navigation & mapping"
    echo -e "  ${GREEN}✓${NC} Audio guidance with priority alerts"
    echo -e "  ${GREEN}✓${NC} Performance: 15-25 FPS"
fi

if [ "$NAV_MODE" = true ]; then
    echo ""
    echo -e "${BOLD}Navigation Panel:${NC}"
    echo -e "  ${GREEN}✓${NC} Overhead compass view"
    echo -e "  ${GREEN}✓${NC} Real-time position tracking"
    echo -e "  ${GREEN}✓${NC} Indoor map display"
fi

echo ""
echo -e "${BOLD}For Blind & Visually Impaired Users:${NC}"
echo -e "  • Clear audio directions with distance information"
echo -e "  • Immediate danger zone warnings (<1m)"
echo -e "  • Safe path suggestions with directional guidance"
echo ""
echo -e "${BOLD}Controls:${NC} Press 'q' to stop | Arrow keys for SLAM controls"
echo ""
echo -e "${GREEN}Starting OrbyGlasses...${NC}"
echo ""

# Run system
if [ "$FAST_MODE" = true ]; then
    if [ "$NAV_MODE" = true ]; then
        python3 src/main.py --config config/config_fast.yaml
    else
        python3 src/main.py --config config/config_fast.yaml
    fi
else
    if [ "$NAV_MODE" = true ]; then
        python3 src/main.py
    else
        python3 src/main.py --separate-slam
    fi
fi

# Cleanup
echo ""
echo -e "${GREEN}System stopped${NC}"
echo ""

deactivate
