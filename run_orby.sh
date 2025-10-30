#!/bin/bash
#
# OrbyGlasses Launcher - Properly configured for pySLAM integration
#
# This script:
# 1. Activates the pySLAM virtual environment
# 2. Sets proper Python paths
# 3. Runs OrbyGlasses with full SLAM and VO capabilities
#

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 OrbyGlasses - AI Navigation System${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate pySLAM environment
echo -e "${YELLOW}📦 Activating pySLAM environment...${NC}"
PYSLAM_DIR="$SCRIPT_DIR/third_party/pyslam"

if [ ! -d "$PYSLAM_DIR" ]; then
    echo -e "${RED}❌ Error: pySLAM not found at $PYSLAM_DIR${NC}"
    exit 1
fi

# Source the pySLAM venv directly
PYSLAM_VENV="$HOME/.python/venvs/pyslam"
if [ -f "$PYSLAM_VENV/bin/activate" ]; then
    source "$PYSLAM_VENV/bin/activate"
    echo -e "${GREEN}✅ pySLAM environment activated${NC}"
else
    echo -e "${RED}❌ Error: pySLAM venv not found at $PYSLAM_VENV${NC}"
    echo -e "${YELLOW}   Please run: cd third_party/pyslam && ./install_all.sh${NC}"
    exit 1
fi

# Set environment variables for MPS fallback (Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Add pySLAM to Python path
export PYTHONPATH="$PYSLAM_DIR:$PYTHONPATH"

# Check Python and dependencies
echo -e "${YELLOW}🔍 Checking dependencies...${NC}"
python3 -c "import cv2; print('  ✓ OpenCV:', cv2.__version__)" || { echo "  ✗ OpenCV not available"; exit 1; }
python3 -c "import torch; print('  ✓ PyTorch:', torch.__version__)" || { echo "  ✗ PyTorch not available"; exit 1; }
python3 -c "import pyslam; print('  ✓ pySLAM: OK')" || { echo "  ✗ pySLAM not available"; exit 1; }

echo -e "${GREEN}✅ All dependencies ready${NC}"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🎯 Launching OrbyGlasses...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

# Run OrbyGlasses
python3 src/main.py "$@"
