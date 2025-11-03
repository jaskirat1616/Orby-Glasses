#!/bin/bash
# OrbyGlasses Launcher Script
# Supports: --video <path>, --show-features, --separate-slam, --no-display
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

# Add pySLAM and thirdparty libraries to Python path
export PYTHONPATH="$PYSLAM_DIR:$PYSLAM_DIR/cpp/lib:$PYSLAM_DIR/thirdparty/g2opy/lib:$PYSLAM_DIR/thirdparty/pydbow3/lib:$PYSLAM_DIR/thirdparty/pangolin:$PYTHONPATH"

# Add current directory to Python path for src imports
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Check Python and dependencies
echo -e "${YELLOW}🔍 Checking dependencies...${NC}"
python3 -c "import cv2; print('  ✓ OpenCV:', cv2.__version__)" || { echo "  ✗ OpenCV not available"; exit 1; }
python3 -c "import torch; print('  ✓ PyTorch:', torch.__version__)" || { echo "  ✗ PyTorch not available"; exit 1; }
python3 -c "import pyslam; print('  ✓ pySLAM: OK')" || { echo "  ✗ pySLAM not available"; exit 1; }
python3 -c "from pyslam.slam.camera import PinholeCamera; print('  ✓ PinholeCamera: OK')" || { echo "  ✗ PinholeCamera not available"; exit 1; }
# pydbow3 removed - using iBoW loop closure instead (more stable)
python3 -c "import pypangolin; print('  ✓ pypangolin: OK')" || { echo "  ✗ pypangolin not available"; exit 1; }

echo -e "${GREEN}✅ All dependencies ready${NC}"
echo -e "${YELLOW}📝 Python path: $PYTHONPATH${NC}"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🎯 Launching OrbyGlasses...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

# Run OrbyGlasses with error handling
set +e  # Don't exit on error
python3 src/main.py "$@"
EXIT_CODE=$?

# Handle different exit codes
if [ $EXIT_CODE -eq 138 ]; then
    echo -e "\n${YELLOW}⚠️  System stopped by user (Ctrl+C)${NC}"
elif [ $EXIT_CODE -eq 139 ] || [ $EXIT_CODE -eq 10 ]; then
    echo -e "\n${RED}❌ Crash detected (Bus error/Segmentation fault)${NC}"
    echo -e "${YELLOW}   This is usually caused by SLAM relocalization.${NC}"
    echo -e "${YELLOW}   Try disabling loop_closure in config/config.yaml:${NC}"
    echo -e "${GREEN}   slam:${NC}"
    echo -e "${GREEN}     loop_closure: false${NC}"
elif [ $EXIT_CODE -ne 0 ]; then
    echo -e "\n${RED}❌ OrbyGlasses exited with error code: $EXIT_CODE${NC}"
fi

exit $EXIT_CODE
