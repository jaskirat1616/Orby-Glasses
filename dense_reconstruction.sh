#!/bin/bash
#
# OrbyGlasses Dense Map Reconstruction
# Generates dense 3D reconstruction from saved SLAM map
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🗺️  OrbyGlasses - Dense 3D Map Reconstruction${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

# Default paths
DEFAULT_MAP_PATH="results/slam_state"
DEFAULT_OUTPUT_PATH="results/dense_reconstruction"

# Parse arguments
MAP_PATH="${1:-$DEFAULT_MAP_PATH}"
OUTPUT_PATH="${2:-$DEFAULT_OUTPUT_PATH}"

echo -e "${YELLOW}📦 Activating pySLAM environment...${NC}"
# Source the pyslam environment
if [ -f "./third_party/pyslam/pyenv-activate.sh" ]; then
    source ./third_party/pyslam/pyenv-activate.sh
else
    echo -e "${RED}❌ Error: pySLAM environment script not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ pySLAM environment activated${NC}"
echo ""

# Check if map exists
if [ ! -d "$MAP_PATH" ]; then
    echo -e "${RED}❌ Error: SLAM map not found at: $MAP_PATH${NC}"
    echo ""
    echo "Please run SLAM first to generate a map:"
    echo "  ./switch_mode.sh slam"
    echo "  ./run_orby.sh"
    echo ""
    echo "Then save the map from the Pangolin window (press 's')"
    exit 1
fi

echo -e "${YELLOW}🔍 Loading SLAM map from: $MAP_PATH${NC}"
echo -e "${YELLOW}💾 Output will be saved to: $OUTPUT_PATH${NC}"
echo ""

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 Starting Dense Reconstruction...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}This will:${NC}"
echo "  1. Load the saved SLAM map"
echo "  2. Generate dense 3D reconstruction (TSDF volume)"
echo "  3. Create point cloud and/or mesh"
echo "  4. Display in 3D viewer"
echo ""
echo -e "${YELLOW}Controls in 3D viewer:${NC}"
echo "  • Mouse: Rotate view"
echo "  • Scroll: Zoom"
echo "  • 's': Save dense map"
echo "  • 'q': Quit"
echo ""

# Run the dense reconstruction
cd third_party/pyslam
python3 main_map_dense_reconstruction.py \
    --path "../../$MAP_PATH" \
    --output_path "../../$OUTPUT_PATH"

echo ""
echo -e "${GREEN}✅ Dense reconstruction complete!${NC}"
echo -e "${YELLOW}📁 Output saved to: $OUTPUT_PATH${NC}"
echo ""
