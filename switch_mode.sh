#!/bin/bash
#
# OrbyGlasses Mode Switcher - Switch between SLAM and VO modes
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

CONFIG_FILE="config/config.yaml"

show_usage() {
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}🔄 OrbyGlasses Mode Switcher${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Usage: ./switch_mode.sh [MODE]"
    echo ""
    echo "Available modes:"
    echo "  slam        - Full SLAM mode (3D mapping, loop closure)"
    echo "  slam_dense  - SLAM + Real-time Dense 3D Reconstruction"
    echo "  vo          - Visual Odometry only (faster, less accurate)"
    echo "  both        - Both SLAM and VO (for comparison)"
    echo "  off         - Disable both (object detection only)"
    echo ""
    echo "Examples:"
    echo "  ./switch_mode.sh slam        # Switch to full SLAM"
    echo "  ./switch_mode.sh slam_dense  # SLAM with live dense mapping"
    echo "  ./switch_mode.sh vo          # Switch to VO only"
    echo "  ./switch_mode.sh both        # Run both together"
    echo ""
}

update_config() {
    local mode=$1

    case $mode in
        slam)
            echo -e "${YELLOW}Configuring FULL SLAM mode...${NC}"
            sed -i.bak 's/^slam:$/slam:/; /^slam:$/,/^[^ ]/ { s/enabled: .*/enabled: true/; }' "$CONFIG_FILE"
            sed -i.bak 's/^visual_odometry:$/visual_odometry:/; /^visual_odometry:$/,/^[^ ]/ { s/enabled: .*/enabled: false/; }' "$CONFIG_FILE"
            # Disable real-time dense mapping for regular SLAM
            sed -i.bak '/dense_reconstruction:/,/depth_trunc:/ { s/enabled: .*/enabled: false/; }' "$CONFIG_FILE"
            echo -e "${GREEN}✅ SLAM mode enabled${NC}"
            echo -e "${BLUE}Windows: Pangolin 3D Viewer, trajectory plots, feature tracking${NC}"
            ;;
        slam_dense)
            echo -e "${YELLOW}Configuring SLAM + REAL-TIME DENSE RECONSTRUCTION mode...${NC}"
            sed -i.bak 's/^slam:$/slam:/; /^slam:$/,/^[^ ]/ { s/enabled: .*/enabled: true/; }' "$CONFIG_FILE"
            sed -i.bak 's/^visual_odometry:$/visual_odometry:/; /^visual_odometry:$/,/^[^ ]/ { s/enabled: .*/enabled: false/; }' "$CONFIG_FILE"
            # Enable real-time dense mapping
            sed -i.bak '/dense_reconstruction:/,/depth_trunc:/ { s/enabled: .*/enabled: true/; }' "$CONFIG_FILE"
            echo -e "${GREEN}✅ SLAM + Dense Reconstruction mode enabled${NC}"
            echo -e "${BLUE}Windows: Pangolin 3D with dense point cloud/mesh${NC}"
            echo ""
            echo -e "${YELLOW}⚠️  IMPORTANT NOTES:${NC}"
            echo -e "${YELLOW}   • Requires stereo/RGBD camera for real-time dense mapping${NC}"
            echo -e "${YELLOW}   • Monocular cameras will skip frames (no depth data)${NC}"
            echo -e "${YELLOW}   • CPU/GPU intensive - expect slower performance${NC}"
            echo ""
            echo -e "${BLUE}For monocular cameras (OrbyGlasses default):${NC}"
            echo -e "  Use post-processing instead:"
            echo -e "    ./switch_mode.sh slam && ./run_orby.sh"
            echo -e "    ./dense_reconstruction.sh"
            ;;
        vo)
            echo -e "${YELLOW}Configuring VISUAL ODOMETRY mode...${NC}"
            sed -i.bak 's/^slam:$/slam:/; /^slam:$/,/^[^ ]/ { s/enabled: .*/enabled: false/; }' "$CONFIG_FILE"
            sed -i.bak 's/^visual_odometry:$/visual_odometry:/; /^visual_odometry:$/,/^[^ ]/ { s/enabled: .*/enabled: true/; }' "$CONFIG_FILE"
            echo -e "${GREEN}✅ Visual Odometry mode enabled${NC}"
            echo -e "${BLUE}Windows: VO trajectory, camera view with features${NC}"
            ;;
        both)
            echo -e "${YELLOW}Configuring BOTH SLAM + VO mode...${NC}"
            sed -i.bak 's/^slam:$/slam:/; /^slam:$/,/^[^ ]/ { s/enabled: .*/enabled: true/; }' "$CONFIG_FILE"
            sed -i.bak 's/^visual_odometry:$/visual_odometry:/; /^visual_odometry:$/,/^[^ ]/ { s/enabled: .*/enabled: true/; }' "$CONFIG_FILE"
            echo -e "${GREEN}✅ SLAM + VO mode enabled${NC}"
            echo -e "${BLUE}Windows: All pySLAM windows (may be resource intensive)${NC}"
            ;;
        off)
            echo -e "${YELLOW}Disabling SLAM and VO...${NC}"
            sed -i.bak 's/^slam:$/slam:/; /^slam:$/,/^[^ ]/ { s/enabled: .*/enabled: false/; }' "$CONFIG_FILE"
            sed -i.bak 's/^visual_odometry:$/visual_odometry:/; /^visual_odometry:$/,/^[^ ]/ { s/enabled: .*/enabled: false/; }' "$CONFIG_FILE"
            echo -e "${GREEN}✅ SLAM and VO disabled${NC}"
            echo -e "${BLUE}Running in object detection only mode${NC}"
            ;;
        *)
            echo -e "${RED}❌ Unknown mode: $mode${NC}"
            show_usage
            exit 1
            ;;
    esac

    # Clean up backup file
    rm -f "$CONFIG_FILE.bak"
}

# Main
if [ $# -eq 0 ]; then
    show_usage
    exit 0
fi

MODE=$1

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Error: Config file not found: $CONFIG_FILE${NC}"
    exit 1
fi

update_config "$MODE"

echo ""
echo -e "${GREEN}🎯 Mode switch complete!${NC}"
echo ""
echo -e "${YELLOW}To run OrbyGlasses:${NC}"
echo "  ./run_orby.sh"
echo ""
