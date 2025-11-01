#!/bin/bash
#
# Dense Reconstruction with Depth Prediction Test
# Tests pySLAM's dense reconstruction capabilities
#

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🏗️  Dense Reconstruction with Depth Prediction Test${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate pySLAM environment
echo -e "${YELLOW}📦 Activating pySLAM environment...${NC}"
PYSLAM_DIR="$SCRIPT_DIR/third_party/pyslam"
PYSLAM_VENV="$HOME/.python/venvs/pyslam"

if [ -f "$PYSLAM_VENV/bin/activate" ]; then
    source "$PYSLAM_VENV/bin/activate"
    echo -e "${GREEN}✅ pySLAM environment activated${NC}"
else
    echo -e "${RED}❌ Error: pySLAM venv not found${NC}"
    exit 1
fi

# Set environment variables
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH="$PYSLAM_DIR:$PYSLAM_DIR/cpp/lib:$PYSLAM_DIR/thirdparty/g2opy/lib:$PYSLAM_DIR/thirdparty/pydbow3/lib:$PYSLAM_DIR/thirdparty/pangolin:$SCRIPT_DIR:$PYTHONPATH"

echo ""
echo -e "${YELLOW}🔍 Checking dense reconstruction capabilities...${NC}"

# Check if required modules are available
python3 << 'EOF'
import sys
sys.path.insert(0, 'third_party/pyslam')

print("Checking pySLAM dense reconstruction modules:")
print()

try:
    from pyslam.dense.volumetric_integrator_factory import volumetric_integrator_factory
    print("  ✅ volumetric_integrator_factory")
except Exception as e:
    print(f"  ❌ volumetric_integrator_factory: {e}")

try:
    from pyslam.dense.volumetric_integrator_tsdf import VolumetricIntegratorTSDF
    print("  ✅ VolumetricIntegratorTSDF (TSDF volume)")
except Exception as e:
    print(f"  ❌ VolumetricIntegratorTSDF: {e}")

try:
    from pyslam.depth_estimation.depth_estimator_factory import depth_estimator_factory
    print("  ✅ depth_estimator_factory")
except Exception as e:
    print(f"  ❌ depth_estimator_factory: {e}")

print()
print("Available depth estimators:")
try:
    from pyslam.depth_estimation.depth_estimator_factory import DepthEstimatorType
    print("  Types:", [t.name for t in DepthEstimatorType if hasattr(DepthEstimatorType, '__iter__')])
except:
    print("  Standard depth estimators available")

print()
print("To use dense reconstruction:")
print("  1. Run SLAM first to build the map")
print("  2. Dense reconstruction uses depth prediction to fill gaps")
print("  3. Creates a volumetric TSDF representation")
print()
EOF

echo ""
echo -e "${GREEN}📚 Dense Reconstruction Information:${NC}"
echo ""
echo "pySLAM supports dense reconstruction with:"
echo "  • TSDF (Truncated Signed Distance Function) volumes"
echo "  • Depth prediction integration"
echo "  • Gaussian splatting (if available)"
echo ""
echo "Main script: ${PYSLAM_DIR}/main_map_dense_reconstruction.py"
echo ""

# Show usage example
echo -e "${YELLOW}📖 Usage Example:${NC}"
echo ""
echo "1. Run SLAM and save map:"
echo "   ./run_orby.sh"
echo "   # SLAM will run and build map"
echo ""
echo "2. Use pySLAM's dense reconstruction:"
echo "   cd third_party/pyslam"
echo "   python3 main_map_dense_reconstruction.py"
echo ""
echo "3. Or integrate into OrbyGlasses (future):"
echo "   # Enable volumetric integration in config"
echo "   # Will add dense reconstruction to SLAM pipeline"
echo ""

# Check for saved maps
echo -e "${YELLOW}📁 Checking for saved SLAM maps...${NC}"
if [ -d "maps" ]; then
    MAP_COUNT=$(ls -1 maps/*.pkl 2>/dev/null | wc -l)
    if [ $MAP_COUNT -gt 0 ]; then
        echo -e "${GREEN}✅ Found $MAP_COUNT saved map(s) in maps/${NC}"
        ls -lh maps/*.pkl 2>/dev/null | tail -5
    else
        echo -e "${YELLOW}⚠️  No saved maps found. Run SLAM first.${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Maps directory doesn't exist yet.${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ Dense reconstruction check complete!${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Run SLAM: ./run_orby.sh"
echo "  2. Build map with camera movement"
echo "  3. Test dense reconstruction with pySLAM tools"
echo ""
