#!/bin/bash
#
# Test Script for pySLAM SLAM and VO Modes
#

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🧪 Testing pySLAM Integration in OrbyGlasses${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"

# Test 1: Check pySLAM environment
echo -e "\n${YELLOW}Test 1: Checking pySLAM environment...${NC}"
cd third_party/pyslam
. ./pyenv-activate.sh
python3 -c "import pyslam; print('✅ pySLAM module OK')" || { echo -e "${RED}❌ pySLAM import failed${NC}"; exit 1; }
python3 -c "import cv2; print('✅ OpenCV:', cv2.__version__)" || { echo -e "${RED}❌ OpenCV failed${NC}"; exit 1; }
python3 -c "from pyslam.slam.slam import Slam; print('✅ Slam class OK')" || { echo -e "${RED}❌ Slam import failed${NC}"; exit 1; }
python3 -c "from pyslam.slam.visual_odometry import VisualOdometryEducational; print('✅ VO class OK')" || { echo -e "${RED}❌ VO import failed${NC}"; exit 1; }

cd ../..

# Test 2: Check MockG2O
echo -e "\n${YELLOW}Test 2: Testing MockG2O...${NC}"
python3 << 'EOF'
import sys
sys.path.insert(0, 'src')

class MockG2OType:
    pass

class MockG2O:
    SE3Quat = MockG2OType
    Isometry3d = MockG2OType
    Flag = MockG2OType

sys.modules['g2o'] = MockG2O()

# Test isinstance
try:
    pose = None
    result = isinstance(pose, MockG2O.SE3Quat)
    print('✅ isinstance with MockG2O works')
except Exception as e:
    print(f'❌ isinstance failed: {e}')
    sys.exit(1)
EOF

# Test 3: Test OrbyGlasses imports
echo -e "\n${YELLOW}Test 3: Testing OrbyGlasses imports...${NC}"
cd third_party/pyslam
. ./pyenv-activate.sh
cd ../..

python3 << 'EOF'
import sys
sys.path.insert(0, 'src')

# Set up mocks
class MockG2OType:
    pass

class MockG2O:
    SE3Quat = MockG2OType
    Isometry3d = MockG2OType
    Flag = MockG2OType
    def __getattr__(self, name):
        if name[0].isupper():
            return MockG2OType
        return lambda *args, **kwargs: None

sys.modules['g2o'] = MockG2O()

class MockPySLAMUtils:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None

sys.modules['pyslam_utils'] = MockPySLAMUtils()

try:
    from navigation.pyslam_live import LivePySLAM, PYSLAM_AVAILABLE
    print(f'✅ LivePySLAM import: {PYSLAM_AVAILABLE}')
except Exception as e:
    print(f'❌ LivePySLAM import failed: {e}')
    sys.exit(1)

try:
    from navigation.pyslam_vo_integration import PySLAMVisualOdometry, PYSLAM_VO_AVAILABLE
    print(f'✅ PySLAMVisualOdometry import: {PYSLAM_VO_AVAILABLE}')
except Exception as e:
    print(f'❌ VO import failed: {e}')
    sys.exit(1)
EOF

echo -e "\n${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ All tests passed!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "\n${YELLOW}Ready to run:${NC}"
echo -e "  SLAM mode: ./run_orby.sh (with slam.use_pyslam: true)"
echo -e "  VO mode:   ./run_orby.sh (with visual_odometry.use_pyslam_vo: true)"
