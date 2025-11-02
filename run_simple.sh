#!/bin/bash
#
# Simple OrbyGlasses Launcher - Without SLAM (More Stable)
#
# Use this if SLAM crashes or you don't need indoor tracking
#

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 OrbyGlasses - Simple Mode (No SLAM)${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════${NC}"

cd "$(dirname "${BASH_SOURCE[0]}")"

# Temporarily disable SLAM
echo -e "${YELLOW}📝 Disabling SLAM for stable operation...${NC}"
cp config/config.yaml config/config.yaml.backup

# Set SLAM to disabled
sed -i.tmp 's/  enabled: true/  enabled: false/' config/config.yaml
rm -f config/config.yaml.tmp

echo -e "${GREEN}✅ Running in simple mode (object detection + distance only)${NC}"
echo ""

# Run with simple environment
export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 src/main.py "$@"

EXIT_CODE=$?

# Restore config
if [ -f config/config.yaml.backup ]; then
    mv config/config.yaml.backup config/config.yaml
    echo -e "${YELLOW}📝 Config restored${NC}"
fi

if [ $EXIT_CODE -ne 0 ]; then
    echo -e "\n${RED}❌ Exit code: $EXIT_CODE${NC}"
fi

exit $EXIT_CODE
