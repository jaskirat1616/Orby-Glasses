#!/bin/bash

# Run OrbyGlasses with pySLAM environment
# This activates the pySLAM venv and runs main.py

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}🚀 Starting OrbyGlasses with pySLAM${NC}"

# Navigate to pySLAM directory and activate
cd /Users/jaskiratsingh/Desktop/OrbyGlasses/third_party/pyslam
. ./pyenv-activate.sh

# Return to main directory
cd /Users/jaskiratsingh/Desktop/OrbyGlasses

# Run main.py with any passed arguments
echo -e "${GREEN}✅ Running main.py with pySLAM...${NC}"
python3 src/main.py "$@"
