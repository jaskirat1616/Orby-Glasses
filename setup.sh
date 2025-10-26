#!/bin/bash

# OrbyGlasses Setup Script
# Sets up the environment for OrbyGlasses on macOS with Apple Silicon

set -e

echo "=========================================="
echo "OrbyGlasses Setup"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: This script is designed for macOS${NC}"
    exit 1
fi

# Check Python version (3.10, 3.11, or 3.12)
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
if [[ ! "$PYTHON_VERSION" =~ ^3\.(10|11|12) ]]; then
    echo -e "${RED}Error: Python 3.10, 3.11, or 3.12 required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}Step 1: Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

echo -e "${GREEN}Step 2: Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

echo -e "${GREEN}Step 3: Installing Python dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo -e "${YELLOW}Warning: requirements.txt not found${NC}"
fi
pip install transformers timm pillow

echo -e "${GREEN}Step 4: Checking for Homebrew...${NC}"
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}Homebrew not found. Installing...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

echo -e "${GREEN}Step 5: Installing system dependencies...${NC}"
if ! brew list portaudio &>/dev/null; then
    brew install portaudio
fi

if ! brew list ffmpeg &>/dev/null; then
    brew install ffmpeg
fi

echo -e "${GREEN}Step 6: Checking for Ollama...${NC}"
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Ollama not found. Installing...${NC}"
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo -e "${GREEN}Step 7: Starting Ollama service...${NC}"
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve &
    sleep 5
fi

echo -e "${GREEN}Step 8: Pulling AI models...${NC}"
echo "Pulling Gemma 3 4B model..."
ollama pull gemma3:4b || echo -e "${YELLOW}Failed to pull Gemma model${NC}"

echo "Pulling Moondream vision model..."
ollama pull moondream || echo -e "${YELLOW}Failed to pull Moondream model${NC}"

echo -e "${GREEN}Step 9: Downloading YOLOv11 model...${NC}"
python3 << 'PYEOF'
from ultralytics import YOLO
import os

os.makedirs('models/yolo', exist_ok=True)

print("Downloading YOLOv11n...")
try:
    model = YOLO('yolo11n.pt')
    print(f"Model downloaded to: {model.ckpt_path}")
except Exception as e:
    print(f"Warning: Could not download model: {e}")
PYEOF

echo -e "${GREEN}Step 10: Setting up Depth Anything V2 model...${NC}"
python3 << 'PYEOF'
import os
import torch
from transformers import pipeline
from PIL import Image
import numpy as np

os.makedirs('models/depth', exist_ok=True)

print("Setting up Depth Anything V2 Small model...")
model_name = "depth-anything/Depth-Anything-V2-Small-hf"

try:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipeline("depth-estimation", model=model_name, device=device)
    print(f"Model loaded successfully")

    # Test
    dummy_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    depth = pipe(dummy_img)["depth"]
    print(f"Test successful: {np.array(depth).shape}")

except Exception as e:
    print(f"Warning: Failed to setup model: {e}")
PYEOF

echo -e "${GREEN}Step 11: Creating directories...${NC}"
mkdir -p data/logs data/maps models/yolo models/depth

echo -e "${GREEN}Step 12: Running tests...${NC}"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS: {torch.backends.mps.is_available()}')"
python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

echo ""
echo -e "${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "To start OrbyGlasses:"
echo "  ./run.sh               (standard mode)"
echo "  ./run.sh --fast        (fast mode)"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
