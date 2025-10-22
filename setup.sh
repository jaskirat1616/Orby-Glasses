#!/bin/bash

# OrbyGlasses Setup Script
# Sets up the complete environment for OrbyGlasses on macOS with Apple Silicon
# Uses Depth Anything V2 Small (depth-anything/Depth-Anything-V2-Small-hf) via Transformers pipeline

set -e  # Exit on any error

echo "=========================================="
echo "OrbyGlasses Setup - Bio-Mimetic Navigation"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: This script is designed for macOS with Apple Silicon${NC}"
    exit 1
fi

# Check if running on Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo -e "${YELLOW}Warning: Not running on Apple Silicon. MPS acceleration may not be available.${NC}"
fi

# Check Python version (3.10, 3.11, or 3.12 recommended)
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
if [[ ! "$PYTHON_VERSION" =~ ^3\.(10|11|12) ]]; then
    echo -e "${RED}Error: Python 3.10, 3.11, or 3.12 is required. Found version: $PYTHON_VERSION${NC}"
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
    echo -e "${YELLOW}Warning: requirements.txt not found. Installing core dependencies.${NC}"
fi
pip install transformers timm pillow  # For Depth Anything V2 pipeline and image handling

echo -e "${GREEN}Step 4: Checking for Homebrew...${NC}"
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}Homebrew not found. Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add Homebrew to PATH
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

echo -e "${GREEN}Step 5: Installing system dependencies...${NC}"
# Install portaudio for PyAudio
if ! brew list portaudio &>/dev/null; then
    brew install portaudio
fi

# Install ffmpeg for audio processing
if ! brew list ffmpeg &>/dev/null; then
    brew install ffmpeg
fi

echo -e "${GREEN}Step 6: Checking for Ollama...${NC}"
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Ollama not found. Installing Ollama...${NC}"
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo -e "${GREEN}Step 7: Starting Ollama service...${NC}"
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve &
    sleep 5
fi

echo -e "${GREEN}Step 8: Pulling Ollama models...${NC}"
echo "Pulling Gemma 3 4B model (this may take a few minutes)..."
ollama pull gemma3:4b || echo -e "${YELLOW}Failed to pull Gemma 3 4B model. Please check Ollama installation.${NC}"

echo "Pulling Moondream vision model..."
ollama pull moondream || echo -e "${YELLOW}Moondream not available, will use Gemma for vision${NC}"

echo -e "${GREEN}Step 9: Downloading YOLOv11 model...${NC}"
python3 << 'PYEOF'
from ultralytics import YOLO
import os

os.makedirs('models/yolo', exist_ok=True)

print("Downloading YOLOv11n...")
try:
    model = YOLO('yolo12n.pt')  # Note: Ensure yolo12n.pt is valid; may need to update to yolov11n.pt
    print(f"Model downloaded successfully to: {model.ckpt_path}")
except Exception as e:
    print(f"Warning: Could not download YOLOv11n model: {e}")
    print("You may need to download it manually from the Ultralytics YOLOv11 repository")
PYEOF

echo -e "${GREEN}Step 10: Setting up Depth Anything V2 Small (ViT-S) model...${NC}"
python3 << 'PYEOF'
import os
import torch
from transformers import pipeline
from PIL import Image
import numpy as np

os.makedirs('models/depth', exist_ok=True)

print("Setting up Depth Anything V2 Small model (depth-anything/Depth-Anything-V2-Small-hf)...")
model_name = "depth-anything/Depth-Anything-V2-Small-hf"

try:
    # Check internet connectivity
    import socket
    socket.create_connection(("huggingface.co", 443), timeout=5)
    
    # Load pipeline to trigger model download/cache
    print(f"Loading {model_name} pipeline...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    pipe = pipeline("depth-estimation", model=model_name, device=device)
    print(f"Depth Anything V2 Small model successfully loaded via Transformers pipeline")
    print(f"Model cached in ~/.cache/huggingface/hub/ (~90MB)")
    
    # Test inference with dummy image
    dummy_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    depth = pipe(dummy_img)["depth"]
    print(f"Test inference successful: Depth map shape {np.array(depth).shape}")

except Exception as e:
    print(f"Warning: Failed to set up Depth Anything V2 Small model: {e}")
    print("Possible causes: No internet, insufficient disk space, or Hugging Face server issues.")
    print("Fallback: Manually download depth_anything_v2_vits.pth from:")
    print("  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true")
    print("Place it in models/depth/ and update config.yaml to use path-based loading.")
PYEOF

echo -e "${GREEN}Step 11: Creating configuration files...${NC}"
if [ ! -f "config/config.yaml" ]; then
    cat > config/config.yaml << 'EOF'
# OrbyGlasses Configuration

# Camera Settings
camera:
  source: 0  # 0 for built-in webcam, or URL like "http://192.168.1.100:8080/video"
  width: 640
  height: 480
  fps: 30

# Model Settings
models:
  yolo:
    path: "models/yolo/yolo11n.pt"
    confidence: 0.5
    iou_threshold: 0.45
    device: "mps"  # mps for Apple Silicon, cpu otherwise

  depth:
    model: "depth-anything/Depth-Anything-V2-Small-hf"  # Hugging Face pipeline model
    device: "mps"
    variant: "vits"  # For reference in inference code

  llm:
    primary: "gemma3:4b"
    vision: "moondream"
    temperature: 0.7
    max_tokens: 150

# Audio Settings
audio:
  tts_engine: "pyttsx3"
  tts_rate: 175
  tts_volume: 1.0
  echolocation_enabled: true
  spatial_audio: true

# Echolocation Settings
echolocation:
  room_dimensions: [10, 10, 3]  # meters [x, y, z]
  sample_rate: 16000
  duration: 0.1  # seconds per beep

# RL Prediction Settings
prediction:
  enabled: true
  model_path: "models/rl/ppo_navigation.zip"
  training_steps: 10000
  save_interval: 1000

# Safety Settings
safety:
  min_safe_distance: 1.5  # meters
  emergency_stop_key: "q"
  max_obstacle_alerts: 5

# Logging
logging:
  level: "INFO"
  log_file: "data/logs/orbyglass.log"
  save_detections: true
  save_frames: false
EOF
    echo "Created default config.yaml"
else
    echo "config.yaml already exists, skipping creation"
fi

echo -e "${GREEN}Step 12: Creating sample test data...${NC}"
mkdir -p data/test_videos
mkdir -p data/logs
touch data/logs/.gitkeep
touch data/test_videos/.gitkeep

echo -e "${GREEN}Step 13: Creating .gitignore...${NC}"
if [ ! -f ".gitignore" ]; then
    cat > .gitignore << 'EOF'
# Virtual Environment
venv/
env/
ENV/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
*.egg-info/
dist/
build/

# Models (large files)
models/yolo/*.pt
models/depth/*.pth
models/rl/*.zip

# Data
data/logs/*.log
data/test_videos/*.mp4
data/test_videos/*.avi

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Ollama (managed separately)
models/ollama/

# Temporary files
*.tmp
*.temp

# Hugging Face cache
.cache/huggingface/
EOF
    echo "Created .gitignore"
else
    echo ".gitignore already exists, skipping creation"
fi

echo -e "${GREEN}Step 14: Running initial tests...${NC}"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"
python3 -c "import transformers; print(f'transformers version: {transformers.__version__}')"

echo ""
echo -e "${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start OrbyGlasses, run:"
echo "  python src/main.py"
echo ""
echo "To run tests:"
echo "  pytest tests/"
echo ""
echo -e "${YELLOW}Notes:${NC}"
echo -e "${YELLOW}- Ensure Ollama is running: ollama serve${NC}"
echo -e "${YELLOW}- Depth Anything V2 Small model is cached in ~/.cache/huggingface/hub/ (~90MB)${NC}"
echo -e "${YELLOW}- If the Depth Anything V2 setup failed, ensure internet connectivity and sufficient disk space.${NC}"
echo -e "${YELLOW}- Manual fallback: Download depth_anything_v2_vits.pth from:${NC}"
echo -e "${YELLOW}  https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true${NC}"
echo -e "${YELLOW}  Place it in models/depth/ and update config.yaml to use path-based loading.${NC}"
echo ""