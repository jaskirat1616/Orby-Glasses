#!/bin/bash

# OrbyGlasses Setup Script
# This script sets up the complete environment for OrbyGlasses on macOS with Apple Silicon

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

echo -e "${GREEN}Step 1: Creating virtual environment...${NC}"
python3.12 -m venv venv || python3 -m venv venv
source venv/bin/activate

echo -e "${GREEN}Step 2: Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

echo -e "${GREEN}Step 3: Installing Python dependencies...${NC}"
pip install -r requirements.txt

echo -e "${GREEN}Step 4: Checking for Homebrew...${NC}"
if ! command -v brew &> /dev/null; then
    echo -e "${YELLOW}Homebrew not found. Installing Homebrew...${NC}"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

echo -e "${GREEN}Step 5: Installing system dependencies...${NC}"
# Install portaudio for PyAudio
brew list portaudio &>/dev/null || brew install portaudio

# Install ffmpeg for audio processing
brew list ffmpeg &>/dev/null || brew install ffmpeg

echo -e "${GREEN}Step 6: Checking for Ollama...${NC}"
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Ollama not found. Installing Ollama...${NC}"
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo -e "${GREEN}Step 7: Starting Ollama service...${NC}"
# Start Ollama in background if not running
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve &
    sleep 5
fi

echo -e "${GREEN}Step 8: Pulling Ollama models...${NC}"
echo "Pulling Gemma 3 4B model (this may take a few minutes)..."
ollama pull gemma2:2b  # Using gemma2:2b as it's more widely available

echo "Pulling Moondream vision model..."
ollama pull moondream || echo -e "${YELLOW}Moondream not available, will use Gemma for vision${NC}"

echo -e "${GREEN}Step 9: Downloading YOLOv11 model...${NC}"
python3 << 'PYEOF'
from ultralytics import YOLO
import os

# Create models directory
os.makedirs('models/yolo', exist_ok=True)

# Download YOLOv11n (nano) model
print("Downloading YOLOv11n...")
model = YOLO('yolo11n.pt')
print(f"Model downloaded successfully to: {model.ckpt_path}")
PYEOF

echo -e "${GREEN}Step 10: Downloading and converting Depth Pro model...${NC}"
python3 << 'PYEOF'
import os
import urllib.request
import torch

os.makedirs('models/depth', exist_ok=True)

print("Downloading Apple Depth Pro model...")
# Apple's Depth Pro model URL
# Note: This is a placeholder. Actual URL from Apple ML Research:
# https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt

depth_model_url = "https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt"
depth_model_path = "models/depth/depth_pro.pt"

try:
    if not os.path.exists(depth_model_path):
        print("Downloading from Apple ML...")
        urllib.request.urlretrieve(depth_model_url, depth_model_path)
        print(f"Depth Pro model downloaded to {depth_model_path}")
    else:
        print("Depth Pro model already exists")
except Exception as e:
    print(f"Warning: Could not download Depth Pro model: {e}")
    print("You may need to download it manually from Apple's ML site")
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
    path: "models/depth/depth_pro.pt"
    device: "mps"

  llm:
    primary: "gemma2:2b"
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
fi

echo -e "${GREEN}Step 12: Creating sample test data...${NC}"
mkdir -p data/test_videos
mkdir -p data/logs
touch data/logs/.gitkeep
touch data/test_videos/.gitkeep

echo -e "${GREEN}Step 13: Creating .gitignore...${NC}"
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
models/depth/*.pt
models/depth/*.mlmodel
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
EOF

echo -e "${GREEN}Step 14: Running initial tests...${NC}"
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

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
echo -e "${YELLOW}Note: Make sure Ollama is running: ollama serve${NC}"
echo ""
