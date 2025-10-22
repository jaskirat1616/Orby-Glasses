#!/bin/bash
# Setup script for Simplified OrbyGlasses with depth estimation

echo "=========================================="
echo "Simplified OrbyGlasses Setup with Depth Estimation"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Step 1: Creating/activating virtual environment...${NC}"
python3 -m venv venv
source venv/bin/activate

echo -e "${GREEN}Step 2: Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

echo -e "${GREEN}Step 3: Installing dependencies...${NC}"
pip install ultralytics torch torchvision torchaudio transformers ollama Pillow opencv-python numpy colorlog pyaudio pydub SpeechRecognition

echo -e "${GREEN}Step 4: Checking for Ollama...${NC}"
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Ollama not found. Installing Ollama...${NC}"
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo -e "${GREEN}Step 5: Starting Ollama service...${NC}"
# Start Ollama in background if not running
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve &
    sleep 5
fi

echo -e "${GREEN}Step 6: Pulling Ollama models...${NC}"
echo "Pulling Gemma 3 4B model..."
ollama pull gemma3:4b

echo -e "${GREEN}Step 7: Creating required directories...${NC}"
mkdir -p models/yolo
mkdir -p data/logs

echo -e "${GREEN}=========================================="
echo "Setup Complete!"
echo "==========================================${NC}"

echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To start the simplified OrbyGlasses, run:"
echo "  python simple_orbyglasses.py"
echo ""
echo "To run with no display (headless), run:"
echo "  python simple_orbyglasses.py --no-display"
echo ""
echo -e "${YELLOW}Note: Make sure Ollama is running: ollama serve${NC}"
echo ""
echo -e "${YELLOW}First run may take longer as models are downloaded${NC}"
echo ""