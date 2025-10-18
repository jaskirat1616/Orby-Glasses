# OrbyGlasses

**Bio-Mimetic Navigation Engine for Visually Impaired Users**

OrbyGlasses is an innovative, AI-powered navigation assistance system designed to help visually impaired individuals navigate safely using smart glasses with a webcam. The system runs entirely locally on Apple Silicon (M2 Max) using cutting-edge computer vision, depth estimation, spatial audio, and AI-generated contextual guidance.

---

## Features

### Core Capabilities

- **Real-Time Object Detection**: YOLOv11n optimized for Apple Silicon MPS acceleration
- **Depth Estimation**: Apple Depth Pro (with MiDaS fallback) for accurate distance measurement
- **Bio-Mimetic Echolocation**: Spatial audio cues simulating bat echolocation using binaural sound
- **AI-Powered Narratives**: Contextual navigation guidance using Ollama (Gemma 3 + Moondream vision models)
- **Predictive Navigation**: Reinforcement learning (PPO) to learn user patterns and predict optimal paths
- **Text-to-Speech**: Real-time audio feedback for obstacle alerts and navigation guidance
- **Privacy-First**: 100% local processing, no cloud dependencies

### Technical Highlights

- **High Accuracy**: Targets 95%+ object detection accuracy with confidence thresholding
- **Low Latency**: Optimized for <100ms per frame processing
- **Adaptive Learning**: RL agent learns from user navigation patterns
- **Modular Architecture**: Easy to extend and customize
- **Comprehensive Testing**: Unit tests for all major components

---

## System Requirements

### Hardware
- **Computer**: MacBook with Apple Silicon (M2 Max recommended)
- **Camera**: Built-in webcam or IP camera (e.g., smart glasses with WiFi streaming)
- **Audio**: Speakers or headphones for spatial audio output

### Software
- **OS**: macOS 13.0+ (Ventura or later)
- **Python**: 3.12
- **Homebrew**: For system dependencies
- **Ollama**: For LLM inference

---

## Quick Start

### 1. Clone the Repository

```bash
cd ~/Desktop
git clone <your-repo-url> OrbyGlasses
cd OrbyGlasses
```

### 2. Run Setup Script

The setup script automates the entire installation process:

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Install Homebrew packages (portaudio, ffmpeg)
- Install and start Ollama
- Download AI models (Gemma 3, Moondream, YOLOv11n)
- Set up project directories

### 3. Activate Environment

```bash
source venv/bin/activate
```

### 4. Run OrbyGlasses

```bash
python src/main.py
```

Press `q` to stop the system.

---

## Configuration

Edit `config/config.yaml` to customize behavior:

```yaml
camera:
  source: 0  # 0 for built-in webcam, or IP camera URL
  width: 640
  height: 480

models:
  yolo:
    confidence: 0.5  # Detection confidence threshold
  llm:
    primary: "gemma2:2b"  # Primary narrative model
    vision: "moondream"   # Vision understanding model

safety:
  min_safe_distance: 1.5  # Meters
```

---

## Usage Examples

### Basic Navigation

```bash
python src/main.py
```

### No Display (Headless Mode)

```bash
python src/main.py --no-display
```

### Save Output Video

```bash
python src/main.py --save-video
```

### Train RL Model

```bash
python src/main.py --train-rl
```

### Use IP Camera

Edit `config/config.yaml`:

```yaml
camera:
  source: "http://192.168.1.100:8080/video"
```

---

## Architecture

### Pipeline Flow

```
Camera Feed
    ↓
Object Detection (YOLOv11)
    ↓
Depth Estimation (Depth Pro/MiDaS)
    ↓
Navigation Summary
    ↓
┌─────────────┬──────────────┬──────────────┐
│             │              │              │
Echolocation  AI Narrative   RL Prediction
(Spatial      (Gemma +       (PPO)
 Audio)        Moondream)
│             │              │
└─────────────┴──────────────┴──────────────┘
    ↓
Audio Output (TTS + Beeps)
```

### Module Descriptions

- **`detection.py`**: YOLO object detection and depth estimation
- **`echolocation.py`**: Spatial audio generation using pyroomacoustics
- **`narrative.py`**: AI narrative generation with Ollama
- **`prediction.py`**: Reinforcement learning path prediction
- **`utils.py`**: Configuration, logging, audio management
- **`main.py`**: Main application entry point

---

## Project Structure

```
OrbyGlasses/
├── src/
│   ├── main.py              # Entry point
│   ├── detection.py         # Object detection & depth
│   ├── echolocation.py      # Spatial audio
│   ├── narrative.py         # AI narratives
│   ├── prediction.py        # RL predictions
│   └── utils.py             # Utilities
├── models/
│   ├── yolo/                # YOLOv11 models
│   ├── depth/               # Depth estimation models
│   └── rl/                  # RL agent checkpoints
├── data/
│   ├── logs/                # Session logs
│   └── test_videos/         # Test videos
├── config/
│   └── config.yaml          # Configuration
├── tests/
│   ├── test_detection.py
│   ├── test_echolocation.py
│   └── test_utils.py
├── requirements.txt
├── setup.sh
└── README.md
```

---

## Testing

Run unit tests:

```bash
pytest tests/ -v
```

Run specific test file:

```bash
pytest tests/test_detection.py -v
```

---

## Development

### Adding New Features

1. Create feature branch:
   ```bash
   git checkout -b feature/your-feature
   ```

2. Implement feature in appropriate module

3. Add unit tests in `tests/`

4. Update configuration if needed

5. Test thoroughly:
   ```bash
   pytest tests/
   python src/main.py
   ```

6. Commit and push:
   ```bash
   git add .
   git commit -m "Add: your feature description"
   git push origin feature/your-feature
   ```

### Customizing Models

**Change YOLO Model:**

```yaml
# config/config.yaml
models:
  yolo:
    path: "models/yolo/yolo11s.pt"  # Use small instead of nano
```

**Change LLM:**

```bash
ollama pull llama3.2:3b
```

```yaml
# config/config.yaml
models:
  llm:
    primary: "llama3.2:3b"
```

---

## Performance Optimization

### For Maximum Speed

- Use YOLOv11n (nano) - already default
- Lower camera resolution: 320x240
- Reduce audio update interval
- Disable vision model if not needed

### For Maximum Accuracy

- Use YOLOv11m (medium) or larger
- Increase confidence threshold
- Enable vision model for scene understanding
- Higher camera resolution: 1280x720

---

## Troubleshooting

### Camera Not Found

```bash
# Test camera
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### MPS Not Available

```bash
# Check MPS support
python -c "import torch; print(torch.backends.mps.is_available())"
```

If False, models will fallback to CPU automatically.

### Ollama Connection Error

```bash
# Start Ollama service
ollama serve

# In another terminal
ollama list  # Check installed models
```

### Audio Issues

```bash
# Check portaudio
brew install portaudio

# Reinstall PyAudio
pip uninstall pyaudio
pip install pyaudio
```

---

## Roadmap

- [ ] Integration with actual smart glasses (e.g., Vuzix, Xreal)
- [ ] Multi-user federated learning with Flower
- [ ] GPS integration for outdoor navigation
- [ ] Obstacle avoidance path planning
- [ ] Voice command interface
- [ ] Mobile app companion
- [ ] Cloud-optional model updates

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

## License

MIT License - See LICENSE file for details

---

## Citation

If you use OrbyGlasses in your research, please cite:

```bibtex
@software{orbyglass2025,
  title={OrbyGlasses: Bio-Mimetic Navigation Engine for Visually Impaired Users},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/OrbyGlasses}
}
```

---

## Acknowledgments

- **Ultralytics** for YOLOv11
- **Intel ISL** for MiDaS depth estimation
- **Anthropic** for Ollama ecosystem
- **PyRoomAcoustics** for spatial audio simulation
- **Stable Baselines3** for RL implementations

---

## Contact

For questions, issues, or collaboration:

- GitHub Issues: [OrbyGlasses Issues](https://github.com/yourusername/OrbyGlasses/issues)
- Email: your.email@example.com

---

**OrbyGlasses - Empowering Independence Through AI**
