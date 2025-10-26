# OrbyGlasses

AI-powered navigation assistant for blind and visually impaired users.

## What It Does

OrbyGlasses uses a camera to detect objects, estimate distances, and provide audio guidance for safe navigation. It works entirely on your computer without sending data to the cloud.

**Key Features:**
- Real-time object detection
- Distance estimation
- Audio directions and warnings
- Indoor mapping and navigation
- Works without GPS
- **Voice commands** - "Hey Orby, where am I?"
- **Location memory** - Save and navigate to rooms
- **Smart context** - Adapts to time of day
- **Emergency mode** - Panic button with alerts
- **Battery optimizer** - Auto power saving

## Quick Start

```bash
./setup.sh   # First time only
./run.sh     # Start the system
```

Press `q` to stop.

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.10, 3.11, or 3.12
- Built-in camera or webcam
- Speakers or headphones

## How It Works

1. **Camera** captures video
2. **Object Detection** identifies objects (YOLOv11)
3. **Depth Estimation** measures distances (Depth Anything V2)
4. **SLAM** tracks your position indoors
5. **Audio Guidance** speaks directions and warnings

## Running Modes

**Standard Mode** (15-25 FPS):
```bash
./run.sh
```
Full features with detailed navigation.

**Fast Mode** (20-30 FPS):
```bash
./run.sh --fast
```
Optimized for speed with core features only.

**With Navigation Panel** (overhead compass view):
```bash
./run.sh --nav              # standard mode with nav panel
./run.sh --fast --nav       # fast mode with nav panel
```
Shows overhead map, compass, and real-time position tracking.

## Audio Feedback

The system provides clear spoken directions:
- **Danger** (<1m): "Stop. Car ahead. Go left."
- **Caution** (1-2.5m): "Chair on your right. Two meters away."
- **Safe** (>2.5m): "Path is clear."

## Configuration

Edit `config/config.yaml` or `config/config_fast.yaml`:

```yaml
camera:
  source: 0        # 0 = built-in camera
  width: 320
  height: 240

models:
  yolo:
    confidence: 0.55   # Higher = fewer false positives

safety:
  danger_distance: 1.0      # meters
  min_safe_distance: 1.5    # meters
```

## Project Structure

```
OrbyGlasses/
├── src/
│   ├── main.py                    # Main program
│   ├── core/                      # Core modules
│   │   ├── detection.py           # Object detection
│   │   ├── depth_anything_v2.py   # Depth estimation
│   │   ├── utils.py               # Configuration and audio
│   │   └── safety_system.py       # Danger detection
│   ├── navigation/                # Navigation modules
│   │   ├── slam_system.py         # Indoor mapping
│   │   └── indoor_navigation.py   # Path planning
│   └── visualization/             # Display modules
├── config/
│   ├── config.yaml                # Standard settings
│   └── config_fast.yaml           # Fast mode settings
├── tests/                         # Test suite
├── models/                        # AI models
└── data/                          # Logs and maps
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run a specific test:

```bash
pytest tests/test_detection.py -v
```

## Features in Detail

### Object Detection
Uses YOLOv11n to identify objects like cars, people, chairs, and doors.

### Depth Estimation
Calculates distances to objects using Depth Anything V2 Small model.

### Visual SLAM
Tracks your position and builds a map of indoor spaces without GPS.

### Indoor Navigation
Plans paths to saved locations (e.g., "kitchen", "desk") and provides turn-by-turn directions.

### Voice Commands 🎤
Talk to OrbyGlasses hands-free:
- **"Hey Orby, what's around me?"** - Get scene description
- **"Hey Orby, is the path clear?"** - Check for obstacles
- **"Hey Orby, where am I?"** - Get current location
- **"Hey Orby, save location kitchen"** - Save current spot
- **"Hey Orby, take me to kitchen"** - Navigate to saved location
- **"Hey Orby, help"** - Activate emergency mode

### Location Memory 📍
**Save your favorite spots:**
```
"Hey Orby, save location bedroom"
"Hey Orby, save location kitchen"
"Hey Orby, save location front door"
```

**Navigate back anytime:**
```
"Hey Orby, take me to bedroom"
```
The system uses SLAM and A* pathfinding to guide you with turn-by-turn audio directions.

### Context-Aware Intelligence 🧠

**Time-of-Day Adaptation:**
- **Night mode** (10pm-5am): Louder alerts, more cautious (2m danger zone)
- **Day mode**: Standard settings
- **Dawn/Dusk**: Moderate adjustments

**Battery Optimizer:**
- Monitors battery level automatically
- Below 20%: Activates power-saving mode
  - Reduces frame processing
  - Disables non-essential features
  - Extends runtime by 40%+

**Emergency Mode:**
Press 'e' key or say "Hey Orby, help" to activate:
- Loud emergency beeping
- Flashing screen (red)
- Voice alert: "HELP. Emergency alert activated"
- Logs emergency event for review

### Safety System
Monitors your surroundings and issues warnings at three levels:
- Danger zone (<1m): Immediate audio alert
- Caution zone (1-2.5m): Regular updates
- Safe zone (>2.5m): Quiet operation

## Privacy

All processing happens locally on your computer. No data is sent to the internet.

## Safety Note

OrbyGlasses is a navigation aid, not a replacement for traditional tools like white canes or guide dogs. Always use it alongside your existing mobility aids.

## Technical Details

**AI Models:**
- YOLOv11n for object detection
- Depth Anything V2 Small for depth estimation
- Moondream for scene understanding
- Gemma 3 4B for language processing

**Performance:**
- Standard mode: 15-25 FPS
- Fast mode: 20-30 FPS
- Depth processing: 70-200ms per frame
- Audio latency: 0.8-2s

**Hardware Acceleration:**
Uses Apple Metal Performance Shaders (MPS) for fast AI inference on M-series chips.

## Troubleshooting

**Camera not working:**
```bash
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"
```

**Ollama not running:**
```bash
ollama serve
```

**Poor performance:**
Use fast mode or reduce camera resolution in the config file.

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - see LICENSE file for details.

## What Makes OrbyGlasses Different

**🎯 AI-Powered Intelligence:**
- Voice-activated assistant responds to natural commands
- Context-aware scene understanding (knows you're in a kitchen vs hallway)
- Predicts obstacles before you reach them
- Learns your home layout

**🔋 Smart Adaptation:**
- Auto-adjusts for night/day (louder at night, more cautious)
- Battery optimizer extends runtime by 40%
- Emergency panic mode with audio/visual alerts

**🗺️ Location Memory:**
- Save unlimited locations by voice
- "Take me to kitchen" - automatic pathfinding
- Works entirely offline

**🎤 Hands-Free Operation:**
- No buttons to press while navigating
- Natural voice commands
- Runs in background thread (zero performance impact)

**🔒 Privacy First:**
- 100% local processing
- No cloud, no data collection
- Your maps stay on your device

## Performance

**Standard Mode:**
- 15-25 FPS
- Full feature set
- Best accuracy

**Fast Mode:**
- 20-30 FPS
- Core features only
- Better battery life

**Battery Saver (auto-activates <20%):**
- Essential features only
- 40%+ runtime extension
- Maintains safety alerts

## Credits

Built with:
- Ultralytics YOLOv11 for object detection
- Depth-Anything-V2 for depth estimation
- Ollama for local AI inference (Moondream, Gemma)
- OpenCV for computer vision
- PyTorch for deep learning

---

For detailed setup instructions, see the comments in `setup.sh` and `run.sh`.
