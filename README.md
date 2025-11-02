# OrbyGlasses

Navigation assistant for blind and visually impaired people.

## What It Does

OrbyGlasses uses your camera to:
- Detect objects around you
- Measure how far they are
- Tell you through audio where to go safely

Everything runs on your computer - no internet needed.

## Quick Start

```bash
./run_orby.sh
```

Press `q` or spacebar to stop.

## What You Need

- Mac computer with M1, M2, M3, or M4 chip
- Python 3.10 or newer
- Camera (built-in or USB)
- Headphones or speakers

## How to Install

```bash
# Install OrbyGlasses
./install_pyslam.sh
pip install -r requirements.txt

# Run it
./run_orby.sh
```

## How It Works

1. Camera sees what's in front of you
2. Computer finds objects (cars, people, chairs, etc.)
3. Computer measures distance to each object
4. Computer tracks where you are indoors
5. Audio tells you where to go and what to avoid

## Audio Warnings

The system speaks to you based on distance:

- **Very close** (<1m): "Stop! Car ahead."
- **Close** (1-2.5m): "Person on your left. 2 meters."
- **Safe** (>2.5m): "Path is clear."

## Emergency Stop

Press SPACEBAR or Q anytime to stop immediately.

The system also stops automatically if:
- Something is too close (<0.5 meters)
- Camera stops working
- System detects a problem

## Settings

Edit `config/config.yaml` to change:

```yaml
camera:
  source: 0        # 0 = built-in camera, 1 = USB camera

safety:
  danger_distance: 1.0      # How close is too close (meters)
  min_safe_distance: 1.5    # Comfortable distance (meters)

audio:
  tts_rate: 220   # How fast to speak (words per minute)
```

## Performance

- Normal mode: 15-25 FPS
- Fast mode: 20-30 FPS (use `./run_orby.sh --fast`)

Your Mac's chip makes it faster:
- M1/M2/M3/M4: 5x faster than normal computers
- Older Macs: Still works, just slower

## Testing

```bash
# Test if everything works
python3 test_production_systems.py
```

## Safety Notice

**Important:** OrbyGlasses helps you navigate, but it's not a replacement for your white cane or guide dog. Always use them together, especially when learning.

This is version 0.9 (beta). We recommend using it with someone nearby while testing.

## Current Status

What works:
- ✅ Object detection
- ✅ Distance measurement
- ✅ Audio warnings
- ✅ Indoor tracking
- ✅ Emergency stop

What's being improved:
- Audio response time (currently 0.5-1 second)
- Battery usage on laptops

## Troubleshooting

**Camera not working:**
```bash
# Check if camera is available
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'Not working')"
```

**No audio:**
```bash
# Test your speakers
say "Testing audio"
```

**Too slow:**
- Use fast mode: `./run_orby.sh --fast`
- Lower camera quality in config.yaml
- Close other programs

## Getting Help

- Problems: https://github.com/jaskirat1616/Orby-Glasses/issues
- Questions: https://github.com/jaskirat1616/Orby-Glasses/discussions

## Privacy

All processing happens on your computer. Nothing is sent to the internet.

## License

Free to use and modify (GPL-3.0). See LICENSE file.

## Built With

- Object detection: YOLOv11
- Distance measurement: Depth Anything V2
- Indoor tracking: pySLAM
- Audio: macOS built-in speech

---

For detailed setup, see [SETUP.md](SETUP.md)
For common problems, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
