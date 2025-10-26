# OrbyGlasses User Guide

This guide helps you get started with OrbyGlasses and use it effectively.

## First Time Setup

1. **Install the system:**
   ```bash
   cd OrbyGlasses
   ./setup.sh
   ```
   This takes 10-15 minutes and downloads required AI models.

2. **Start OrbyGlasses:**
   ```bash
   ./run.sh
   ```

## Daily Use

1. **Start the system:**
   ```bash
   cd OrbyGlasses
   ./run.sh
   ```

2. **Listen for audio guidance:**
   - The system speaks directions and warnings
   - Danger alerts are immediate and loud
   - Regular updates every 1-2 seconds

3. **Stop the system:**
   - Press `q` key
   - Or close the terminal window

## Performance Modes

### Standard Mode (Default)
```bash
./run.sh
```
- Full features
- Best accuracy
- 15-25 FPS
- Indoor navigation enabled

### Fast Mode
```bash
./run.sh --fast
```
- Core features only
- Better performance
- 20-30 FPS
- Reduced battery usage

## Understanding Audio Alerts

**Immediate Danger (<1 meter):**
- "Stop. Car ahead. Go left."
- Speaks every 0.25 seconds
- High priority

**Caution Zone (1-2.5 meters):**
- "Chair on your right. Two meters away."
- Speaks every 1.2 seconds
- Medium priority

**Safe Zone (>2.5 meters):**
- "Path is clear."
- Occasional updates
- Low priority

## Using Indoor Navigation

**Save a location:**
1. Stand at the location you want to save
2. Wait for the system to map the area
3. The system automatically saves map data

**Navigate to a saved location:**
Indoor navigation is automatic when SLAM is enabled.

## Adjusting Settings

Edit `config/config.yaml`:

**Adjust danger distances:**
```yaml
safety:
  danger_distance: 1.0        # Immediate danger
  min_safe_distance: 1.5      # Caution zone
  caution_distance: 2.5       # Safe zone
```

**Change audio speed:**
```yaml
audio:
  tts_rate: 190               # Words per minute
  tts_volume: 1.0             # 0.0 to 1.0
```

**Improve detection accuracy:**
```yaml
models:
  yolo:
    confidence: 0.6           # Higher = fewer false positives
```

**Reduce CPU usage:**
```yaml
performance:
  depth_skip_frames: 3        # Process every 4th frame
  max_detections: 5           # Track fewer objects
```

## Common Issues

### "No camera found"
- Check camera is connected
- Test with: `python3 -c "import cv2; cap = cv2.VideoCapture(0); print('OK' if cap.isOpened() else 'FAIL')"`
- Try changing `camera: source: 0` to `1` in config

### "Ollama not running"
- Start Ollama: `ollama serve`
- Or restart the system with `./run.sh`

### "Poor performance / Low FPS"
- Use fast mode: `./run.sh --fast`
- Reduce camera resolution in config
- Close other applications
- Check CPU usage with Activity Monitor

### "Audio not working"
- Check speaker volume
- Check audio settings in System Preferences
- Verify `tts_volume: 1.0` in config

### "Inaccurate distances"
- Lighting is important - avoid very dark/bright areas
- Objects need to be in camera view
- Some surfaces (glass, mirrors) may confuse depth estimation

## Battery Tips

For longer battery life:
1. Use fast mode: `./run.sh --fast`
2. Reduce camera resolution
3. Disable features you don't need in config
4. Close other applications

## Safety Tips

1. **Use with other mobility aids:**
   OrbyGlasses assists navigation but doesn't replace white canes or guide dogs.

2. **Test in safe environments first:**
   Practice at home before using outdoors.

3. **Check battery level:**
   System requires significant power.

4. **Good lighting helps:**
   System works best in normal indoor/outdoor lighting.

5. **Audio clarity:**
   Use headphones in noisy environments.

## Privacy

- All processing happens on your computer
- No data sent to the internet
- Camera feed is not recorded or saved
- Maps are stored locally in `data/maps/`

## Technical Support

**Check logs for errors:**
```bash
cat data/logs/orbyglass.log
```

**Run tests:**
```bash
pytest tests/ -v
```

**Verify installation:**
```bash
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
```

## Updates

To update OrbyGlasses:
```bash
git pull
./setup.sh
```

## Keyboard Controls

- `q` - Quit the application
- Arrow keys - Control SLAM view (if visualized)

## File Locations

- **Configuration:** `config/config.yaml`
- **Logs:** `data/logs/orbyglass.log`
- **Maps:** `data/maps/`
- **Models:** `models/yolo/` and `models/depth/`

## Getting Help

1. Check this user guide
2. Read README.md for technical details
3. Check the logs in `data/logs/`
4. Run tests to diagnose issues

---

For developers and contributors, see README.md for architecture details and contribution guidelines.
