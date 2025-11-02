# Fixing Common Problems

Quick solutions for OrbyGlasses issues.

## Camera Problems

### Camera Not Working

**Symptoms:** Error message "Could not open camera" or black screen

**Solutions:**

1. **Check if camera exists:**
   ```bash
   python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Works' if cap.isOpened() else 'Not working'); cap.release()"
   ```

2. **Try different camera numbers:**
   ```bash
   # Edit config.yaml and try: 0, 1, or 2
   camera:
     source: 1  # Change this number
   ```

3. **Give permission:**
   - Go to System Settings → Privacy & Security → Camera
   - Turn on for Terminal

4. **Close other programs using camera** (Zoom, FaceTime, etc.)

### Camera Shows Black Screen

**Solutions:**

1. **Try lower quality:**
   ```yaml
   camera:
     width: 320   # Instead of 640
     height: 240  # Instead of 480
   ```

2. **Update macOS** to latest version

3. **Restart computer**

### Poor Image Quality

**Solutions:**

1. **Increase camera quality:**
   ```yaml
   camera:
     width: 640
     height: 480
   ```

2. **Turn on lights** - system needs good lighting

3. **Clean camera lens**

## Navigation Problems

### System Stops Tracking Position

**Symptoms:** "Tracking lost" messages

**Solutions:**

1. **Move slowly** - fast movement causes problems

2. **Add visual features:**
   - Blank walls don't work well
   - Put up posters or furniture

3. **Improve lighting** - avoid dark rooms

4. **Adjust settings:**
   ```yaml
   slam:
     orb_features: 8000  # Increase from 5000
   ```

## Speed Problems

### Too Slow (Less than 10 FPS)

**Solutions:**

1. **Use fast mode:**
   ```bash
   ./run_orby.sh --fast
   ```

2. **Lower camera quality:**
   ```yaml
   camera:
     width: 320
     height: 240
   ```

3. **Turn off extra features:**
   ```yaml
   slam:
     visualization: false  # Don't show 3D view

   features:
     mapping3d: false
     occupancy_grid_3d: false
   ```

4. **Close other programs** to free up computer resources

### High CPU or Memory Usage

**Solutions:**

1. **Check what's using resources:**
   ```bash
   top -pid $(pgrep -f main.py)
   ```

2. **Limit map size:**
   ```yaml
   slam:
     max_keyframes: 100
     max_map_points: 5000
   ```

3. **Restart OrbyGlasses**

## Audio Problems

### No Sound

**Solutions:**

1. **Test system audio:**
   ```bash
   say "Testing audio"
   ```

2. **Check speakers:**
   - System Settings → Sound → Output
   - Make sure correct device is selected

3. **Check if audio is turned on:**
   ```yaml
   audio:
     enabled: true
   ```

4. **Reinstall audio:**
   ```bash
   pip uninstall pyttsx3
   pip install pyttsx3
   ```

### Audio Too Slow

**Solutions:**

1. **Speed up speech:**
   ```yaml
   audio:
     tts_rate: 240  # Increase from 220
   ```

2. **Use fast mode:**
   ```bash
   ./run_orby.sh --fast
   ```

### Audio Cuts Out

**Solutions:**

1. **Reduce how often it talks:**
   ```yaml
   audio:
     min_time_between_warnings: 3.0  # Increase from 2.0
   ```

## Installation Problems

### Python Not Found

**Solution:**
```bash
brew install python@3.11
```

### pip install Fails

**Solution:**
```bash
# Use virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### pySLAM Won't Install

**Solution:**
```bash
# Install build tools
xcode-select --install

# Install dependencies
brew install cmake opencv eigen boost

# Try again
./install_pyslam.sh
```

## Mac-Specific Problems

### "Cannot open because developer cannot be verified"

**Solution:**
```bash
xattr -d com.apple.quarantine run_orby.sh
chmod +x run_orby.sh
```

### "Operation not permitted"

**Solution:**
- System Settings → Privacy & Security → Full Disk Access
- Add Terminal

## Still Having Problems?

### Reset Everything

```bash
# Remove saved data
rm -rf data/maps/* data/logs/*

# Restart OrbyGlasses
./run_orby.sh
```

### Get Help

1. **Check logs:**
   ```bash
   tail -f data/logs/orbyglass.log
   ```

2. **Report problem:**
   - Go to: https://github.com/jaskirat1616/Orby-Glasses/issues
   - Include:
     - Your Mac model (M1, M2, etc.)
     - macOS version
     - Python version
     - Error message
     - What you were doing

3. **Ask questions:**
   - Go to: https://github.com/jaskirat1616/Orby-Glasses/discussions

## Quick Fixes

```bash
# Check if everything is installed
python3 test_production_systems.py

# Reset OrbyGlasses
git checkout config/config.yaml
./run_orby.sh

# Reinstall everything
pip install -r requirements.txt --force-reinstall
./install_pyslam.sh
```
