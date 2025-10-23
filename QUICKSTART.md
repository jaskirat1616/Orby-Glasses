# OrbyGlasses - Quick Start

## Installation

```bash
# Clone repository
git clone https://github.com/jaskirat1616/Orby-Glasses.git
cd Orby-Glasses

# Run setup (first time only)
chmod +x setup.sh
./setup.sh
```

---

## Running

### Simple Version (Recommended)
**Fast, accurate, easy to use**

```bash
./run.sh
```

**Features**:
- Object detection
- Depth estimation
- Safety warnings
- Audio guidance
- SLAM navigation
- 15-20 FPS

**Perfect for**: Daily use by blind users

---

### Full Version
**All features, slower**

```bash
./start.sh
```

**Additional features**:
- Trajectory prediction
- 3D mapping
- Scene understanding
- Conversation system
- 12-15 FPS

**Perfect for**: Testing and development

---

## Controls

- **Press 'q'**: Emergency stop
- **Voice command**: "hey orby" (full version only)

---

## What It Does

1. **Detects objects** around you (people, cars, chairs, doors, etc.)
2. **Measures distances** accurately (uses object sizes + depth camera)
3. **Warns about dangers**:
   - < 0.4m: "STOP! Object ahead"
   - < 1.0m: "Danger: Chair on your left"
   - < 2.0m: "Caution: Person 1.5m away"
4. **Guides with voice** (smart priority, no repetition)
5. **Tracks position** indoors (SLAM navigation)
6. **Remembers locations** you save

---

## Troubleshooting

### Camera Not Found
```bash
# Try different camera index
# Edit config/fast.yaml:
camera:
  source: 1  # Change from 0 to 1
```

### Low FPS
```bash
# Edit config/fast.yaml:
performance:
  depth_skip_frames: 4  # Skip more frames
slam:
  enabled: false  # Disable SLAM
```

### No Audio
- Check speakers/headphones connected
- Check volume is up
- Check microphone permissions (for voice commands)

---

## Files

- `run.sh` - Simple version (recommended)
- `start.sh` - Full version
- `config/fast.yaml` - Fast settings
- `config/best.yaml` - Best settings
- `SIMPLE_PIPELINE.md` - Technical details
- `SUMMARY.md` - Complete overview

---

## Next Steps

1. **Test it**: Walk around with the system
2. **Adjust settings**: Edit `config/fast.yaml`
3. **Read docs**: Check `SIMPLE_PIPELINE.md`
4. **Give feedback**: Report issues on GitHub

---

## Support

- GitHub: https://github.com/jaskirat1616/Orby-Glasses
- Issues: https://github.com/jaskirat1616/Orby-Glasses/issues
- Docs: See `SIMPLE_PIPELINE.md` and `SUMMARY.md`
