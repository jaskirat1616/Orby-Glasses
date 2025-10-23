# OrbyGlasses - Quick Start

## Launch in 10 Seconds

```bash
./run.sh
```

That's it! 🚀

---

## What You'll See

### 3 Windows
1. **Robot Vision** - Main camera with clean overlays
2. **Depth Sensor** - Real-time depth map
3. **Navigation Map** - SLAM top-down view

### Audio Guidance
- "Path clear"
- "Stop. Car ahead. Go left"
- "Person ahead. Slow down"

Simple, clear, actionable.

---

## Performance

- **Target**: 30+ FPS (no SLAM)
- **Typical**: 25-35 FPS
- **With SLAM**: 15-20 FPS

---

## Controls

- **q** - Quit
- Mouse/keyboard work in windows

---

## Config (Optional)

Edit `config/config.yaml`:

```yaml
# Maximum speed
slam:
  enabled: false  # 30+ FPS

# Good balance
slam:
  enabled: true   # 15-20 FPS with mapping
```

---

## Troubleshooting

**Camera not found?**
- Check USB camera is connected
- Try `ls /dev/video*`

**Low FPS?**
- Disable SLAM in config
- Increase `depth_skip_frames`

**No audio?**
- Check Ollama is running: `ollama serve`
- Audio works without it (just simpler)

---

## Features

✓ Smart motion caching (2-3x faster)
✓ Predictive collision avoidance
✓ Robot-style UI
✓ Simple audio for blind users
✓ Production error handling

---

## Next Steps

1. **Test it**: Walk around, see it work
2. **Tune it**: Adjust config for your hardware
3. **Use it**: Help blind users navigate

---

## Need More?

- Full guide: `PRODUCTION_READY.md`
- Features: `BREAKTHROUGH_FEATURES.md`
- Code: `src/`

---

**Made with ❤️ for accessibility**
