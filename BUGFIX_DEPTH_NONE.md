# Bug Fix: Depth None Handling

## Issue

When running with depth model disabled (`models.depth.enabled: false`), the audio cue generator crashed with:

```
TypeError: '<' not supported between instances of 'NoneType' and 'NoneType'
```

**Location**: `src/core/echolocation.py:518`

## Root Cause

The `min()` function was trying to compare `None` depth values when the depth model was disabled. Since depth estimation was turned off for performance, all detections had `depth: None`, causing the comparison to fail.

```python
# OLD CODE (broken):
closest = min(danger_objects, key=lambda x: x.get('depth', 10))
# If x.get('depth') returns None, min() cannot compare None values
```

## Fix Applied

Updated the lambda function to handle `None` depths gracefully:

```python
# NEW CODE (fixed):
closest = min(danger_objects, key=lambda x: x.get('depth') if x.get('depth') is not None else 999)
depth = closest.get('depth')
if depth is not None:
    message = f"Warning! {closest['label']} {depth:.1f} meters ahead"
else:
    message = f"Warning! {closest['label']} ahead - distance unknown"
```

**Key changes:**
1. Use `999` as fallback value for `None` depths in sorting
2. Check if depth is `None` before formatting message
3. Provide fallback message when depth is unknown

## Files Modified

- `src/core/echolocation.py` (lines 517-531)

## Testing

```bash
# Should now work without errors:
./run_orby.sh --video /path/to/video.mp4 --show-features

# Expected audio messages (no depth model):
# - "Warning! person ahead - distance unknown"
# - "Caution. car detected"
# - "3 objects detected. Path clear"
```

## Impact

✅ **Fixed**: System no longer crashes when depth model is disabled
✅ **Improved**: Better error handling for missing depth data
✅ **User-friendly**: Informative messages even without depth info

## Related Configuration

This fix is particularly important for the speed-optimized configuration:

```yaml
# config/config.yaml (speed mode)
models:
  depth:
    enabled: false  # Disabled for 2-3x FPS gain

# Now works correctly with this configuration
```

## Future Improvements

Potential enhancements:
- [ ] Use bounding box size to estimate relative distance
- [ ] Use SLAM map points for coarse distance estimates
- [ ] Cache last known depth for tracked objects

## Commit Message

```
fix: handle None depth values in audio cue generator

- Add None check in min() lambda for safe comparison
- Provide fallback messages when depth unavailable
- Fixes crash when depth model is disabled
- Improves error handling in echolocation.py:517-531
```
