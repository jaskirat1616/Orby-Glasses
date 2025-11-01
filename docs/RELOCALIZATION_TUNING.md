# Relocalization Tuning Guide

When SLAM loses tracking, relocalization tries to find where the camera is by matching current features against the map. Sometimes relocalization fails even when detecting good candidates.

## ✅ OrbyGlasses Auto-Tuning

**OrbyGlasses now automatically applies aggressive relocalization tuning** for real-world success:

- ✅ **20 inlier threshold** (down from 50) - More realistic for monocular SLAM
- ✅ **Relaxed PnP solver** - Uses chi-square threshold of 7.815 (98% confidence)
- ✅ **Larger search windows** - 15px coarse, 5px fine (finds more matches)
- ✅ **Lenient matching** - Ratio test 0.85-0.95 (more tolerant)

**You don't need to configure anything** - it's already optimized!

## Understanding Relocalization Failures

From the logs, you'll see:
```
Relocalizer: Detected candidates: 277 with [74, 76, 78, 69, 68, 72]
Relocalizer: num_matches (277,78): 157  ← Good number of matches!
Relocalizer: performing MLPnPsolver iterations for keyframe 78
Relocalizer: failed  ← But PnP solver fails
```

### Why Does This Happen?

**The process:**
1. ✅ Loop detector finds candidate keyframes (working)
2. ✅ Feature matching finds correspondences (157 matches - good!)
3. ❌ PnP solver can't find enough inliers (failing)

**Common causes:**
- Not enough features detected (1313 vs 5000 target)
- Matches aren't geometrically consistent
- Parameters too strict for real-world conditions
- Camera moved too far from original viewpoint

## Quick Fixes

### 1. Check Feature Count

Look for this in logs:
```
detector: ORB , #features: 1313 , [kp-filter: KDT_NMS ]
```

If below 2000, you have tracking issues:

**Fix in config.yaml:**
```yaml
slam:
  orb_features: 5000  # Should detect 2500-4000 in practice
```

**Check environment:**
- ✅ Well-textured surfaces (patterns, posters, furniture)
- ❌ Blank walls, uniform colors
- ✅ Good lighting (bright, even)
- ❌ Shadows, reflections, darkness

### 2. Relax Relocalization Parameters

OrbyGlasses now automatically uses relaxed parameters:

**Auto-applied in pyslam_live.py:**
```python
Parameters.kRelocalizationMinKpsMatches = 10         # Was 15
Parameters.kRelocalizationPoseOpt1MinMatches = 8     # Was 10
Parameters.kRelocalizationDoPoseOpt2NumInliers = 30  # Was 50
Parameters.kRelocalizationFeatureMatchRatioTest = 0.8  # Was 0.75
```

These make relocalization more tolerant of imperfect conditions.

### 3. Use IBOW Loop Detector

IBOW builds vocabulary incrementally and is more robust:

**Already enabled by default** in OrbyGlasses (falls back to DBOW3 if unavailable).

### 4. Prevent Tracking Loss

Better to avoid losing tracking:

**Move camera slowly:**
- 20-30 cm/second maximum
- Smooth, steady motion
- No rapid rotations

**Maintain overlap:**
- Keep 60%+ of previous view in frame
- Don't jump to completely new areas
- Gradual transitions

**Good environments:**
- Rich visual texture
- Good lighting
- Stable camera mount

## Advanced Tuning

If relocalization still fails frequently, manually adjust parameters.

### Edit Parameters Directly

**File:** `third_party/pyslam/pyslam/config_parameters.py`

```python
# Relocalization (line ~205)
kRelocalizationMinKpsMatches = 8  # Minimum matches to attempt relocalization
kRelocalizationPoseOpt1MinMatches = 6  # Minimum for first pose optimization
kRelocalizationDoPoseOpt2NumInliers = 25  # Threshold for second optimization
kRelocalizationFeatureMatchRatioTest = 0.85  # Match ratio (higher = more lenient)
kRelocalizationMaxReprojectionDistanceMapSearchCoarse = 15  # Pixels for coarse search
kRelocalizationMaxReprojectionDistanceMapSearchFine = 5  # Pixels for fine search
```

### Parameter Guidelines

**kRelocalizationMinKpsMatches** (default: 15 → 10)
- Minimum feature matches to attempt relocalization
- Lower = tries with fewer matches (more attempts, more false positives)
- Higher = only tries with strong evidence (fewer attempts, fewer false positives)
- Recommended: 8-12 for difficult environments

**kRelocalizationPoseOpt1MinMatches** (default: 10 → 8)
- Minimum matches for first pose optimization pass
- Lower = more lenient initial check
- Recommended: 6-10

**kRelocalizationDoPoseOpt2NumInliers** (default: 50 → 30)
- Inlier threshold to proceed with second optimization
- **This is often the bottleneck**
- Lower = accepts solutions with fewer geometrically consistent matches
- You're getting 157 total matches but probably 20-40 inliers
- Recommended: 25-35 for real-world use

**kRelocalizationFeatureMatchRatioTest** (default: 0.75 → 0.8)
- Lowe's ratio test for descriptor matching
- Higher values (closer to 1.0) = more lenient
- Range: 0.7-0.9
- Recommended: 0.8-0.85

## Debugging Relocalization

### Enable Debug Logging

**Edit config_parameters.py:**
```python
kRelocalizationDebugAndPrintToFile = True
```

Logs saved to: `third_party/pyslam/logs/relocalizer.log`

### Analyze Failure Patterns

**Look for:**
```
Relocalizer: num_matches (277,78): 157  ← Total matches
# ... PnP solver runs ...
# How many inliers were found? (not shown in standard output)
Relocalizer: failed
```

**If you consistently see:**
- Good matches (100+) but still fails → Lower `kRelocalizationDoPoseOpt2NumInliers`
- Few matches (<50) → Improve environment/features
- Random successes → Parameters are borderline, relax slightly

## Real-World Tips

### 1. Revisit Areas Multiple Times

Loop closure works better when you:
- Return to the same area from similar viewpoint
- Circle back frequently
- Create "anchor points" with distinctive features

### 2. Create "Relocalization Friendly" Maps

When mapping:
- **Start area**: Make it feature-rich (posters, objects)
- **Return often**: Circle back to start every 30-60 seconds
- **Vary viewpoints**: Look at same objects from different angles
- **Avoid**: Long corridors, blank walls, moving objects

### 3. Watch for RELOCALIZE State

```
state: RELOCALIZE  ← Tracking lost, trying to recover
```

When you see this:
- **Stop moving** immediately
- **Look at textured area** you've seen before
- **Slow pan** to give system time to match
- **Don't give up** - may take 5-10 frames

### 4. Keyframe Strategy

More keyframes = better relocalization:

**config.yaml:**
```yaml
slam:
  max_frames_between_keyframes: 15  # Create keyframes more frequently
```

Downside: Higher CPU usage, larger maps

## Monitoring Success Rate

### Good Relocalization Behavior

```
Relocalizer: Detected candidates: 5
Relocalizer: num_matches (150,145): 234
Relocalizer: performing MLPnPsolver iterations for keyframe 145
Relocalization successful, last reloc frame id: 150
```

### Poor Relocalization Behavior

```
Relocalizer: Detected candidates: 12  ← Many candidates
Relocalizer: num_matches (all): <50   ← But few matches
Relocalizer: failed                   ← Consistent failure
```

**Diagnosis:** Environment too uniform, need more features.

## When to Reset vs Persist

### Reset if:
- Tracking lost for >30 seconds
- Moved to completely new area
- Lighting changed dramatically
- Environment changed (objects moved)

### Keep trying if:
- Lost tracking briefly (<10 seconds)
- Still in same general area
- Good features still visible
- Relocalization attempts show matches (even if failing)

## See Also

- [SLAM Troubleshooting](SLAM_TRACKING_TROUBLESHOOTING.md) - Prevent tracking loss
- [Normal Environment Guide](NORMAL_ENVIRONMENT_GUIDE.md) - Environment optimization
- [pySLAM Modifications](PYSLAM_MODIFICATIONS.md) - System compatibility changes
