# SLAM and Indoor Navigation for OrbyGlasses

## Overview

OrbyGlasses now includes **Visual SLAM (Simultaneous Localization and Mapping)** for indoor navigation. This allows the system to:

1. **Track your position** in indoor spaces (no GPS needed)
2. **Build maps** of your environment as you explore
3. **Navigate to goals** using voice commands like "Take me to the kitchen"
4. **Remember locations** for future navigation
5. **Plan paths** around obstacles using A* algorithm

## What is SLAM?

**SLAM = Simultaneous Localization and Mapping**

Think of it like this:
- **Localization**: "Where am I?" (Your position in the building)
- **Mapping**: "What does this place look like?" (Building a map as you walk)
- **Simultaneous**: Doing both at the same time!

### Why is this useful for blind users?

**Without SLAM:**
- "There's a chair 2 meters ahead"
- Reactive navigation (responds to immediate obstacles)

**With SLAM:**
- "You are in the hallway, 5 meters from the kitchen"
- "Turn left in 3 meters, then the bathroom is on your right"
- Goal-oriented navigation (helps you reach destinations)

## Features

### 1. Visual Odometry
- Tracks your movement using only the camera (no IMU needed!)
- Works with your simple USB webcam
- Estimates position in 3D space

### 2. Indoor Mapping
- Builds a 2D occupancy grid of your environment
- Remembers obstacles and clear paths
- Saves maps for reuse (no need to re-map every time)

### 3. Goal-Based Navigation
- Voice command: "Hey Orby, take me to the bathroom"
- System plans a path and guides you step-by-step
- Replans if obstacles appear in the path

### 4. Location Memory
- Voice command: "Hey Orby, remember this as the kitchen"
- System saves the location
- Later: "Hey Orby, take me to the kitchen"

## How It Works

### Step 1: Feature Detection
The camera detects unique visual features (corners, edges) in each frame:
```
Frame 1: [Feature A at (10,20), Feature B at (50,60), ...]
Frame 2: [Feature A at (12,21), Feature B at (52,61), ...]
         → Camera moved 2 pixels right, 1 pixel forward
```

### Step 2: Pose Estimation
By tracking how features move, we estimate camera position:
```
Frame 1: Position (0, 0, 0)
Frame 2: Position (0.1, 0.05, 0.0) → Moved 10cm right, 5cm forward
Frame 3: Position (0.25, 0.12, 0.0)
```

### Step 3: Map Building
As you move, the system builds a 2D grid map:
```
Occupancy Grid (top view):
. . . . . . . .
. # # . . . . .    # = obstacle
. # # . . . . .    . = free space
. . . X . . . .    X = your position
. . . . . . . .    G = goal
. . . G . . . .
```

### Step 4: Path Planning
When you set a goal, A* algorithm finds the best path:
```
. . . . . . . .
. # # . . . . .
. # # 3 2 1 . .    Numbers show the path
. . . X 4 5 6 .    from your position (X)
. . . . . 7 G .    to the goal (G)
```

### Step 5: Navigation Guidance
System gives turn-by-turn directions:
```
"Continue straight for 3 meters"
"Turn left in 2 meters"
"Bathroom is ahead on your right, 1 meter"
"Arrived at bathroom"
```

## Usage

### Basic SLAM Tracking

```python
from src.slam import MonocularSLAM
from src.utils import ConfigManager

# Initialize
config = ConfigManager("config/config.yaml")
slam = MonocularSLAM(config)

# Process each frame
for frame in camera_frames:
    result = slam.process_frame(frame)

    print(f"Position: {result['position']}")
    print(f"Tracking quality: {result['tracking_quality']}")
    print(f"Map points: {result['num_map_points']}")
```

### Indoor Navigation with Goals

```python
from src.slam import MonocularSLAM
from src.indoor_navigation import IndoorNavigator

# Initialize
slam = MonocularSLAM(config)
navigator = IndoorNavigator(slam, config)

# Save current location
navigator.save_location("kitchen")

# Later, navigate to it
navigator.set_goal("kitchen")

# Get guidance
guidance = navigator.get_navigation_guidance()
print(guidance)  # "Turn left in 3 meters toward kitchen"
```

### Voice Control Integration

With conversational navigation enabled:

```
User: "Hey Orby, remember this as the bathroom"
Orby: "Location saved: bathroom"

[User walks to another room]

User: "Hey Orby, take me to the bathroom"
Orby: "Navigating to bathroom. Continue straight for 5 meters."

[User walks 5 meters]

Orby: "Turn right in 2 meters."

[User turns right]

Orby: "Bathroom is ahead on your left, 1 meter."
Orby: "Arrived at bathroom."
```

## Configuration

In `config/config.yaml`:

```yaml
# SLAM Settings
slam:
  enabled: true                   # Enable SLAM
  grid_size: [200, 200]           # Map size (20m x 20m with 0.1m cells)
  grid_resolution: 0.1            # 10cm per grid cell
  save_maps: true                 # Save maps for later
  visualize: true                 # Show SLAM window

# Indoor Navigation
indoor_navigation:
  enabled: true                   # Enable navigation
  path_planning: true             # Enable A* pathfinding
  save_locations: true            # Remember named locations
```

## Visualization

When `slam.visualize: true`, you'll see:

### SLAM Tracking Window
- Green dots: Detected features
- Position: Your current 3D coordinates
- Tracking quality: 0-1 score (higher = better)
- Map points: Number of landmarks in the map

### Occupancy Grid Window
- Black: Occupied (obstacles)
- White: Free space
- Green dot: Your position
- Red dot: Goal (if set)
- Blue line: Planned path

## Performance

### Computational Cost
- **SLAM processing**: ~30ms per frame
- **Path planning**: ~10ms (only when needed)
- **Total overhead**: ~40ms
- **Still real-time**: Can run at 20+ FPS

### Memory Usage
- **Map storage**: ~5MB for typical indoor space
- **Trajectory history**: ~1MB
- **Total**: ~6MB additional memory

## Limitations & Challenges

### Current Limitations

1. **Monocular SLAM drift**: Position estimates accumulate error over time
   - **Solution**: Loop closure detection (planned)
   - **Workaround**: Re-initialize periodically

2. **Feature-poor environments**: Blank walls are hard to track
   - **Solution**: Use in textured environments
   - **Future**: Add sensor fusion (ultrasonic)

3. **Scale ambiguity**: Monocular can't determine absolute scale
   - **Current**: Assumes average human walking speed
   - **Future**: IMU integration for scale correction

4. **2D navigation only**: Doesn't handle stairs/elevators yet
   - **Future**: Multi-floor mapping

### Best Practices

**For Best SLAM Performance:**
- ✓ Well-lit environments
- ✓ Textured walls (pictures, patterns)
- ✓ Steady head movement
- ✓ Regular key framing

**Avoid:**
- ✗ Very dark rooms
- ✗ Completely blank walls
- ✗ Rapid head movements
- ✗ Extreme close-ups (< 0.3m)

## Saving & Loading Maps

### Save Map
```python
# After exploring an environment
slam.save_map("my_home.json")
# Saved to: data/maps/my_home.json
```

### Load Map
```python
# On next session
slam.load_map("data/maps/my_home.json")
# System now knows the environment
# Can navigate immediately without re-mapping
```

### Map File Format
```json
{
  "timestamp": 1234567890,
  "num_keyframes": 42,
  "num_map_points": 1537,
  "keyframes": [...],
  "map_points": [...],
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
}
```

## Troubleshooting

### "Lost tracking: only X matches"
**Problem**: Not enough visual features detected
**Solutions**:
- Improve lighting
- Move to area with more visual texture
- Slow down head movement
- Re-initialize SLAM

### "No path found"
**Problem**: Path planner can't find route to goal
**Solutions**:
- Check if goal is reachable (not behind closed door)
- Ensure occupancy grid is updated
- Try replanning with `navigator._plan_path()`

### "SLAM position drifting"
**Problem**: Position estimates becoming inaccurate
**Solutions**:
- Re-initialize SLAM periodically
- Save checkpoints at known locations
- Enable loop closure (future feature)

## Future Enhancements

### Planned Features

1. **Loop Closure Detection**
   - Recognize when you return to a known location
   - Correct accumulated drift
   - More accurate long-term tracking

2. **Multi-Floor Mapping**
   - Detect stairs/elevators
   - Build 3D maps of buildings
   - Navigate across floors

3. **Collaborative SLAM**
   - Multiple users share maps
   - Crowdsourced indoor mapping
   - Public building databases

4. **IMU Fusion** (if hardware added)
   - More accurate scale estimation
   - Better handling of fast movements
   - Drift correction

5. **Semantic Mapping**
   - Label room types (kitchen, bathroom, bedroom)
   - Understand door locations
   - Detect stairs, elevators

## Research & Publications

This SLAM system implements concepts from:

- **ORB-SLAM** (Mur-Artal et al., 2015)
- **Monocular Visual Odometry** (Scaramuzza & Fraundorfer, 2011)
- **A* Path Planning** (Hart et al., 1968)

### Potential Paper Topics

1. "Monocular SLAM for Assistive Navigation in Visually Impaired Users"
2. "Goal-Oriented Indoor Navigation for Blind Users Using Visual SLAM"
3. "User Study: Impact of SLAM-Based Navigation on Blind Independence"

## Contributing

To improve SLAM performance:

1. **Collect data**: Run in diverse environments, save maps
2. **Report issues**: What environments fail? When does tracking break?
3. **Propose features**: What navigation features would help most?

## License

Same as OrbyGlasses main project (MIT)

---

**SLAM transforms OrbyGlasses from obstacle avoidance to true indoor navigation.**
