# OrbyGlasses - Project Structure

Simple, organized structure for easy navigation.

## Directory Layout

```
OrbyGlasses/
├── src/                    # Source code
│   ├── core/              # Essential modules
│   │   ├── detection.py       # Object detection & depth
│   │   ├── utils.py           # Config, logging, audio
│   │   ├── echolocation.py    # Audio beaconing
│   │   ├── narrative.py       # AI narration
│   │   ├── smart_cache.py     # Performance caching
│   │   ├── safety_system.py   # Safety checks
│   │   ├── object_tracker.py  # Object tracking
│   │   └── error_handler.py   # Error handling
│   │
│   ├── navigation/        # SLAM & path planning
│   │   ├── slam_system.py     # Main SLAM system
│   │   ├── slam_tracking.py   # Feature tracking
│   │   ├── slam_map_viewer.py # Map visualization
│   │   └── indoor_navigation.py # A* path planning
│   │
│   ├── visualization/     # Display & UI
│   │   └── robot_ui.py        # Main UI overlay
│   │
│   ├── features/          # Optional features (can disable)
│   │   ├── conversation.py           # Voice commands
│   │   ├── trajectory_prediction.py  # GNN prediction
│   │   ├── scene_understanding.py    # VLM scene analysis
│   │   ├── occupancy_grid_3d.py      # 3D voxel mapping
│   │   ├── point_cloud_viewer.py     # 3D point clouds
│   │   ├── movement_visualizer.py    # Movement tracking
│   │   ├── coordinate_transformer.py # Coordinate systems
│   │   ├── mapping3d.py              # 3D mapping
│   │   ├── prediction.py             # RL prediction
│   │   └── social_navigation.py      # Social norms
│   │
│   └── main.py            # Main entry point
│
├── config/                # Configuration
│   └── config.yaml        # Simple settings file
│
├── docs/                  # Documentation
│   ├── OPTIMIZATION_2024.md  # Latest improvements
│   └── PROJECT_STRUCTURE.md  # This file
│
├── data/                  # Runtime data
│   ├── logs/              # Log files
│   └── maps/              # Saved SLAM maps
│
├── models/                # AI models
│   ├── yolo/              # YOLOv11 weights
│   ├── depth/             # Depth Pro cache
│   └── rl/                # RL models (if used)
│
├── tests/                 # Unit tests
│
├── run.sh                 # Start script
└── README.md              # Main documentation
```

## Core Modules (Always Used)

**detection.py** - Object detection (YOLOv11n) + depth estimation (Depth Pro)
**utils.py** - Configuration loading, logging, audio management
**slam_system.py** - Visual SLAM for indoor navigation
**robot_ui.py** - Clean UI with depth zones and safety arrows
**main.py** - Main application logic

## Optional Features (Can Disable in config.yaml)

All files in `src/features/` are optional and disabled by default.
Enable them in `config.yaml` if needed.

## Removed Duplicates

- `main_simple.py` - Duplicate of main.py
- `slam.py` - Using slam_system.py instead
- `minimal_ui.py` - Using robot_ui.py
- `demo_overlay.py` - Not needed
- `blind_navigation.py` - Merged into main.py
- `audio_priority.py` - Simple logic, inlined

## Key Files to Edit

- `config/config.yaml` - Change settings
- `src/main.py` - Main logic
- `src/core/detection.py` - Detection/depth
- `src/navigation/slam_system.py` - SLAM
- `src/visualization/robot_ui.py` - UI overlay

## Simple Structure Benefits

1. **Easy to find** - Organized by purpose
2. **Clear separation** - Core vs optional
3. **No duplicates** - One file per purpose
4. **Simple names** - Self-explanatory
5. **Clean imports** - Grouped by category
