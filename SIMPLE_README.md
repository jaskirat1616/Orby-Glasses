# Simplified OrbyGlasses

A lightweight version of the OrbyGlasses bio-mimetic navigation system for visually impaired users, without SLAM, 3D mapping, occupancy maps, and point clouds for faster performance and simpler setup.

## Features

- Real-time object detection using YOLOv11
- Real depth estimation using Depth Anything V2
- Audio feedback for navigation guidance
- Accurate distance measurements using depth maps
- Danger and caution zone alerts
- Simple text-to-speech guidance
- No complex 3D mapping or SLAM required

## Requirements

- Python 3.8+
- macOS (for text-to-speech) or Linux/Windows with pyttsx3
- Camera/webcam
- Ollama for local LLM processing

## Installation

1. Run the setup script:
```bash
./setup_simple.sh
```

2. Make sure Ollama is running:
```bash
ollama serve
```

3. In a separate terminal, start the application:
```bash
python simple_orbyglasses.py
```

## Usage

- The application will start processing video from your default camera
- Audio guidance will be provided through your system speakers
- Objects are detected and displayed with bounding boxes
- Danger alerts are provided for objects within 0.4m
- Caution alerts for objects within 1.5m
- Press 'q' to quit the application

## Configuration

The application uses hardcoded configuration for simplicity. You can modify the `ConfigManager` class in the `simple_orbyglasses.py` file to adjust parameters like:

- Detection confidence thresholds
- Audio update intervals
- Safety distances
- Camera settings

## Key Differences from Full Version

- Removed SLAM functionality
- Removed 3D mapping and point clouds
- Removed occupancy grid mapping
- Simplified audio system using macOS 'say' command
- Real depth estimation using Depth Anything V2 instead of simple size-based estimation
- Reduced computational requirements (depth estimation runs every Nth frame)
- Faster startup and response times