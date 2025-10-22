# Tests Directory

This directory contains all the test files for the OrbyGlasses project. Each test file focuses on a specific feature or component of the system.

## Test Files Overview

### Audio & Speech Tests
- **test_audio_complete.py** - Tests that audio sentences complete fully with a 5-second interval to prevent cutoff
- **test_audio_fix.py** - Tests fixes for audio system issues, particularly related to speech completion and queue management
- **test_microphone.py** - Tests microphone access and basic functionality for speech recognition
- **test_ollama_audio.py** - Tests integration between Ollama (AI model) and the audio system

### Navigation & Mapping Tests
- **test_slam.py** - Unit tests for the SLAM (Simultaneous Localization and Mapping) system, including map point creation, tracking, and path planning
- **test_enhanced_slam.py** - Tests enhanced SLAM functionality with improved mapping and navigation features
- **test_occupancy_grid_3d.py** - Tests 3D occupancy grid mapping for environment representation
- **test_trajectory_prediction.py** - Tests trajectory prediction algorithms for path planning

### Computer Vision Tests
- **test_detection.py** - Tests object detection functionality for identifying people, obstacles, and other objects
- **test_echolocation.py** - Tests echolocation-like computer vision algorithms for obstacle detection
- **test_interactive_grid.py** - Tests interactive grid systems for user interaction with the mapping system

### Social & Interaction Tests
- **test_social_navigation.py** - Tests social navigation AI that provides guidance based on cultural norms and crowd behavior
- **test_social_navigation_integration.py** - Tests integration between social navigation and other navigation systems
- **test_conversation_debug.py** - Tests conversation debugging features for the voice interaction system

### Integration & Utility Tests
- **test_integration.py** - Tests integration between different system components (audio, vision, navigation)
- **test_improved_prompts.py** - Tests improved prompt engineering for better AI responses
- **test_utils.py** - Tests utility functions and helper classes used throughout the system

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run a specific test file:
```bash
pytest tests/test_slam.py
```

To run with verbose output:
```bash
pytest tests/ -v
```

## Test Categories

- **Unit Tests**: Test individual components in isolation (e.g., test_slam.py, test_detection.py)
- **Integration Tests**: Test interaction between multiple components (e.g., test_integration.py)
- **Performance Tests**: Benchmark system performance and responsiveness (various files)
- **Feature Tests**: Test specific features like audio, navigation, or social behavior (categorized by prefix)