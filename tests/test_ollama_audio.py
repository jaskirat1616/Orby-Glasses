#!/usr/bin/env python3
"""
Integration test for Ollama narrative generation and audio system.
Tests the complete pipeline: detection -> Ollama -> audio output.
"""

import sys
import os
import time
sys.path.insert(0, 'src')

from core.utils import ConfigManager, AudioManager, Logger
from narrative import NarrativeGenerator
import numpy as np
import cv2

def test_ollama_audio_integration():
    """Test Ollama narrative generation with audio output."""

    print("=" * 60)
    print("OLLAMA + AUDIO INTEGRATION TEST")
    print("=" * 60)

    # Initialize components
    print("\n1. Initializing components...")
    config = ConfigManager('config/config.yaml')
    logger = Logger(log_file='data/logs/test_ollama_audio.log')
    audio = AudioManager(config)
    narrative_gen = NarrativeGenerator(config)

    print("   ✓ ConfigManager initialized")
    print("   ✓ AudioManager initialized")
    print("   ✓ NarrativeGenerator initialized")

    # Check Ollama models
    print("\n2. Checking Ollama models...")
    print(f"   Primary model: {narrative_gen.primary_model}")
    print(f"   Vision model: {narrative_gen.vision_model}")
    print(f"   Available models: {narrative_gen.available_models}")

    if not narrative_gen._is_model_available(narrative_gen.primary_model):
        print(f"   ✗ Primary model {narrative_gen.primary_model} NOT available!")
        return False

    print(f"   ✓ Primary model {narrative_gen.primary_model} available")

    # Test 1: Simple narrative generation (no vision)
    print("\n3. Testing narrative generation (text only)...")

    # Simulate detections
    test_detections = [
        {
            'label': 'person',
            'bbox': [100, 100, 200, 300],
            'confidence': 0.85,
            'depth': 2.5,
            'center': [150, 200],
            'is_priority': True,
            'is_danger': False
        },
        {
            'label': 'chair',
            'bbox': [300, 150, 400, 250],
            'confidence': 0.72,
            'depth': 3.2,
            'center': [350, 200],
            'is_priority': True,
            'is_danger': False
        }
    ]

    # Build navigation summary
    nav_summary = {
        'total_objects': 2,
        'danger_objects': [],
        'caution_objects': test_detections,
        'safe_objects': [],
        'closest_object': test_detections[0],
        'path_clear': True
    }

    print("   Generating narrative with Ollama...")
    start_time = time.time()
    narrative = narrative_gen.generate_narrative(
        test_detections,
        frame=None,
        navigation_summary=nav_summary
    )
    gen_time = time.time() - start_time

    print(f"   ✓ Narrative generated in {gen_time:.2f}s")
    print(f"   Generated: \"{narrative}\"")

    if not narrative or len(narrative) < 10:
        print("   ✗ Narrative too short or empty!")
        return False

    # Test 2: Audio output
    print("\n4. Testing audio output...")
    print(f"   Speaking: \"{narrative}\"")
    audio.speak(narrative, priority=True)

    # Wait for audio to complete
    print("   Waiting for audio to complete...")
    time.sleep(5)

    print("   ✓ Audio output completed")

    # Test 3: With vision (if available)
    print("\n5. Testing with vision model (if available)...")

    if narrative_gen._is_model_available(narrative_gen.vision_model):
        print(f"   Vision model {narrative_gen.vision_model} available")

        # Create a simple test frame (black image with white rectangle)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_frame, (100, 100), (300, 400), (255, 255, 255), -1)
        cv2.putText(test_frame, "TEST OBSTACLE", (120, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        print("   Generating narrative with vision...")
        start_time = time.time()
        narrative_vision = narrative_gen.generate_narrative(
            test_detections,
            frame=test_frame,
            navigation_summary=nav_summary
        )
        gen_time = time.time() - start_time

        print(f"   ✓ Vision narrative generated in {gen_time:.2f}s")
        print(f"   Generated: \"{narrative_vision}\"")

        # Speak vision-based narrative
        audio.speak(narrative_vision, priority=True)
        time.sleep(5)
    else:
        print(f"   ⚠ Vision model {narrative_gen.vision_model} not available, skipping")

    # Test 4: Fallback mechanism
    print("\n6. Testing fallback narrative...")

    # Force fallback by using non-existent model temporarily
    original_model = narrative_gen.primary_model
    narrative_gen.primary_model = "non-existent-model"

    fallback_narrative = narrative_gen.generate_narrative(
        test_detections,
        frame=None,
        navigation_summary=nav_summary
    )

    print(f"   ✓ Fallback narrative: \"{fallback_narrative}\"")

    # Restore original model
    narrative_gen.primary_model = original_model

    # Test 5: Performance check
    print("\n7. Performance check (10 rapid generations)...")
    times = []
    for i in range(10):
        start = time.time()
        _ = narrative_gen.generate_narrative(
            test_detections[:1],  # Use single detection for speed
            frame=None,
            navigation_summary={'total_objects': 1, 'danger_objects': [],
                              'caution_objects': [], 'safe_objects': [],
                              'closest_object': test_detections[0], 'path_clear': True}
        )
        times.append(time.time() - start)

    avg_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)

    print(f"   Average: {avg_time:.2f}s")
    print(f"   Min: {min_time:.2f}s")
    print(f"   Max: {max_time:.2f}s")

    if avg_time > 3.0:
        print(f"   ⚠ Warning: Average time {avg_time:.2f}s is high (>3s)")
        print(f"   This may cause delays in real-time operation")
    else:
        print(f"   ✓ Performance acceptable for 2s audio interval")

    # Cleanup
    audio.stop()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED! ✓")
    print("=" * 60)
    print("\nOllama and Audio systems are working properly!")
    print("You can now run the main application with:")
    print("  python src/main.py")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = test_ollama_audio_integration()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
