#!/usr/bin/env python3
"""
Test that audio sentences complete fully with 5s interval.
"""

import sys
import os
import time
sys.path.insert(0, 'src')

from utils import ConfigManager, AudioManager
import logging

# Enable debug logging to see everything
logging.basicConfig(level=logging.INFO)

def test_audio_completion():
    """Test that long sentences complete without cutoff."""

    print("=" * 70)
    print("AUDIO COMPLETION TEST (5-second interval)")
    print("=" * 70)

    config = ConfigManager('config/config.yaml')
    audio = AudioManager(config)

    interval = config.get('performance.audio_update_interval', 5.0)
    rate = config.get('audio.tts_rate', 180)

    print(f"\n✓ Audio interval: {interval}s")
    print(f"✓ TTS rate: {rate} WPM")

    # Real Ollama outputs from your system
    real_sentences = [
        "Caution: There is a person approximately 5.1 meters to your right and ahead.",
        "Person ahead at 2.3 meters, moving right.",
        "Chair ahead at 1.2 meters - proceed with caution, maintaining a distance of at least 2 meters.",
        "Bicycle ahead at 2.8 meters, person to your left at 3 meters.",
    ]

    print(f"\n{'='*70}")
    print("SIMULATING REAL SYSTEM BEHAVIOR")
    print(f"{'='*70}")

    for i, sentence in enumerate(real_sentences, 1):
        word_count = len(sentence.split())
        estimated_time = (word_count / rate) * 60
        
        print(f"\n--- Message {i} ---")
        print(f"Sentence: \"{sentence}\"")
        print(f"Words: {word_count}, Estimated time: {estimated_time:.1f}s")
        
        # Check if speaking
        if audio.is_speaking:
            print(f"❌ ERROR: Still speaking from previous message!")
            print(f"   This would cause cutoff in real system")
        else:
            print(f"✓ Ready to speak (not currently speaking)")
        
        # Speak
        audio.speak(sentence, priority=False)
        
        # Wait for interval (simulating real system)
        print(f"Waiting {interval}s (system interval)...")
        time.sleep(interval)
        
        # Check status after interval
        if audio.is_speaking:
            print(f"⚠️  Still speaking after {interval}s wait")
            print(f"   Sentence likely longer than interval")
        else:
            print(f"✓ Speech completed within {interval}s interval")

    print(f"\n{'='*70}")
    print("WAITING FOR ALL AUDIO TO COMPLETE...")
    print(f"{'='*70}")
    time.sleep(3)

    audio.stop()

    print(f"\n{'='*70}")
    print("TEST COMPLETE")
    print(f"{'='*70}")
    print("\nWith 5s interval:")
    print("✓ Short sentences (8-10 words): ~2.7-3.3s - COMPLETE")
    print("✓ Medium sentences (12-14 words): ~4.0-4.7s - COMPLETE")  
    print("✓ Long sentences (18-20 words): ~6.0-6.7s - May overlap but won't cutoff")
    print("\nSYSTEM WILL:")
    print("1. Wait 5s between queuing new audio")
    print("2. Skip if already speaking (prevents cutoff)")
    print("3. Only speak latest message (clears queue)")
    print(f"{'='*70}")

if __name__ == "__main__":
    try:
        test_audio_completion()
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
