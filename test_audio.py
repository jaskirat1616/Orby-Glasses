#!/usr/bin/env python3
"""
Test Audio System

Verifies that macOS say command works in OrbyGlasses.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("\n" + "="*60)
print("TESTING ORBGLASSES AUDIO SYSTEM")
print("="*60 + "\n")

# Test 1: Direct macOS say
print("1️⃣  Testing macOS 'say' command directly...")
import subprocess
import time

start = time.time()
result = subprocess.run(['say', '-v', 'Samantha', '-r', '220', 'Testing OrbyGlasses audio. Can you hear me?'],
                       capture_output=True, timeout=10)
duration = time.time() - start

if result.returncode == 0:
    print(f"   ✅ macOS 'say' command works ({duration:.1f}s)")
    print("   Did you hear: 'Testing OrbyGlasses audio. Can you hear me?'")
else:
    print(f"   ❌ macOS 'say' failed: {result.stderr.decode()}")

# Test 2: AudioManager from utils.py
print("\n2️⃣  Testing OrbyGlasses AudioManager...")
try:
    from core.utils import ConfigManager, AudioManager

    config = ConfigManager('config/config.yaml')
    audio = AudioManager(config)

    print("   Queueing test message...")
    audio.speak("This is OrbyGlasses audio manager speaking. Testing one two three.")

    print("   ✅ AudioManager initialized")
    print("   Waiting 5 seconds for audio to play...")
    time.sleep(5)

    print("\n   Did you hear: 'This is OrbyGlasses audio manager speaking'?")

except Exception as e:
    print(f"   ❌ AudioManager failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: FastAudioManager
print("\n3️⃣  Testing FastAudioManager...")
try:
    from core.fast_audio import FastAudioManager, emergency_alert

    fast_audio = FastAudioManager(rate=220, voice="Samantha")

    print("   Testing emergency alert...")
    emergency_alert(fast_audio, "Emergency stop! Obstacle ahead!")

    time.sleep(3)

    print("\n   Did you hear: 'Stop! Emergency stop! Obstacle ahead!'?")

    fast_audio.shutdown()

except Exception as e:
    print(f"   ❌ FastAudioManager failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("AUDIO TEST COMPLETE")
print("="*60)
print("\nIf you didn't hear any audio:")
print("1. Check System Settings > Sound > Output device")
print("2. Check volume is not muted")
print("3. Try: say 'test' in terminal")
print("4. Check if other apps have sound")
print()
