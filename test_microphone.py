#!/usr/bin/env python3
"""
Test microphone and speech recognition for OrbyGlasses
"""

import sys

print("=" * 60)
print("OrbyGlasses - Microphone & Speech Recognition Test")
print("=" * 60)

# Test 1: Check if SpeechRecognition is installed
print("\n[Test 1] Checking SpeechRecognition installation...")
try:
    import speech_recognition as sr
    print(f"✓ SpeechRecognition {sr.__version__} is installed")
except ImportError as e:
    print(f"✗ SpeechRecognition not found: {e}")
    print("\nTo install, run:")
    print("  pip install SpeechRecognition")
    sys.exit(1)

# Test 2: Check microphone access
print("\n[Test 2] Checking microphone access...")
try:
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print(f"✓ Microphone found: {mic}")

    # List all microphones
    print("\nAvailable microphones:")
    for index, name in enumerate(sr.Microphone.list_microphone_names()):
        print(f"  [{index}] {name}")

except Exception as e:
    print(f"✗ Microphone error: {e}")
    sys.exit(1)

# Test 3: Calibrate for ambient noise
print("\n[Test 3] Calibrating for ambient noise...")
print("Please wait, calibrating...")
try:
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    print(f"✓ Calibrated. Energy threshold: {recognizer.energy_threshold}")
except Exception as e:
    print(f"✗ Calibration error: {e}")
    sys.exit(1)

# Test 4: Test wake phrase detection
print("\n[Test 4] Testing wake phrase detection...")
print("\nSay 'hey orby' now (you have 5 seconds)...")

try:
    with mic as source:
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)

    print("Processing speech...")
    text = recognizer.recognize_google(audio).lower()
    print(f"\nYou said: '{text}'")

    if "hello" in text or "hallo" in text:
        print("✓ Wake phrase DETECTED!")
    else:
        print("✗ Wake phrase NOT detected, but speech recognition is working")
        print("  Try saying 'hey orby' more clearly")

except sr.WaitTimeoutError:
    print("✗ No speech detected (timeout)")
    print("  Make sure your microphone is working and not muted")
except sr.UnknownValueError:
    print("✗ Could not understand speech")
    print("  Try speaking more clearly")
except sr.RequestError as e:
    print(f"✗ Google Speech Recognition error: {e}")
    print("  Check your internet connection")
except Exception as e:
    print(f"✗ Unexpected error: {e}")

# Test 5: Interactive test
print("\n[Test 5] Interactive test (optional)")
print("Say something, and I'll repeat it back (5 seconds)...")
print("Press Ctrl+C to skip")

try:
    with mic as source:
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)

    text = recognizer.recognize_google(audio)
    print(f"\nYou said: '{text}'")

except KeyboardInterrupt:
    print("\nSkipped")
except sr.WaitTimeoutError:
    print("No speech detected")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Test complete!")
print("\nIf all tests passed, conversational navigation should work.")
print("Run OrbyGlasses with: python src/main.py")
print("=" * 60)
