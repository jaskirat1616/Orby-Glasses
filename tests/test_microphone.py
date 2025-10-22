#!/usr/bin/env python3
"""
Simple test to check if microphone access works
"""
import speech_recognition as sr

def test_microphone():
    print("Testing microphone access...")
    
    # Create a recognizer instance
    recognizer = sr.Recognizer()
    
    # Try to access the microphone
    try:
        with sr.Microphone() as source:
            print("Microphone accessed successfully!")
            print("Calibrating for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            print("Calibration complete!")
    except OSError as e:
        print(f"Error accessing microphone: {e}")
        print("This might be due to:")
        print("- No microphone detected")
        print("- Microphone permissions not granted on macOS")
        print("- Another application using the microphone")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_microphone()
    if success:
        print("\nMicrophone test: PASSED")
    else:
        print("\nMicrophone test: FAILED")