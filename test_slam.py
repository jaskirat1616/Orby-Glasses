#!/usr/bin/env python3
"""
Simple SLAM test to diagnose issues
"""

import cv2
import numpy as np
import sys
sys.path.insert(0, 'src')

from navigation.accurate_slam import AccurateSLAM
from core.utils import ConfigManager

# Load config
config = ConfigManager('config/config.yaml')

# Initialize SLAM
slam = AccurateSLAM(config)

# Open camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("=" * 60)
print("SLAM DIAGNOSTIC TEST")
print("=" * 60)
print("Press 'q' to quit")
print()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    frame_count += 1

    # Process
    result = slam.process_frame(frame)

    # Show info every 10 frames
    if frame_count % 10 == 0:
        print(f"\nFrame {frame_count}:")
        print(f"  State: {result['tracking_state']}")
        print(f"  Map points: {result['num_map_points']}")
        print(f"  Keyframes: {result['num_keyframes']}")
        print(f"  Initialized: {result['initialized']}")

        if result['initialized']:
            print(f"  ✅ SLAM WORKING!")
            print(f"  Position: {result['position']}")

    # Show frame
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
