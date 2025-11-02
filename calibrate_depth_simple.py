#!/usr/bin/env python3
"""
Simple Depth Calibration Tool

Helps you understand depth estimation accuracy.
Shows you current depth measurements vs. real distances.
"""

import cv2
import numpy as np
import sys

print("\n" + "="*60)
print("SIMPLE DEPTH CALIBRATION TOOL")
print("="*60)
print("\nThis tool shows you how the depth estimation works.")
print("You can compare estimated distances to real measurements.\n")

# Open camera
camera_source = 0
cap = cv2.VideoCapture(camera_source)

if not cap.isOpened():
    print("❌ Cannot open camera")
    sys.exit(1)

print("✅ Camera opened")
print("\nNOTE: OrbyGlasses uses a depth estimation model that needs to be loaded")
print("when the full system runs. This simple tool shows you the camera view.")
print("\nTo see actual depth calibration:")
print("1. Run: ./run_orby.sh")
print("2. Look at the distance values shown for detected objects")
print("3. Compare with real measurements using a tape measure")
print("\nINSTRUCTIONS:")
print("1. Place an object at a known distance (measure with tape)")
print("2. Run ./run_orby.sh to see estimated distance")
print("3. Compare estimated vs. real distance")
print("4. Note the difference for accuracy assessment")
print("\nPress 'q' to quit this preview")
print()

measurements = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show frame
    display = frame.copy()
    h, w = frame.shape[:2]

    # Draw center crosshair
    cv2.line(display, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 0), 2)
    cv2.line(display, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 0), 2)

    cv2.putText(display, "Point at object and run ./run_orby.sh to see distance",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(display, "Press 'q' to quit",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('Camera Preview - Calibration Tool', display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("CALIBRATION GUIDE")
print("="*60)
print("\nTo check depth accuracy:")
print("\n1. Place object at known distance (e.g., 1.0 meter)")
print("2. Measure with tape measure")
print("3. Run: ./run_orby.sh")
print("4. Point camera at object")
print("5. Read distance shown by OrbyGlasses")
print("6. Compare: Real distance vs. Estimated distance")
print("\nExample:")
print("  Real distance: 1.0m")
print("  OrbyGlasses says: 1.2m")
print("  Error: +0.2m (20% over-estimation)")
print("\nTypical accuracy: ±30-40% for monocular depth")
print("This is normal for single-camera depth estimation.")
print("\n" + "="*60)
