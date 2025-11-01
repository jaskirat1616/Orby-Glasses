#!/usr/bin/env python3
"""Test ORB feature detection with different parameters"""
import cv2
import sys

print("Testing ORB feature detection...")
print()

cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)
    print("Using camera 0")
else:
    print("Using camera 1")

ret, frame = cap.read()
if not ret:
    print("Cannot read frame")
    sys.exit(1)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

print("\n1. Default ORB (strict):")
orb_default = cv2.ORB_create(nfeatures=5000)
kp_default = orb_default.detect(gray, None)
print(f"   Features: {len(kp_default)}")

print("\n2. Permissive ORB (for typical environments):")
orb_permissive = cv2.ORB_create(
    nfeatures=5000,
    edgeThreshold=10,  # Lower = more features
    fastThreshold=10,  # Lower = more features
    patchSize=31
)
kp_permissive = orb_permissive.detect(gray, None)
print(f"   Features: {len(kp_permissive)}")

print("\n3. Very permissive ORB (for challenging environments):")
orb_very_permissive = cv2.ORB_create(
    nfeatures=5000,
    edgeThreshold=5,
    fastThreshold=5,
    patchSize=31
)
kp_very_permissive = orb_very_permissive.detect(gray, None)
print(f"   Features: {len(kp_very_permissive)}")

cap.release()

print("\n✅ Recommendation:")
if len(kp_permissive) > 2000:
    print("   Permissive settings should work great!")
elif len(kp_very_permissive) > 2000:
    print("   Use very permissive settings")
else:
    print("   Environment is very challenging - improve lighting")

