#!/usr/bin/env python3
"""Test ORB feature detection"""

import cv2
import numpy as np

# Open camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Create ORB detector - ULTRA SENSITIVE
orb = cv2.ORB_create(
    nfeatures=3000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=10,
    firstLevel=0,
    WTA_K=2,
    scoreType=cv2.ORB_HARRIS_SCORE,
    patchSize=31,
    fastThreshold=5
)

print("Testing ORB feature detection...")
print("Press 'q' to quit")
print()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Detect features
    kps, desc = orb.detectAndCompute(sharpened, None)

    # Draw features
    vis = cv2.drawKeypoints(frame, kps, None, color=(0,255,0))

    # Show count
    cv2.putText(vis, f"Features: {len(kps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    print(f"\rDetected: {len(kps):4d} features", end='', flush=True)

    cv2.imshow('Features', vis)
    cv2.imshow('Enhanced', sharpened)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nDone!")
