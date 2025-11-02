#!/usr/bin/env python3
"""
Depth Calibration Tool

Help calibrate distance measurements for OrbyGlasses.
Compares depth model output to real measurements.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import cv2
import numpy as np
from core.depth_anything_v2 import DepthAnythingV2
from core.utils import ConfigManager

print("\n" + "="*60)
print("DEPTH CALIBRATION TOOL")
print("="*60)
print("\nThis tool helps calibrate distance measurements.")
print("You'll need a measuring tape or ruler.\n")

# Initialize
config = ConfigManager('config/config.yaml')
depth_estimator = DepthAnythingV2(config)

# Open camera
camera_source = config.get('camera.source', 0)
cap = cv2.VideoCapture(camera_source)

if not cap.isOpened():
    print("❌ Cannot open camera")
    sys.exit(1)

print("✅ Camera opened")
print("\nINSTRUCTIONS:")
print("1. Place an object at a known distance (measure with tape)")
print("2. Point camera at the object")
print("3. Press SPACE to measure")
print("4. Enter the real distance when prompted")
print("5. Repeat for different distances (0.5m, 1m, 2m, 3m, 5m)")
print("6. Press 'q' to finish and generate calibration")
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

    cv2.putText(display, "Point at object, press SPACE to measure",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, f"Measurements: {len(measurements)}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, "Press 'q' to finish",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Calibration', display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        # Measure depth
        print("\n📏 Measuring depth...")
        depth_map = depth_estimator.estimate_depth(frame)

        # Get depth at center
        center_depth = depth_map[h//2, w//2]

        # Get depth in a small region around center
        region = depth_map[h//2-10:h//2+10, w//2-10:w//2+10]
        median_depth = np.median(region)
        mean_depth = np.mean(region)

        print(f"   Center depth: {center_depth:.3f}")
        print(f"   Median depth (20x20 region): {median_depth:.3f}")
        print(f"   Mean depth (20x20 region): {mean_depth:.3f}")

        # Ask for real distance
        try:
            real_distance = float(input("\n   Enter REAL distance in meters (e.g., 1.5): "))

            measurements.append({
                'measured': median_depth,
                'real': real_distance,
                'ratio': real_distance / median_depth if median_depth > 0 else 0
            })

            print(f"   ✅ Recorded: {median_depth:.3f} → {real_distance:.2f}m (ratio: {measurements[-1]['ratio']:.3f})")

        except ValueError:
            print("   ❌ Invalid input, skipping")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Analyze measurements
print("\n" + "="*60)
print("CALIBRATION RESULTS")
print("="*60)

if len(measurements) == 0:
    print("\n❌ No measurements taken. Run again and press SPACE to measure.")
    sys.exit(0)

print(f"\nTotal measurements: {len(measurements)}")
print("\nRaw data:")
print(f"{'Measured':<12} {'Real':<10} {'Ratio':<10}")
print("-" * 35)
for m in measurements:
    print(f"{m['measured']:<12.3f} {m['real']:<10.2f} {m['ratio']:<10.3f}")

# Calculate calibration factor
ratios = [m['ratio'] for m in measurements if m['ratio'] > 0]
if ratios:
    avg_ratio = np.mean(ratios)
    std_ratio = np.std(ratios)
    median_ratio = np.median(ratios)

    print(f"\nCalibration factors:")
    print(f"  Mean ratio:   {avg_ratio:.3f}")
    print(f"  Median ratio: {median_ratio:.3f}")
    print(f"  Std dev:      {std_ratio:.3f}")

    print(f"\n✅ RECOMMENDED CALIBRATION:")
    print(f"\nUpdate src/core/depth_anything_v2.py line 116:")
    print(f"\nOLD:")
    print(f"  metric_depth = normalized * 10.0")
    print(f"\nNEW:")
    print(f"  metric_depth = normalized * 10.0 * {median_ratio:.3f}  # Calibrated")

    print(f"\nOr even better, use:")
    print(f"  metric_depth = depth * {median_ratio:.3f}  # Direct scaling")

    # Test accuracy
    print(f"\nAccuracy test:")
    for m in measurements:
        calibrated = m['measured'] * median_ratio
        error = abs(calibrated - m['real'])
        error_pct = (error / m['real'] * 100) if m['real'] > 0 else 0
        print(f"  Real: {m['real']:.2f}m, Calibrated: {calibrated:.2f}m, Error: {error:.2f}m ({error_pct:.1f}%)")

else:
    print("\n❌ Could not calculate calibration")

print("\n" + "="*60)
