#!/bin/bash
echo "🔧 Quick Fix for Tracking Issues"
echo "================================="
echo ""
echo "Checking environment..."
echo ""

# Test camera and feature detection
python3 << 'PYTHON'
import cv2
import sys

print("📷 Testing camera 1...")
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("❌ Camera 1 not available - trying camera 0")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No camera available!")
        sys.exit(1)
    print("✅ Using camera 0")
    CAMERA_INDEX = 0
else:
    print("✅ Camera 1 working")
    CAMERA_INDEX = 1

ret, frame = cap.read()
if not ret:
    print("❌ Cannot read frame")
    cap.release()
    sys.exit(1)

# Check brightness
brightness = frame.mean()
print(f"\n💡 Brightness: {brightness:.1f}")
if brightness < 80:
    print("   ⚠️  Too dark - turn on more lights!")
elif brightness > 180:
    print("   ⚠️  Too bright - avoid direct sunlight!")
else:
    print("   ✅ Lighting OK")

# Detect features
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
orb = cv2.ORB_create(nfeatures=3000)
kp = orb.detect(gray, None)
num_features = len(kp)

print(f"\n🎯 Features detected: {num_features}")
if num_features < 500:
    print("   ❌ TOO FEW features!")
    print("   📋 Point camera at:")
    print("      • Books on shelf")
    print("      • Desk with items")
    print("      • Posters/pictures")
    print("      • Textured surfaces")
    print("      ❌ NOT blank walls!")
elif num_features < 1500:
    print("   ⚠️  Low features - try better area")
else:
    print("   ✅ Good features!")

cap.release()

print(f"\n🔧 Recommended settings:")
print(f"   camera.source: {CAMERA_INDEX}")
print(f"   slam.orb_features: {max(3000, num_features + 500)}")

PYTHON

echo ""
echo "✅ Diagnostic complete!"
echo ""
echo "To fix tracking issues:"
echo "  1. Improve lighting (turn on lights)"
echo "  2. Point at textured surfaces"
echo "  3. Move slowly at startup"
echo "  4. Run: ./run_orby.sh"
