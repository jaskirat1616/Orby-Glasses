#!/bin/bash
# Test OrbyGlasses with live webcam and feature matching visualization

cd /Users/jaskiratsingh/Desktop/OrbyGlasses

echo "🎥 Testing OrbyGlasses with Webcam + Feature Matching"
echo ""
echo "Windows you'll see:"
echo "  1. OrbyGlasses - Main camera view"
echo "  2. Feature Matching - Side-by-side frames with matched features"
echo ""
echo "Press 'q' in any window to quit"
echo ""
echo "Starting in 3 seconds..."
sleep 3

# Run with default camera (index 0) and show features
python3 src/main.py --show-features

echo ""
echo "Test complete!"
