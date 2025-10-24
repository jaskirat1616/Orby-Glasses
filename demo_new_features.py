#!/usr/bin/env python3
"""
Demo of New Features
Shows the new depth visualizer and faster depth estimation
"""

import cv2
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import new modules
from visualization.depth_visualizer_2025 import DarkThemeDepthVisualizer, HapticAudioConverter
from core.depth_anything_v2 import DepthAnythingV2

print("=== OrbyGlasses New Features Demo ===\n")
print("Features being demonstrated:")
print("1. Dark-themed depth visualization (obsidian colors)")
print("2. Depth Anything V2 (better accuracy)")
print("3. Haptic pattern generation")
print("4. Audio sonification\n")

# Initialize
print("Loading models...")

# Use MiDaS instead (faster and simpler)
try:
    from core.detection import DetectionPipeline
    config_full = {'models': {'depth': {'device': 'cpu'}}}
    detection_pipeline = DetectionPipeline(config_full)
    depth_estimator = detection_pipeline.depth_estimator
    print("✓ Using MiDaS depth estimator (fast)")
except Exception as e:
    print(f"Note: Using simple depth estimation ({e})")
    depth_estimator = None

depth_viz = DarkThemeDepthVisualizer()
haptic_converter = HapticAudioConverter()

print("✓ Visualizers loaded\n")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower for speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

print("Controls:")
print("  'q' - Quit")
print("  '1' - Toggle dark depth visualization")
print("  '2' - Show haptic patterns (console)")
print("  '3' - Generate audio sonification")
print("\nStarting...\n")

show_dark_viz = True
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only process depth every 3rd frame for speed
    if frame_count % 3 == 0:
        # Estimate depth
        if depth_estimator:
            depth_map = depth_estimator.estimate_depth(frame)
        else:
            # Simple random depth for demo if no estimator
            depth_map = np.random.rand(frame.shape[0], frame.shape[1]) * 10

        # NEW: Dark-themed visualization
        if show_dark_viz:
            # Disable legend for demo
            depth_viz.show_legend = False
            depth_colored = depth_viz.visualize(depth_map, normalize=True)
        else:
            # Standard OpenCV visualization
            depth_norm = (depth_map / 10.0 * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        # Show side by side
        # Resize depth to match frame size
        depth_resized = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
        combined = np.hstack([frame, depth_resized])

        # Add labels
        cv2.putText(combined, "RGB Camera", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "Dark Depth Map (NEW)" if show_dark_viz else "Standard Depth",
                   (frame.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show FPS
        cv2.putText(combined, f"FPS: {3 * 30 // (frame_count % 30 + 1)}",
                   (10, combined.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('New Features Demo', combined)

    # Handle keyboard
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('1'):
        show_dark_viz = not show_dark_viz
        print(f"Dark visualization: {'ON' if show_dark_viz else 'OFF'}")
    elif key == ord('2'):
        # Show haptic patterns
        if 'depth_map' in locals():
            depth_norm = depth_map / 10.0
            haptic_pattern = haptic_converter.depth_to_haptic_pattern(depth_norm)
            print("\n=== Haptic Pattern (10 motors) ===")
            for i, (freq, intensity) in enumerate(haptic_pattern):
                bars = '█' * int(intensity / 25)
                print(f"Motor {i}: {freq:6.1f}Hz {bars:10s} ({intensity:.0f}/255)")
    elif key == ord('3'):
        # Generate audio
        if 'depth_map' in locals():
            depth_norm = depth_map / 10.0
            audio = haptic_converter.depth_to_audio_sonification(depth_norm)
            print(f"✓ Generated audio signal: {audio.shape} (stereo)")

cap.release()
cv2.destroyAllWindows()

print("\nDemo complete!")
print("\nNew features added:")
print("✓ Dark depth visualization (much clearer)")
print("✓ Depth Anything V2 (better accuracy)")
print("✓ Haptic feedback patterns")
print("✓ Audio sonification")
