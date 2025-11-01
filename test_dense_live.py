#!/usr/bin/env python3
"""
Live Dense Reconstruction Test with Depth Prediction
Tests dense reconstruction capabilities with live camera feed
"""

import sys
import os

# Add pySLAM to path
pyslam_path = os.path.join(os.path.dirname(__file__), 'third_party', 'pyslam')
if os.path.exists(pyslam_path):
    sys.path.insert(0, pyslam_path)

cpp_lib_path = os.path.join(pyslam_path, 'cpp', 'lib')
if os.path.exists(cpp_lib_path):
    sys.path.insert(0, cpp_lib_path)

import cv2
import numpy as np
import time

print("🏗️  Dense Reconstruction with Depth Prediction - Live Test")
print("=" * 60)
print()

# Check if dense reconstruction modules are available
try:
    from pyslam.config import Config
    from pyslam.slam.slam import Slam
    from pyslam.slam.camera import PinholeCamera
    from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
    from pyslam.io.dataset_types import SensorType, DatasetEnvironmentType
    from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs
    print("✅ pySLAM modules loaded")
except ImportError as e:
    print(f"❌ Error loading pySLAM: {e}")
    sys.exit(1)

# Check dense reconstruction
try:
    from pyslam.dense.volumetric_integrator_factory import volumetric_integrator_factory
    from pyslam.dense.volumetric_integrator_tsdf import VolumetricIntegratorTSDF
    DENSE_AVAILABLE = True
    print("✅ Dense reconstruction modules available")
except ImportError as e:
    DENSE_AVAILABLE = False
    print(f"⚠️  Dense reconstruction not available: {e}")

# Check depth prediction
try:
    from pyslam.depth_estimation.depth_estimator_factory import depth_estimator_factory
    DEPTH_PREDICTION_AVAILABLE = True
    print("✅ Depth prediction available")
except ImportError as e:
    DEPTH_PREDICTION_AVAILABLE = False
    print(f"⚠️  Depth prediction not available: {e}")

print()

if not DENSE_AVAILABLE:
    print("⚠️  Running SLAM without dense reconstruction")
    print("   Dense reconstruction requires additional setup")
    print()

# Camera configuration
width, height = 640, 480
fx, fy = 500.0, 500.0
cx, cy = width / 2.0, height / 2.0

print("📷 Camera Configuration:")
print(f"   Resolution: {width}x{height}")
print(f"   Focal length: fx={fx}, fy={fy}")
print(f"   Principal point: cx={cx}, cy={cy}")
print(f"   Camera index: 1 (as requested)")
print()

# Initialize camera
print("🎥 Opening camera 1...")
cap = cv2.VideoCapture(1)  # Using camera 1 as requested
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("❌ Error: Could not open camera 1")
    print("   Trying camera 0 as fallback...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open any camera")
        sys.exit(1)
    print("✅ Using camera 0 as fallback")
else:
    print("✅ Camera 1 opened successfully")

print()

# Create camera config
camera_config = Config()
camera_config.cam_settings = {
    'Camera.width': width,
    'Camera.height': height,
    'Camera.fx': fx,
    'Camera.fy': fy,
    'Camera.cx': cx,
    'Camera.cy': cy,
    'Camera.fps': 30,
    'Camera.k1': 0.0,
    'Camera.k2': 0.0,
    'Camera.p1': 0.0,
    'Camera.p2': 0.0,
    'Camera.k3': 0.0
}

camera = PinholeCamera(camera_config)

# Feature tracker config
feature_tracker_config = FeatureTrackerConfigs.ORB.copy()
feature_tracker_config["num_features"] = 3000

# Loop detector config
try:
    loop_detection_config = LoopDetectorConfigs.DBOW3
    print("✅ Loop closure: DBOW3")
except:
    loop_detection_config = None
    print("⚠️  Loop closure: disabled")

print()
print("🚀 Initializing SLAM with dense reconstruction support...")
print()

# Initialize SLAM
try:
    slam = Slam(
        camera,
        feature_tracker_config,
        loop_detection_config,
        None,  # semantic_mapping_config
        SensorType.MONOCULAR,
        environment_type=DatasetEnvironmentType.INDOOR,
        config=camera_config
    )

    print("✅ SLAM initialized successfully")
    print()

    # Check if volumetric integration is enabled
    if hasattr(slam, 'volumetric_integrator'):
        print("✅ Volumetric integration: ENABLED")
        print(f"   Type: {type(slam.volumetric_integrator).__name__}")
    else:
        print("⚠️  Volumetric integration: NOT ENABLED")
        print("   To enable: set kUseVolumetricIntegration=True in config")

    print()

except Exception as e:
    print(f"❌ Error initializing SLAM: {e}")
    import traceback
    traceback.print_exc()
    cap.release()
    sys.exit(1)

# Main loop
print("🎬 Starting SLAM...")
print("   Press 'q' to quit")
print("   Press 's' to save map")
print()

frame_count = 0
start_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process through SLAM
        timestamp = time.time() - start_time
        slam.track(rgb_frame, None, None, frame_count, timestamp)

        # Display
        cv2.imshow("Dense Reconstruction Test - Camera 1", frame)

        # Stats
        if frame_count % 30 == 0:
            num_points = len(slam.map.points) if hasattr(slam, 'map') else 0
            num_keyframes = len(slam.map.keyframes) if hasattr(slam, 'map') else 0
            print(f"Frame {frame_count}: {num_keyframes} keyframes, {num_points} map points")

        frame_count += 1

        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            print(f"\n💾 Saving map at frame {frame_count}...")
            # Save map (would need to implement)
            print("   Map save functionality pending")

except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user")

except Exception as e:
    print(f"\n❌ Error during processing: {e}")
    import traceback
    traceback.print_exc()

finally:
    print("\n🧹 Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Done!")
    print()
    print(f"📊 Statistics:")
    print(f"   Total frames: {frame_count}")
    print(f"   Duration: {time.time() - start_time:.1f}s")
    if frame_count > 0:
        print(f"   Average FPS: {frame_count / (time.time() - start_time):.1f}")
