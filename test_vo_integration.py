#!/usr/bin/env python3
"""
Test Visual Odometry Integration with OrbyGlasses

This script tests the pySLAM Visual Odometry integration
without running the full OrbyGlasses system.
"""

import sys
import os
import cv2
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.utils import ConfigManager
from navigation.pyslam_vo_integration import PySLAMVisualOdometry, PYSLAM_VO_AVAILABLE

def test_visual_odometry():
    """Test Visual Odometry integration."""
    print("🧪 Testing Visual Odometry Integration...")
    
    if not PYSLAM_VO_AVAILABLE:
        print("❌ pySLAM Visual Odometry not available")
        return False
    
    # Load configuration
    config_manager = ConfigManager("config/config.yaml")
    config = config_manager.config
    
    # Enable Visual Odometry
    config['visual_odometry']['enabled'] = True
    config['visual_odometry']['use_pyslam_vo'] = True
    
    print("✅ Configuration loaded")
    
    # Initialize Visual Odometry
    try:
        vo = PySLAMVisualOdometry(config)
        print("✅ Visual Odometry initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Visual Odometry: {e}")
        return False
    
    # Start Visual Odometry
    if not vo.start():
        print("❌ Failed to start Visual Odometry")
        return False
    
    print("✅ Visual Odometry started")
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return False
    
    print("✅ Camera opened")
    print("Press 'q' to quit, 's' to save trajectory image")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        # Process frame
        vo_result = vo.process_frame(frame)
        
        if vo_result and frame_count % 30 == 0:  # Log every 30 frames
            pos = vo_result.get('position', [0, 0, 0])
            print(f"Frame {frame_count}: Position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
        # Get trajectory image
        traj_img = vo.get_trajectory_image()
        
        # Display images
        cv2.imshow("Camera", frame)
        cv2.imshow("Trajectory", traj_img)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"trajectory_{int(time.time())}.png", traj_img)
            print("✅ Trajectory image saved")
        
        frame_count += 1
    
    # Cleanup
    vo.cleanup()
    cap.release()
    cv2.destroyAllWindows()
    
    print("✅ Visual Odometry test completed")
    return True

if __name__ == "__main__":
    success = test_visual_odometry()
    if success:
        print("🎉 Visual Odometry integration test PASSED!")
    else:
        print("❌ Visual Odometry integration test FAILED!")
        sys.exit(1)
