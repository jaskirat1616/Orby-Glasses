#!/usr/bin/env python3
"""
Integration test to verify all components work together
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_integration():
    """Test that all components work together."""
    print("Testing integration of components...")
    
    try:
        # Import all components
        from slam import MonocularSLAM
        from occupancy_grid_3d import OccupancyGrid3D
        from movement_visualizer import MovementVisualizer
        from coordinate_transformer import CoordinateTransformer
        
        # Create mock config
        class MockConfig:
            def get(self, key, default=None):
                config = {
                    # SLAM config
                    'slam.enabled': True,
                    'slam.orb_features': 1000,
                    'slam.min_matches': 15,
                    'slam.pose_alpha': 0.7,
                    'slam.temporal_consistency_check': True,
                    
                    # Occupancy grid config
                    'occupancy_grid_3d.enabled': True,
                    'occupancy_grid_3d.grid_size': [10.0, 10.0, 3.0],
                    'occupancy_grid_3d.resolution': 0.1,
                    'occupancy_grid_3d.temporal_consistency': True,
                    'occupancy_grid_3d.temporal_filtering': True,
                    
                    # Movement visualizer config
                    'movement_visualizer.enabled': True,
                    'movement_visualizer.canvas_size': [800, 800],
                    'movement_visualizer.trail_length': 100,
                    
                    # Camera config
                    'mapping3d.fx': 500.0,
                    'mapping3d.fy': 500.0,
                    'camera.width': 320,
                    'camera.height': 320,
                }
                return config.get(key, default)
        
        config = MockConfig()
        
        # Initialize components
        print("  Initializing SLAM...")
        slam = MonocularSLAM(config)
        
        print("  Initializing Occupancy Grid...")
        grid = OccupancyGrid3D(config)
        
        print("  Initializing Movement Visualizer...")
        mv = MovementVisualizer(config)
        
        print("  Initializing Coordinate Transformer...")
        ct = CoordinateTransformer(config)
        
        # Simulate a simple tracking scenario
        print("  Running simulation...")
        
        for i in range(5):
            # Create dummy frame and depth
            # In a real scenario, these would come from the camera
            dummy_frame = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
            dummy_depth = np.random.rand(320, 320).astype(np.float32) * 0.5 + 0.5  # 0.5-1.0 range
            
            # Process with SLAM
            result = slam.process_frame(dummy_frame, dummy_depth)
            
            # Update occupancy grid
            grid.update_from_depth(dummy_depth, result['pose'])
            
            # Update movement visualizer
            mv.update(result)
            
            # Test coordinate transformation
            world_point = ct.pixel_to_world(
                np.array([160, 120]),  # center pixel
                2.0,  # depth
                result['pose']
            )
            
            print(f"    Frame {i+1}: Position {result['position'][:2]}, "
                  f"Tracking quality: {result['tracking_quality']:.2f}")
        
        # Verify all components have data
        slam_pos_count = len(slam.position_history)
        grid_voxels = len(grid.occupied_voxels)
        mv_positions = len(mv.position_trail)
        
        print(f"  SLAM position history: {slam_pos_count}")
        print(f"  Grid occupied voxels: {grid_voxels}")
        print(f"  Movement visualizer positions: {mv_positions}")
        print(f"  Coordinate transformer test: {world_point[:2]}")
        
        # All components should be properly integrated
        success = (slam_pos_count > 0 and 
                  grid_voxels >= 0 and  # May be 0 if no obstacles detected in dummy data
                  mv_positions > 0 and
                  len(world_point) == 3)
        
        print(f"  Integration test: {'✓ PASSED' if success else '✗ FAILED'}")
        return success
        
    except Exception as e:
        print(f"  Integration test: ✗ FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("OrbyGlasses Components Integration Test")
    print("=" * 50)
    
    success = test_integration()
    
    print("\n" + "=" * 50)
    print(f"Integration Test: {'✓ PASSED' if success else '✗ FAILED'}")
    print("=" * 50)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)