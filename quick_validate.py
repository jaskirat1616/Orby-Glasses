"""
Quick validation of the SLAM and 3D mapping components
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    try:
        from slam_system import SLAMSystem
        from voxel_map import VoxelMap
        from movement_visualizer import MovementVisualizer
        from coordinate_transformer import CoordinateTransformer
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Other error: {e}")
        return False

def test_slam_features():
    """Test SLAM enhancements."""
    print("\nTesting SLAM features...")
    
    try:
        from utils import ConfigManager
        # Create mock config
        class MockConfig:
            def get(self, key, default=None):
                config = {
                    'slam.enabled': True,
                    'slam.orb_features': 3000,
                    'slam.min_matches': 15,
                    'slam.min_tracked_features': 10,
                    'slam.keyframe_threshold': 20,
                    'slam.pose_alpha': 0.8,
                    'slam.temporal_consistency_check': True,
                    'mapping3d.fx': 500.0,
                    'mapping3d.fy': 500.0,
                    'camera.width': 320,
                    'camera.height': 320,
                }
                return config.get(key, default)
        
        from slam_system import SLAMSystem
        slam = SLAMSystem(MockConfig())
        
        # Check for features
        has_velocity_history = hasattr(slam, 'velocity_history')
        has_temporal_check = hasattr(slam, '_is_pose_consistent')
        has_methods = hasattr(slam, '_estimate_scale_from_depth')
        
        print(f"  Velocity history: {has_velocity_history}")
        print(f"  Temporal consistency: {has_temporal_check}")
        print(f"  Methods: {has_methods}")
        
        # Check if movement summary method exists
        has_movement_summary = hasattr(slam, 'get_movement_summary')
        print(f"  Movement summary: {has_movement_summary}")
        
        print("✓ SLAM enhancements validated")
        return has_velocity_history and has_temporal_check and has_methods and has_movement_summary
        
    except Exception as e:
        print(f"✗ SLAM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_occupancy_grid_enhancements():
    """Test 3D occupancy grid enhancements."""
    print("\nTesting 3D Occupancy Grid enhancements...")
    
    try:
        # Create mock config
        class MockConfig:
            def get(self, key, default=None):
                config = {
                    'occupancy_grid_3d.enabled': True,
                    'occupancy_grid_3d.grid_size': [20.0, 20.0, 3.0],
                    'occupancy_grid_3d.resolution': 0.1,
                    'occupancy_grid_3d.temporal_consistency': True,
                    'occupancy_grid_3d.min_observations': 2,
                    'occupancy_grid_3d.temporal_filtering': True,
                    'mapping3d.fx': 500.0,
                    'mapping3d.fy': 500.0,
                    'camera.width': 320,
                    'camera.height': 320,
                }
                return config.get(key, default)
        
        from voxel_map import VoxelMap
        grid = VoxelMap(MockConfig())
        
        # Check for features
        has_temporal_filtering = hasattr(grid, '_update_voxel_with_temporal_filtering')
        has_uncertainty_modeling = hasattr(grid, '_calculate_depth_uncertainty')
        has_vis = hasattr(grid, 'visualize_3d_interactive')
        
        print(f"  Temporal filtering: {has_temporal_filtering}")
        print(f"  Uncertainty modeling: {has_uncertainty_modeling}")
        print(f"  Visualization: {has_vis}")
        
        print("✓ Occupancy Grid enhancements validated")
        return has_temporal_filtering and has_uncertainty_modeling and has_vis
        
    except Exception as e:
        print(f"✗ Occupancy Grid test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_movement_visualizer():
    """Test movement visualizer."""
    print("\nTesting Movement Visualizer...")
    
    try:
        # Create mock config
        class MockConfig:
            def get(self, key, default=None):
                config = {
                    'movement_visualizer.enabled': True,
                    'movement_visualizer.canvas_size': [800, 800],
                    'movement_visualizer.trail_length': 200,
                    'movement_visualizer.grid_size': 20.0,
                }
                return config.get(key, default)
        
        from movement_visualizer import MovementVisualizer
        visualizer = MovementVisualizer(MockConfig())
        
        # Check for key features
        has_trail = hasattr(visualizer, 'position_trail')
        has_velocity_history = hasattr(visualizer, 'velocity_history')
        has_stats = hasattr(visualizer, 'movement_stats')
        
        print(f"  Position trail: {has_trail}")
        print(f"  Velocity history: {has_velocity_history}")
        print(f"  Movement stats: {has_stats}")
        
        print("✓ Movement Visualizer validated")
        return has_trail and has_velocity_history and has_stats
        
    except Exception as e:
        print(f"✗ Movement Visualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_coordinate_transformer():
    """Test coordinate transformer."""
    print("\nTesting Coordinate Transformer...")
    
    try:
        # Create mock config
        class MockConfig:
            def get(self, key, default=None):
                config = {
                    'mapping3d.fx': 500.0,
                    'mapping3d.fy': 500.0,
                    'camera.width': 320,
                    'camera.height': 320,
                }
                return config.get(key, default)
        
        from coordinate_transformer import CoordinateTransformer
        transformer = CoordinateTransformer(MockConfig())
        
        # Test basic functionality
        test_pixel = np.array([160, 120])
        test_depth = 2.0
        test_pose = np.eye(4)
        
        world_point = transformer.pixel_to_world(test_pixel, test_depth, test_pose)
        pixel_back = transformer.world_to_pixel(world_point, test_pose)
        
        # Check if transformations work
        pixel_error = np.linalg.norm(test_pixel - pixel_back[:2]) if len(pixel_back) >= 2 else float('inf')
        
        print(f"  Basic transformation error: {pixel_error:.6f}")
        print(f"  Camera matrix: {transformer.fx}, {transformer.fy}")
        
        print("✓ Coordinate Transformer validated")
        return pixel_error < 10  # Allow some error due to depth projection
        
    except Exception as e:
        print(f"✗ Coordinate Transformer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run validation tests."""
    print("=" * 60)
    print("OrbyGlasses SLAM and 3D Mapping Validation")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("SLAM Enhancements", test_slam_features),
        ("Occupancy Grid Enhancements", test_occupancy_grid_enhancements),
        ("Movement Visualizer", test_movement_visualizer),
        ("Coordinate Transformer", test_coordinate_transformer),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    all_passed = True
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)