"""
Test script for OrbyGlasses SLAM and 3D mapping
Validates the deep analysis, 3D occupancy grid, SLAM without IMU, and movement visualization
"""

import sys
import os
import numpy as np
import cv2
import time
from typing import Dict, Tuple

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import ConfigManager
from src.slam import MonocularSLAM
from src.occupancy_grid_3d import OccupancyGrid3D
from src.movement_visualizer import MovementVisualizer
from src.coordinate_transformer import CoordinateTransformer


def create_test_config():
    """Create a test configuration."""
    config_dict = {
        # SLAM settings - parameters for better accuracy
        'slam.enabled': True,
        'slam.orb_features': 2000,
        'slam.min_matches': 20,
        'slam.min_tracked_features': 15,
        'slam.keyframe_threshold': 25,
        'slam.scale_threshold': 0.1,
        'slam.reprojection_threshold': 3.0,
        'slam.min_depth': 0.1,
        'slam.max_depth': 10.0,
        'slam.pose_alpha': 0.7,
        'slam.loop_closure': False,
        'slam.bundle_adjustment': False,
        'slam.temporal_consistency_check': True,
        'slam.max_position_jump': 0.5,
        'slam.max_rotation_jump': 0.5,
        'slam.scale_factor': 1.2,
        'slam.nlevels': 8,
        'slam.edge_threshold': 10,
        'slam.fast_threshold': 20,

        # 3D Mapping settings
        'mapping3d.fx': 500.0,
        'mapping3d.fy': 500.0,

        # 3D Occupancy Grid settings - for accuracy
        'occupancy_grid_3d.enabled': True,
        'occupancy_grid_3d.grid_size': [10.0, 10.0, 3.0],
        'occupancy_grid_3d.resolution': 0.1,
        'occupancy_grid_3d.log_odds_occupied': 0.8,
        'occupancy_grid_3d.log_odds_free': -0.6,
        'occupancy_grid_3d.log_odds_min': -6.0,
        'occupancy_grid_3d.log_odds_max': 6.0,
        'occupancy_grid_3d.depth_uncertainty': 0.05,
        'occupancy_grid_3d.max_range': 5.0,
        'occupancy_grid_3d.min_range': 0.1,
        'occupancy_grid_3d.range_variance': 0.02,
        'occupancy_grid_3d.temporal_consistency': True,
        'occupancy_grid_3d.min_observations': 2,
        'occupancy_grid_3d.confirmation_threshold': 1.0,
        'occupancy_grid_3d.decay_rate': 0.001,
        'occupancy_grid_3d.near_resolution': 0.05,
        'occupancy_grid_3d.near_distance': 1.0,
        'occupancy_grid_3d.far_resolution': 0.2,
        'occupancy_grid_3d.update_interval': 0.1,
        'occupancy_grid_3d.subsample_step': 2,
        'occupancy_grid_3d.temporal_filtering': True,
        'occupancy_grid_3d.max_history_length': 10,
        'occupancy_grid_3d.history_decay': 0.95,
        'occupancy_grid_3d.visualize': True,
        'occupancy_grid_3d.show_2d_slice': False,
        'occupancy_grid_3d.max_voxels_per_ray': 200,

        # Movement Visualizer settings
        'movement_visualizer.enabled': True,
        'movement_visualizer.canvas_size': [800, 800],
        'movement_visualizer.trail_length': 200,
        'movement_visualizer.grid_size': 20.0,
        'movement_visualizer.show_velocity': True,
        'movement_visualizer.show_orientation': True,
        'movement_visualizer.show_grid': True,
        'movement_visualizer.show_stats': True,

        # Camera settings
        'camera.width': 320,
        'camera.height': 320,
    }
    
    # Create a mock ConfigManager that returns values from the dict
    class MockConfig:
        def get(self, key, default=None):
            return config_dict.get(key, default)
    
    return MockConfig()


def generate_test_frame(frame_num: int, width: int = 320, height: int = 320) -> np.ndarray:
    """Generate a synthetic test frame for SLAM testing."""
    # Create a base image with some texture
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some synthetic features that can be tracked
    center_x, center_y = width // 2, height // 2
    
    # Add a moving pattern to simulate camera movement
    offset_x = int(10 * np.sin(frame_num * 0.1))
    offset_y = int(10 * np.cos(frame_num * 0.1))
    
    # Draw some distinctive patterns
    cv2.rectangle(img, (center_x - 50 + offset_x, center_y - 50 + offset_y),
                  (center_x - 30 + offset_x, center_y - 30 + offset_y), (255, 0, 0), -1)
    cv2.circle(img, (center_x + 30 + offset_x, center_y - 40 + offset_y), 15, (0, 255, 0), -1)
    cv2.circle(img, (center_x - 40 + offset_x, center_y + 30 + offset_y), 12, (0, 0, 255), -1)
    
    # Add some random texture for feature detection
    for _ in range(20):
        pt = (np.random.randint(0, width), np.random.randint(0, height))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.circle(img, pt, 2, color, -1)
    
    return img


def generate_test_depth(frame_num: int, width: int = 320, height: int = 320) -> np.ndarray:
    """Generate a synthetic depth map for testing."""
    depth = np.ones((height, width), dtype=np.float32) * 0.8  # Default depth of 0.8m
    
    # Create some depth variation to simulate objects
    center_x, center_y = width // 2, height // 2
    
    # Add a "wall" in the center
    for y in range(height):
        for x in range(width):
            dist_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            if dist_center < 80:
                depth[y, x] = 0.5  # Closer
            elif dist_center < 120:
                depth[y, x] = 0.7
            else:
                depth[y, x] = 1.0  # Further away
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.05, depth.shape).astype(np.float32)
    depth = np.clip(depth + noise, 0.1, 5.0)
    
    # Normalize to 0-1 range
    depth_normalized = depth / 5.0  # Assuming max depth is 5m
    
    return depth_normalized


def test_slam_accuracy():
    """Test SLAM accuracy and performance."""
    print("Testing SLAM accuracy...")
    
    config = create_test_config()
    slam = MonocularSLAM(config)
    
    # Test with synthetic frames
    positions = []
    tracking_quality = []
    processing_times = []
    
    for i in range(50):  # Test with 50 frames
        start_time = time.time()
        
        frame = generate_test_frame(i)
        depth_map = generate_test_depth(i)
        
        result = slam.process_frame(frame, depth_map)
        
        end_time = time.time()
        
        positions.append(result['position'])
        tracking_quality.append(result['tracking_quality'])
        processing_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        if i % 10 == 0:
            print(f"Processed frame {i+1}/50, tracking quality: {result['tracking_quality']:.2f}")
    
    # Analyze results
    avg_quality = np.mean(tracking_quality)
    avg_time = np.mean(processing_times)
    
    print(f"SLAM Results:")
    print(f"  Average tracking quality: {avg_quality:.3f}")
    print(f"  Average processing time: {avg_time:.2f}ms")
    print(f"  Total positions tracked: {len(positions)}")
    
    # Calculate trajectory length
    total_distance = 0
    for i in range(1, len(positions)):
        dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
        total_distance += dist
    
    print(f"  Total trajectory length: {total_distance:.3f}m")
    
    return avg_quality > 0.3  # Should have reasonable tracking quality


def test_occupancy_grid():
    """Test 3D occupancy grid mapping."""
    print("\nTesting 3D Occupancy Grid...")
    
    config = create_test_config()
    grid = OccupancyGrid3D(config)
    
    # Initialize with identity pose
    initial_pose = np.eye(4, dtype=np.float32)
    
    # Simulate a few frames with depth data
    for i in range(20):
        depth_map = generate_test_depth(i)
        
        # Move the camera slightly for each frame to build the map
        pose = initial_pose.copy()
        pose[0, 3] = i * 0.1  # Move along X axis
        pose[1, 3] = 0.05 * np.sin(i * 0.2)  # Small oscillation in Y
        
        grid.update_from_depth(depth_map, pose)
        
        if i % 5 == 0:
            stats = grid.get_stats()
            print(f"Updated grid {i+1}/20, occupied voxels: {stats['occupied_voxels']}")
    
    # Get statistics
    stats = grid.get_stats()
    occupied_voxels = len(grid.occupied_voxels)
    
    print(f"Occupancy Grid Results:")
    print(f"  Total updates: {stats['total_updates']}")
    print(f"  Occupied voxels: {occupied_voxels}")
    print(f"  Grid size: {stats['grid_size_meters']}")
    
    # Test visualization
    try:
        vis = grid.visualize_3d_interactive(np.array([0, 0, 0]))
        print("  ✓ Visualization successful")
    except Exception as e:
        print(f"  ✗ Visualization failed: {e}")
    
    return occupied_voxels > 0  # Should have created some occupied voxels


def test_coordinate_transformer():
    """Test coordinate transformation system."""
    print("\nTesting Coordinate Transformer...")
    
    config = create_test_config()
    transformer = CoordinateTransformer(config)
    
    # Test basic transformations
    pixel = np.array([160, 120])  # Point in image
    depth = 2.0  # 2 meters away
    pose = np.eye(4)  # Identity pose
    
    # Convert pixel to world
    world_point = transformer.pixel_to_world(pixel, depth, pose)
    print(f"  Pixel {pixel} at depth {depth}m -> World: {world_point}")
    
    # Convert back to pixel
    pixel_back = transformer.world_to_pixel(world_point, pose)
    print(f"  World {world_point} -> Pixel: {pixel_back}")
    
    # Check if we get approximately the same pixel back
    pixel_error = np.linalg.norm(pixel - pixel_back)
    print(f"  Pixel reprojection error: {pixel_error:.4f}px")
    
    # Test with a transformed pose
    pose_transformed = pose.copy()
    pose_transformed[0, 3] = 1.0  # Move camera 1m along X
    pose_transformed[1, 3] = 0.5  # Move camera 0.5m along Y
    pose_transformed[2, 3] = 0.2  # Move camera 0.2m along Z
    
    world_point2 = transformer.pixel_to_world(pixel, depth, pose_transformed)
    print(f"  With transformed pose, world: {world_point2}")
    
    # Test rotation conversion
    euler = np.array([0.1, 0.2, 0.3])  # Small rotations
    rot_matrix = transformer.euler_to_rotation_matrix(euler)
    euler_back = transformer.rotation_matrix_to_euler(rot_matrix)
    euler_error = np.linalg.norm(euler - euler_back)
    print(f"  Euler rotation error: {euler_error:.6f}")
    
    print(f"  ✓ Coordinate transformation tests completed")
    
    return pixel_error < 1.0 and euler_error < 1e-10


def test_movement_visualization():
    """Test movement visualization."""
    print("\nTesting Movement Visualization...")
    
    config = create_test_config()
    visualizer = MovementVisualizer(config)
    
    # Simulate some SLAM results
    for i in range(30):
        # Create simulated SLAM result
        t = i * 0.1  # Time step
        x = 0.5 * np.cos(t)  # Circular motion
        y = 0.5 * np.sin(t)  # Circular motion
        z = 0.0
        
        pose = np.eye(4)
        pose[0, 3] = x
        pose[1, 3] = y
        pose[2, 3] = z
        
        slam_result = {
            'position': [x, y, z],
            'pose': pose,
            'tracking_quality': 0.9,
            'num_matches': 40,
            'is_keyframe': False,
            'num_map_points': 50,
            'relative_movement': [0.1 * np.cos(t), 0.1 * np.sin(t), 0, 0, 0, 0.1]
        }
        
        visualizer.update(slam_result, time.time() - 10 + t)
        
        if i % 10 == 0:
            stats = visualizer.get_stats()
            print(f"  Updated visualizer {i+1}/30, distance: {stats['total_distance']:.3f}m")
    
    # Get visualization
    vis_img = visualizer.visualize()
    print(f"  Visualization shape: {vis_img.shape}")
    
    # Get stats
    stats = visualizer.get_stats()
    print(f"  Final stats - Distance: {stats['total_distance']:.3f}m, "
          f"Speed: {stats['current_speed']:.3f}m/s")
    
    print(f"  ✓ Movement visualization test completed")
    
    return vis_img is not None


def main():
    """Run all tests."""
    print("=" * 60)
    print("OrbyGlasses SLAM and 3D Mapping Validation")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test SLAM accuracy
    try:
        slam_ok = test_slam_accuracy()
        print(f"SLAM Test: {'✓ PASSED' if slam_ok else '✗ FAILED'}")
        all_tests_passed = all_tests_passed and slam_ok
    except Exception as e:
        print(f"SLAM Test: ✗ FAILED with error: {e}")
        all_tests_passed = False
    
    # Test occupancy grid
    try:
        grid_ok = test_occupancy_grid()
        print(f"Occupancy Grid Test: {'✓ PASSED' if grid_ok else '✗ FAILED'}")
        all_tests_passed = all_tests_passed and grid_ok
    except Exception as e:
        print(f"Occupancy Grid Test: ✗ FAILED with error: {e}")
        all_tests_passed = False
    
    # Test coordinate transformation
    try:
        coord_ok = test_coordinate_transformer()
        print(f"Coordinate Transformation Test: {'✓ PASSED' if coord_ok else '✗ FAILED'}")
        all_tests_passed = all_tests_passed and coord_ok
    except Exception as e:
        print(f"Coordinate Transformation Test: ✗ FAILED with error: {e}")
        all_tests_passed = False
    
    # Test movement visualization
    try:
        mv_ok = test_movement_visualization()
        print(f"Movement Visualization Test: {'✓ PASSED' if mv_ok else '✗ FAILED'}")
        all_tests_passed = all_tests_passed and mv_ok
    except Exception as e:
        print(f"Movement Visualization Test: ✗ FAILED with error: {e}")
        all_tests_passed = False
    
    print("\n" + "=" * 60)
    print(f"Overall Result: {'✓ ALL TESTS PASSED' if all_tests_passed else '✗ SOME TESTS FAILED'}")
    print("=" * 60)
    
    return all_tests_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)