"""
Unit tests for 3D Occupancy Grid Mapping
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from occupancy_grid_3d import OccupancyGrid3D
from utils import ConfigManager


class TestOccupancyGrid3D(unittest.TestCase):
    """Test suite for 3D Occupancy Grid."""

    def setUp(self):
        """Set up test fixtures."""
        # Create minimal config for testing
        self.config = ConfigManager()
        self.config.config = {
            'occupancy_grid_3d': {
                'enabled': True,
                'grid_size': [10.0, 10.0, 3.0],
                'resolution': 0.1,
                'max_range': 5.0,
                'min_range': 0.1,
                'log_odds_occupied': 0.7,
                'log_odds_free': -0.4,
                'log_odds_min': -5.0,
                'log_odds_max': 5.0,
                'subsample_step': 4,
                'update_interval': 0.1,
                'visualize': False
            },
            'mapping3d': {
                'fx': 500,
                'fy': 500
            },
            'camera': {
                'width': 320,
                'height': 320
            }
        }

        self.grid = OccupancyGrid3D(self.config)

    def test_initialization(self):
        """Test grid initialization."""
        self.assertIsNotNone(self.grid)
        self.assertTrue(self.grid.enabled)
        self.assertEqual(self.grid.resolution, 0.1)
        self.assertEqual(self.grid.grid_size, [10.0, 10.0, 3.0])
        self.assertEqual(len(self.grid.grid), 0)  # Empty at start

    def test_world_to_voxel_conversion(self):
        """Test coordinate conversion from world to voxel."""
        # Test origin
        voxel = self.grid.world_to_voxel(np.array([0.0, 0.0, 0.0]))
        self.assertEqual(voxel[0], 50)  # Grid origin is at -5m, so 0m is at index 50
        self.assertEqual(voxel[1], 50)
        self.assertEqual(voxel[2], 0)

        # Test positive offset
        voxel = self.grid.world_to_voxel(np.array([1.0, 1.0, 1.0]))
        self.assertEqual(voxel[0], 60)  # 1m / 0.1m = 10 voxels from origin
        self.assertEqual(voxel[1], 60)
        self.assertEqual(voxel[2], 10)

    def test_voxel_to_world_conversion(self):
        """Test coordinate conversion from voxel to world."""
        # Test voxel center
        world = self.grid.voxel_to_world((50, 50, 0))
        np.testing.assert_array_almost_equal(world, np.array([0.05, 0.05, 0.05]))

        # Test another voxel
        world = self.grid.voxel_to_world((60, 60, 10))
        np.testing.assert_array_almost_equal(world, np.array([1.05, 1.05, 1.05]))

    def test_is_valid_voxel(self):
        """Test voxel bounds checking."""
        # Valid voxels
        self.assertTrue(self.grid.is_valid_voxel((0, 0, 0)))
        self.assertTrue(self.grid.is_valid_voxel((50, 50, 15)))
        self.assertTrue(self.grid.is_valid_voxel((99, 99, 29)))

        # Invalid voxels (out of bounds)
        self.assertFalse(self.grid.is_valid_voxel((-1, 0, 0)))
        self.assertFalse(self.grid.is_valid_voxel((0, -1, 0)))
        self.assertFalse(self.grid.is_valid_voxel((0, 0, -1)))
        self.assertFalse(self.grid.is_valid_voxel((100, 0, 0)))
        self.assertFalse(self.grid.is_valid_voxel((0, 100, 0)))
        self.assertFalse(self.grid.is_valid_voxel((0, 0, 30)))

    def test_bresenham_3d_straight_line(self):
        """Test 3D Bresenham line algorithm with straight lines."""
        # Test horizontal line
        voxels = self.grid._bresenham_3d((0, 0, 0), (5, 0, 0))
        self.assertEqual(len(voxels), 6)  # Start + 5 steps
        self.assertEqual(voxels[0], (0, 0, 0))
        self.assertEqual(voxels[-1], (5, 0, 0))

        # Test vertical line
        voxels = self.grid._bresenham_3d((0, 0, 0), (0, 0, 5))
        self.assertEqual(len(voxels), 6)
        self.assertEqual(voxels[0], (0, 0, 0))
        self.assertEqual(voxels[-1], (0, 0, 5))

    def test_bresenham_3d_diagonal_line(self):
        """Test 3D Bresenham with diagonal lines."""
        # Test 3D diagonal
        voxels = self.grid._bresenham_3d((0, 0, 0), (5, 5, 5))
        self.assertGreater(len(voxels), 5)
        self.assertEqual(voxels[0], (0, 0, 0))
        self.assertEqual(voxels[-1], (5, 5, 5))

    def test_update_voxel_occupied(self):
        """Test updating a voxel as occupied."""
        voxel = (50, 50, 10)

        # Initially unknown (log-odds = 0)
        self.assertEqual(self.grid.grid.get(voxel, 0.0), 0.0)

        # Update as occupied
        self.grid._update_voxel(voxel, occupied=True)
        self.assertGreater(self.grid.grid[voxel], 0.0)
        self.assertIn(voxel, self.grid.occupied_voxels)

        # Update again (should increase confidence)
        prev_value = self.grid.grid[voxel]
        self.grid._update_voxel(voxel, occupied=True)
        self.assertGreater(self.grid.grid[voxel], prev_value)

    def test_update_voxel_free(self):
        """Test updating a voxel as free space."""
        voxel = (50, 50, 10)

        # Update as free
        self.grid._update_voxel(voxel, occupied=False)
        self.assertLess(self.grid.grid[voxel], 0.0)
        self.assertIn(voxel, self.grid.free_voxels)

        # Update again (should decrease confidence)
        prev_value = self.grid.grid[voxel]
        self.grid._update_voxel(voxel, occupied=False)
        self.assertLess(self.grid.grid[voxel], prev_value)

    def test_ray_cast_update(self):
        """Test ray casting update."""
        # Cast ray from origin to point
        start = np.array([0.0, 0.0, 1.0])
        end = np.array([2.0, 0.0, 1.0])

        updates = self.grid._ray_cast_update(start, end)
        self.assertGreater(updates, 0)

        # Check that endpoint is marked as occupied
        end_voxel = self.grid.world_to_voxel(end)
        self.assertGreater(self.grid.grid.get(end_voxel, 0.0), 0.0)

    def test_update_from_depth(self):
        """Test updating grid from depth map."""
        # Create synthetic depth map (320x320)
        depth_map = np.ones((320, 320), dtype=np.float32) * 0.5  # 50% = 2.5m depth

        # Camera pose at origin looking forward
        camera_pose = np.eye(4, dtype=np.float32)
        camera_pose[2, 3] = 1.0  # 1m height

        # Update grid
        initial_size = len(self.grid.grid)
        self.grid.update_from_depth(depth_map, camera_pose)

        # Grid should have been updated
        self.assertGreater(len(self.grid.grid), initial_size)
        self.assertGreater(self.grid.total_updates, 0)

    def test_is_occupied(self):
        """Test occupancy checking."""
        # Mark a voxel as occupied
        point = np.array([1.0, 1.0, 1.0])
        voxel = self.grid.world_to_voxel(point)

        # Initially not occupied
        self.assertFalse(self.grid.is_occupied(point))

        # Mark as occupied (multiple times for confidence)
        for _ in range(5):
            self.grid._update_voxel(voxel, occupied=True)

        # Now should be occupied
        self.assertTrue(self.grid.is_occupied(point))

    def test_get_occupancy_probability(self):
        """Test occupancy probability calculation."""
        point = np.array([1.0, 1.0, 1.0])
        voxel = self.grid.world_to_voxel(point)

        # Initially unknown (0.5)
        prob = self.grid.get_occupancy_probability(point)
        self.assertAlmostEqual(prob, 0.5, places=1)

        # Mark as occupied
        for _ in range(10):
            self.grid._update_voxel(voxel, occupied=True)

        # Should have high probability
        prob = self.grid.get_occupancy_probability(point)
        self.assertGreater(prob, 0.9)

        # Mark as free
        for _ in range(20):
            self.grid._update_voxel(voxel, occupied=False)

        # Should have low probability
        prob = self.grid.get_occupancy_probability(point)
        self.assertLess(prob, 0.1)

    def test_get_occupied_voxels(self):
        """Test getting occupied voxel positions."""
        # Initially empty
        occupied = self.grid.get_occupied_voxels(threshold=0.0)
        self.assertEqual(len(occupied), 0)

        # Mark some voxels as occupied
        for i in range(5):
            voxel = (50 + i, 50, 10)
            for _ in range(5):
                self.grid._update_voxel(voxel, occupied=True)

        # Get occupied voxels
        occupied = self.grid.get_occupied_voxels(threshold=0.0)
        self.assertEqual(len(occupied), 5)

    def test_get_2d_slice(self):
        """Test extracting 2D slice from grid."""
        # Create some occupied voxels at height z=1.0
        for i in range(10):
            point = np.array([float(i) * 0.1, 0.0, 1.0])
            voxel = self.grid.world_to_voxel(point)
            for _ in range(5):
                self.grid._update_voxel(voxel, occupied=True)

        # Extract slice at z=1.0
        slice_2d = self.grid.get_2d_slice(z_height=1.0)

        # Should be 2D array
        self.assertEqual(len(slice_2d.shape), 2)
        self.assertEqual(slice_2d.shape[0], self.grid.grid_dims[1])
        self.assertEqual(slice_2d.shape[1], self.grid.grid_dims[0])

    def test_clear_grid(self):
        """Test clearing the grid."""
        # Add some data
        for i in range(10):
            voxel = (50 + i, 50, 10)
            self.grid._update_voxel(voxel, occupied=True)

        self.assertGreater(len(self.grid.grid), 0)

        # Clear
        self.grid.clear()

        # Should be empty
        self.assertEqual(len(self.grid.grid), 0)
        self.assertEqual(len(self.grid.occupied_voxels), 0)
        self.assertEqual(len(self.grid.free_voxels), 0)
        self.assertEqual(self.grid.total_updates, 0)

    def test_get_stats(self):
        """Test statistics collection."""
        # Add some data
        for i in range(10):
            voxel = (50 + i, 50, 10)
            self.grid._update_voxel(voxel, occupied=True)

        stats = self.grid.get_stats()

        # Check stats structure
        self.assertIn('total_voxels_stored', stats)
        self.assertIn('occupied_voxels', stats)
        self.assertIn('free_voxels', stats)
        self.assertIn('total_updates', stats)
        self.assertIn('grid_size_meters', stats)
        self.assertIn('resolution', stats)
        self.assertIn('memory_usage_mb', stats)

        # Check values
        self.assertEqual(stats['total_voxels_stored'], 10)
        self.assertEqual(stats['resolution'], 0.1)

    def test_log_odds_clamping(self):
        """Test that log-odds values are clamped correctly."""
        voxel = (50, 50, 10)

        # Update many times as occupied
        for _ in range(100):
            self.grid._update_voxel(voxel, occupied=True)

        # Should be clamped to max
        self.assertLessEqual(self.grid.grid[voxel], self.grid.log_odds_max)

        # Update many times as free
        for _ in range(200):
            self.grid._update_voxel(voxel, occupied=False)

        # Should be clamped to min
        self.assertGreaterEqual(self.grid.grid[voxel], self.grid.log_odds_min)

    def test_multiple_depth_updates(self):
        """Test multiple sequential depth map updates."""
        # Create depth maps with different patterns
        depth_map1 = np.ones((320, 320), dtype=np.float32) * 0.3
        depth_map2 = np.ones((320, 320), dtype=np.float32) * 0.5
        depth_map3 = np.ones((320, 320), dtype=np.float32) * 0.7

        camera_pose = np.eye(4, dtype=np.float32)
        camera_pose[2, 3] = 1.0

        # Sequential updates
        self.grid.update_from_depth(depth_map1, camera_pose)
        count1 = len(self.grid.grid)

        self.grid.update_from_depth(depth_map2, camera_pose)
        count2 = len(self.grid.grid)

        self.grid.update_from_depth(depth_map3, camera_pose)
        count3 = len(self.grid.grid)

        # Grid should grow with updates
        self.assertGreater(count2, 0)
        self.assertGreater(count3, 0)


def run_tests():
    """Run all tests and print results."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOccupancyGrid3D)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*70)
    print("3D Occupancy Grid Test Results")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
