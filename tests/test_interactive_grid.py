"""
Quick test for interactive 3D occupancy grid visualization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import cv2
from occupancy_grid_3d import OccupancyGrid3D
from core.utils import ConfigManager

def main():
    print("Testing Interactive 3D Occupancy Grid Visualization")
    print("=" * 60)

    # Create config
    config = ConfigManager()
    config.config = {
        'occupancy_grid_3d': {
            'enabled': True,
            'grid_size': [10.0, 10.0, 3.0],
            'resolution': 0.2,  # Larger voxels for testing
            'max_range': 5.0,
            'min_range': 0.1,
            'log_odds_occupied': 0.7,
            'log_odds_free': -0.4,
            'log_odds_min': -5.0,
            'log_odds_max': 5.0,
            'subsample_step': 4,
            'update_interval': 0.1,
            'visualize': True
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

    # Initialize grid
    grid = OccupancyGrid3D(config)
    print(f"✓ Grid initialized: {grid.grid_size} @ {grid.resolution}m resolution")

    # Create some test obstacles (simulating walls and objects)
    print("\nAdding test obstacles...")

    # Front wall
    for x in range(-20, 21, 2):
        for z in range(0, 15, 2):
            point = np.array([x * 0.1, 2.0, z * 0.1])
            voxel = grid.world_to_voxel(point)
            if grid.is_valid_voxel(voxel):
                for _ in range(5):
                    grid._update_voxel(voxel, occupied=True)

    # Side walls
    for y in range(-20, 21, 2):
        for z in range(0, 15, 2):
            # Left wall
            point = np.array([-2.0, y * 0.1, z * 0.1])
            voxel = grid.world_to_voxel(point)
            if grid.is_valid_voxel(voxel):
                for _ in range(5):
                    grid._update_voxel(voxel, occupied=True)

            # Right wall
            point = np.array([2.0, y * 0.1, z * 0.1])
            voxel = grid.world_to_voxel(point)
            if grid.is_valid_voxel(voxel):
                for _ in range(5):
                    grid._update_voxel(voxel, occupied=True)

    # Add some obstacles in the middle
    for i in range(10):
        point = np.array([
            np.random.uniform(-1.5, 1.5),
            np.random.uniform(-1.5, 1.5),
            np.random.uniform(0.0, 2.0)
        ])
        voxel = grid.world_to_voxel(point)
        if grid.is_valid_voxel(voxel):
            for _ in range(5):
                grid._update_voxel(voxel, occupied=True)

    stats = grid.get_stats()
    print(f"✓ Added obstacles: {stats['occupied_voxels']} occupied voxels")

    # Camera position
    camera_pos = np.array([0.0, 0.0, 1.5])

    print("\n" + "=" * 60)
    print("INTERACTIVE CONTROLS:")
    print("=" * 60)
    print("Mouse Wheel       : Zoom in/out")
    print("Arrow Keys        : Pan view (Up/Down/Left/Right)")
    print("Q/E              : Rotate around Z axis")
    print("W/S              : Rotate around X axis (tilt)")
    print("+/-              : Zoom in/out (alternative)")
    print("R                : Reset view")
    print("ESC or Q (lower) : Quit")
    print("=" * 60)
    print("\nShowing visualization window...")

    # Setup window
    cv2.namedWindow('3D Occupancy Grid', cv2.WINDOW_NORMAL)

    # Mouse callback
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:  # Scroll up
                grid.handle_mouse_wheel(1)
            else:  # Scroll down
                grid.handle_mouse_wheel(-1)

    cv2.setMouseCallback('3D Occupancy Grid', mouse_callback)

    # Main loop
    while True:
        # Generate visualization
        vis = grid.visualize_3d_interactive(camera_pos)

        # Display
        cv2.imshow('3D Occupancy Grid', vis)

        # Handle keyboard
        key = cv2.waitKey(30) & 0xFF

        if key == 27 or key == ord('q'):  # ESC or 'q'
            break

        # Update view
        if grid.update_view_controls(key):
            pass  # View updated, will redraw

    cv2.destroyAllWindows()
    print("\nTest complete!")

if __name__ == '__main__':
    main()
