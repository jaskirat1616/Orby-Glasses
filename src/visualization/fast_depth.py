"""
Fast Depth Visualization
Ultra-fast depth map coloring for real-time performance
"""

import cv2
import numpy as np


class FastDepthVisualizer:
    """
    Super fast depth visualization.
    Uses OpenCV colormaps for maximum speed.
    """

    def __init__(self):
        """Initialize fast depth visualizer."""
        # Use JET colormap (very fast)
        self.colormap = cv2.COLORMAP_JET

    def visualize(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Fast depth visualization.

        Args:
            depth_map: Depth map (any range)

        Returns:
            Colored depth map (BGR)
        """
        # Normalize to 0-255
        depth_min = np.min(depth_map[depth_map > 0]) if np.any(depth_map > 0) else 0
        depth_max = np.max(depth_map)

        if depth_max > depth_min:
            normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(depth_map, dtype=np.uint8)

        # Apply colormap (hardware accelerated)
        colored = cv2.applyColorMap(normalized, self.colormap)

        return colored


class SimpleDepthVisualizer:
    """
    Simple dark-themed depth visualization.
    Faster than full obsidian colormap.
    """

    def __init__(self):
        """Initialize simple visualizer."""
        pass

    def visualize(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Simple colored depth map.

        Args:
            depth_map: Depth map (meters)

        Returns:
            Colored depth (BGR)
        """
        h, w = depth_map.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        # Simple thresholding (very fast)
        # Close (< 1m): Red
        mask_close = depth_map < 1.0
        colored[mask_close] = [0, 0, 255]

        # Medium (1-3m): Yellow
        mask_medium = (depth_map >= 1.0) & (depth_map < 3.0)
        colored[mask_medium] = [0, 255, 255]

        # Far (>3m): Green
        mask_far = depth_map >= 3.0
        colored[mask_far] = [0, 255, 0]

        return colored


# Example usage
if __name__ == "__main__":
    import time

    # Create test depth
    depth = np.random.rand(480, 640) * 10

    # Test fast visualizer
    fast_viz = FastDepthVisualizer()

    start = time.time()
    for _ in range(100):
        result = fast_viz.visualize(depth)
    elapsed = time.time() - start

    print(f"Fast visualizer: {100/elapsed:.1f} FPS")

    # Test simple visualizer
    simple_viz = SimpleDepthVisualizer()

    start = time.time()
    for _ in range(100):
        result = simple_viz.visualize(depth)
    elapsed = time.time() - start

    print(f"Simple visualizer: {100/elapsed:.1f} FPS")

    print("\nUse FastDepthVisualizer for maximum speed!")
