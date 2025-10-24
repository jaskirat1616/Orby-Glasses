"""
Fast Depth Visualization
Ultra-fast depth map coloring for real-time performance
"""

import cv2
import numpy as np


class FastDepthVisualizer:
    """
    Super fast depth visualization with dark colors.
    Uses custom dark colormap for better visibility.
    """

    def __init__(self, use_dark=True):
        """
        Initialize fast depth visualizer.

        Args:
            use_dark: Use dark color scheme (default True)
        """
        # Use dark colormap by default
        if use_dark:
            self.colormap = cv2.COLORMAP_TWILIGHT  # Dark blue/purple
        else:
            self.colormap = cv2.COLORMAP_JET  # Bright colors

    def visualize(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Fast depth visualization with dark colors.

        Args:
            depth_map: Depth map (any range)

        Returns:
            Colored depth map (BGR) with dark theme
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

        # Darken the image by 30% for better dark theme
        colored = (colored * 0.7).astype(np.uint8)

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
        Simple colored depth map with dark colors.

        Args:
            depth_map: Depth map (meters)

        Returns:
            Colored depth (BGR) with dark theme
        """
        h, w = depth_map.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        # Dark color scheme (darker versions)
        # Very close (< 0.5m): Dark red
        mask_very_close = depth_map < 0.5
        colored[mask_very_close] = [0, 0, 180]  # Dark red

        # Close (0.5-1.5m): Red/Orange
        mask_close = (depth_map >= 0.5) & (depth_map < 1.5)
        colored[mask_close] = [0, 50, 200]  # Orange-red

        # Medium (1.5-3m): Yellow-green
        mask_medium = (depth_map >= 1.5) & (depth_map < 3.0)
        colored[mask_medium] = [0, 150, 150]  # Dark yellow

        # Far (3-5m): Green
        mask_far = (depth_map >= 3.0) & (depth_map < 5.0)
        colored[mask_far] = [0, 120, 0]  # Dark green

        # Very far (>5m): Blue
        mask_very_far = depth_map >= 5.0
        colored[mask_very_far] = [100, 50, 0]  # Dark blue

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
