"""
Movement Visualizer for OrbyGlasses
Provides movement visualization for SLAM trajectory and path planning
"""

import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional, Dict
from collections import deque
import time


class MovementVisualizer:
    """
    Movement visualization system for SLAM trajectory and motion analysis.
    Provides real-time visualization of camera movement, velocity, and path planning.
    """
    
    def __init__(self, config):
        """
        Initialize the movement visualizer.
        
        Args:
            config: Configuration object containing visualization parameters
        """
        self.config = config
        self.enabled = config.get('movement_visualizer.enabled', True)
        
        # Visualization parameters
        self.canvas_size = config.get('movement_visualizer.canvas_size', (800, 800))
        self.trail_length = config.get('movement_visualizer.trail_length', 200)
        self.grid_size = config.get('movement_visualizer.grid_size', 20.0)  # meters
        
        # Trail and trajectory storage
        self.position_trail = deque(maxlen=self.trail_length)
        self.velocity_history = deque(maxlen=50)
        self.orientation_history = deque(maxlen=50)
        
        # Movement analysis
        self.movement_stats = {
            'total_distance': 0.0,
            'average_speed': 0.0,
            'current_speed': 0.0,
            'max_speed': 0.0,
            'direction_changes': 0
        }
        
        # Reference frame for relative movement
        self.reference_position = np.array([0.0, 0.0, 0.0])
        self.last_position = np.array([0.0, 0.0, 0.0])
        self.last_update_time = time.time()
        
        # Visualization settings
        self.show_velocity = config.get('movement_visualizer.show_velocity', True)
        self.show_orientation = config.get('movement_visualizer.show_orientation', True)
        self.show_grid = config.get('movement_visualizer.show_grid', True)
        self.show_stats = config.get('movement_visualizer.show_stats', True)
        
        logging.info("Movement Visualizer initialized")
        logging.info(f"  Trail length: {self.trail_length} positions")
        logging.info(f"  Canvas size: {self.canvas_size}")
        logging.info(f"  Grid size: {self.grid_size}m")

    def update(self, slam_result: Dict, current_time: Optional[float] = None):
        """
        Update movement visualization with new SLAM data.
        
        Args:
            slam_result: Dictionary containing SLAM results including pose and position
            current_time: Optional timestamp for velocity calculation
        """
        if not self.enabled:
            return
            
        if current_time is None:
            current_time = time.time()
        
        # Extract position from slam result
        position = np.array(slam_result.get('position', [0, 0, 0]))
        
        # Calculate velocity
        dt = current_time - self.last_update_time
        if dt > 0.001:  # Avoid division by zero
            displacement = position - self.last_position
            velocity = displacement / dt
            speed = np.linalg.norm(velocity)
            
            # Update velocity history
            self.velocity_history.append((velocity, speed, current_time))
            
            # Update movement statistics
            self.movement_stats['current_speed'] = speed
            self.movement_stats['max_speed'] = max(self.movement_stats['max_speed'], speed)
            self.movement_stats['average_speed'] = (
                (self.movement_stats['average_speed'] * (len(self.velocity_history) - 1) + speed) / 
                max(1, len(self.velocity_history))
            )
            
            # Update total distance
            self.movement_stats['total_distance'] += np.linalg.norm(displacement)

        # Store position in trail
        self.position_trail.append(position.copy())
        self.last_position = position.copy()
        self.last_update_time = current_time
        
        # Store orientation if available
        if 'pose' in slam_result:
            pose = slam_result['pose']
            # Extract orientation (simplified as yaw angle)
            rotation = pose[:3, :3]
            # Extract yaw angle (rotation around Z-axis)
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            self.orientation_history.append((yaw, current_time))

    def visualize(self) -> np.ndarray:
        """
        Create movement visualization canvas.
        
        Returns:
            Visualization image
        """
        if not self.enabled:
            # Return empty canvas if disabled
            return np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8)
        
        canvas = np.ones((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8) * 240  # Light gray background
        
        if len(self.position_trail) < 2:
            # Show waiting message if no data
            cv2.putText(canvas, "Waiting for movement data...",
                       (self.canvas_size[0]//2 - 100, self.canvas_size[1]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
            return canvas
        
        # Draw grid if enabled
        if self.show_grid:
            self._draw_grid(canvas)
        
        # Draw trajectory trail
        self._draw_trajectory(canvas)
        
        # Draw current position indicator
        self._draw_current_position(canvas)
        
        # Draw velocity vector if enabled
        if self.show_velocity and len(self.velocity_history) > 0:
            self._draw_velocity_vector(canvas)
        
        # Draw orientation indicator if enabled
        if self.show_orientation and len(self.orientation_history) > 0:
            self._draw_orientation(canvas)
        
        # Draw stats if enabled
        if self.show_stats:
            self._draw_statistics(canvas)
        
        # Draw title
        cv2.putText(canvas, "Movement Trajectory", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return canvas
    
    def _draw_grid(self, canvas: np.ndarray):
        """Draw coordinate grid on canvas."""
        center_x, center_y = self.canvas_size[0] // 2, self.canvas_size[1] // 2
        scale = min(self.canvas_size) / self.grid_size  # pixels per meter
        
        # Draw grid lines
        grid_spacing = 1.0  # meters
        grid_count = int(self.grid_size / grid_spacing) // 2
        
        for i in range(-grid_count, grid_count + 1):
            # Vertical lines
            x = int(center_x + i * grid_spacing * scale)
            if 0 <= x < self.canvas_size[0]:
                cv2.line(canvas, (x, 0), (x, self.canvas_size[1]), (220, 220, 220), 1)
                
            # Horizontal lines  
            y = int(center_y + i * grid_spacing * scale)
            if 0 <= y < self.canvas_size[1]:
                cv2.line(canvas, (0, y), (self.canvas_size[0], y), (220, 220, 220), 1)
        
        # Draw center cross
        cv2.line(canvas, (center_x, 0), (center_x, self.canvas_size[1]), (200, 200, 200), 2)
        cv2.line(canvas, (0, center_y), (self.canvas_size[0], center_y), (200, 200, 200), 2)
        
        # Label axes
        cv2.putText(canvas, "X", (self.canvas_size[0] - 30, center_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        cv2.putText(canvas, "Y", (center_x + 10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    def _world_to_canvas(self, pos: np.ndarray) -> Tuple[int, int]:
        """
        Convert world coordinates to canvas coordinates.
        
        Args:
            pos: World position [x, y, z]
            
        Returns:
            Canvas coordinates (x, y)
        """
        center_x, center_y = self.canvas_size[0] // 2, self.canvas_size[1] // 2
        scale = min(self.canvas_size) / self.grid_size  # pixels per meter
        
        canvas_x = int(center_x + pos[0] * scale)
        canvas_y = int(center_y - pos[1] * scale)  # Flip Y axis (image coordinates)
        
        # Ensure coordinates are within valid range
        canvas_x = max(0, min(self.canvas_size[0] - 1, canvas_x))
        canvas_y = max(0, min(self.canvas_size[1] - 1, canvas_y))
        
        return canvas_x, canvas_y

    def _draw_trajectory(self, canvas: np.ndarray):
        """Draw the movement trajectory."""
        if len(self.position_trail) < 2:
            return
        
        # Convert positions to canvas coordinates
        points = []
        for pos in self.position_trail:
            x, y = self._world_to_canvas(pos)
            # Only add points that are valid integers within canvas bounds
            if isinstance(x, (int, np.integer)) and isinstance(y, (int, np.integer)):
                if 0 <= x < self.canvas_size[0] and 0 <= y < self.canvas_size[1]:
                    points.append((int(x), int(y)))
        
        # Draw trajectory line
        for i in range(len(points) - 1):
            pt1 = points[i]
            pt2 = points[i+1]
            # Ensure points are valid for OpenCV
            if (isinstance(pt1, tuple) and len(pt1) == 2 and 
                isinstance(pt2, tuple) and len(pt2) == 2 and
                all(isinstance(coord, (int, np.integer)) for coord in pt1 + pt2)):
                
                alpha = i / len(points) if len(points) > 0 else 1.0  # Avoid division by zero
                color = (int(200 * alpha), 100, int(255 * (1 - alpha)))  # From blue to red
                cv2.line(canvas, pt1, pt2, color, 3)
        
        # Draw trajectory dots (more recent positions are brighter/green)
        for i, point in enumerate(points):
            if (isinstance(point, tuple) and len(point) == 2 and
                all(isinstance(coord, (int, np.integer)) for coord in point) and
                0 <= point[0] < self.canvas_size[0] and 0 <= point[1] < self.canvas_size[1]):
                
                alpha = i / len(points) if len(points) > 1 else 1.0
                size = 2 if i < len(points) - 10 else 4  # Larger for recent positions
                color = (0, int(255 * alpha), int(255 * (1 - alpha)))  # Green for recent
                cv2.circle(canvas, point, size, color, -1)

    def _draw_current_position(self, canvas: np.ndarray):
        """Draw indicator for current position."""
        if len(self.position_trail) == 0:
            return
        
        current_pos = self.position_trail[-1]
        x, y = self._world_to_canvas(current_pos)
        
        if not (0 <= x < self.canvas_size[0] and 0 <= y < self.canvas_size[1]):
            return
        
        # Draw current position as large circle
        cv2.circle(canvas, (x, y), 10, (0, 255, 0), -1)  # Green
        cv2.circle(canvas, (x, y), 12, (0, 100, 0), 2)   # Darker green border
        cv2.putText(canvas, "NOW", (x + 15, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)

    def _draw_velocity_vector(self, canvas: np.ndarray):
        """Draw velocity vector."""
        if len(self.velocity_history) < 1:
            return
        
        # Get most recent velocity
        velocity, speed, _ = self.velocity_history[-1]
        current_pos = self.position_trail[-1]
        
        start_x, start_y = self._world_to_canvas(current_pos)
        
        # Scale velocity vector for visualization
        scale_factor = 50  # pixels per unit of speed
        end_x = int(start_x + velocity[0] * scale_factor)
        end_y = int(start_y - velocity[1] * scale_factor)  # Flip Y axis
        
        if not (0 <= end_x < self.canvas_size[0] and 0 <= end_y < self.canvas_size[1]):
            return
        
        # Draw velocity vector
        cv2.arrowedLine(canvas, (start_x, start_y), (end_x, end_y), (255, 0, 0), 3, tipLength=0.2)
        
        # Draw speed information
        cv2.putText(canvas, f"Speed: {speed:.2f} m/s", (end_x + 10, end_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    def _draw_orientation(self, canvas: np.ndarray):
        """Draw orientation indicator."""
        if len(self.orientation_history) < 1:
            return
        
        current_pos = self.position_trail[-1]
        orientation, _ = self.orientation_history[-1]
        
        pos_x, pos_y = self._world_to_canvas(current_pos)
        
        # Draw orientation arrow (length scaled for visibility)
        arrow_length = 30
        end_x = int(pos_x + arrow_length * np.cos(orientation))
        end_y = int(pos_y - arrow_length * np.sin(orientation))  # Flip Y axis
        
        cv2.arrowedLine(canvas, (pos_x, pos_y), (end_x, end_y), (0, 0, 255), 2, tipLength=0.3)
        
        # Draw direction label
        cv2.putText(canvas, f"Dir: {np.degrees(orientation):.1f}°", 
                   (end_x + 10, end_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def _draw_statistics(self, canvas: np.ndarray):
        """Draw movement statistics."""
        stats_y = 50  # Starting Y position for stats
        line_height = 25
        
        # Draw background for stats
        overlay = canvas.copy()
        cv2.rectangle(overlay, (self.canvas_size[0] - 250, 40), 
                     (self.canvas_size[0] - 10, 180), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        
        # Draw stats text
        cv2.putText(canvas, f"Distance: {self.movement_stats['total_distance']:.2f}m", 
                   (self.canvas_size[0] - 240, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        stats_y += line_height
        cv2.putText(canvas, f"Current Speed: {self.movement_stats['current_speed']:.2f}m/s", 
                   (self.canvas_size[0] - 240, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        stats_y += line_height
        cv2.putText(canvas, f"Average Speed: {self.movement_stats['average_speed']:.2f}m/s", 
                   (self.canvas_size[0] - 240, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        stats_y += line_height
        cv2.putText(canvas, f"Max Speed: {self.movement_stats['max_speed']:.2f}m/s", 
                   (self.canvas_size[0] - 240, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        stats_y += line_height
        cv2.putText(canvas, f"Trajectory: {len(self.position_trail)} pts", 
                   (self.canvas_size[0] - 240, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def reset(self):
        """Reset all movement data."""
        self.position_trail.clear()
        self.velocity_history.clear()
        self.orientation_history.clear()
        
        self.movement_stats = {
            'total_distance': 0.0,
            'average_speed': 0.0,
            'current_speed': 0.0,
            'max_speed': 0.0,
            'direction_changes': 0
        }
        
        self.last_position = np.array([0.0, 0.0, 0.0])
        self.reference_position = np.array([0.0, 0.0, 0.0])
        self.last_update_time = time.time()
        
        logging.info("Movement visualizer reset")

    def get_trajectory(self) -> np.ndarray:
        """Get current trajectory as numpy array."""
        if len(self.position_trail) == 0:
            return np.zeros((0, 3))
        return np.array(list(self.position_trail))

    def get_velocity_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get velocity profile over time."""
        if len(self.velocity_history) == 0:
            return np.zeros(0), np.zeros(0)
        
        times = np.array([t for _, _, t in self.velocity_history])
        speeds = np.array([s for _, s, _ in self.velocity_history])
        
        return times, speeds

    def get_stats(self) -> Dict:
        """Get current movement statistics."""
        stats = self.movement_stats.copy()
        stats['position_count'] = len(self.position_trail)
        stats['velocity_samples'] = len(self.velocity_history)
        return stats


# Example usage function
def example_usage():
    """Example of how to use the MovementVisualizer."""
    import yaml
    
    # Create a mock config
    config = {
        'movement_visualizer.enabled': True,
        'movement_visualizer.canvas_size': [800, 800],
        'movement_visualizer.trail_length': 200,
        'movement_visualizer.grid_size': 20.0,
        'movement_visualizer.show_velocity': True,
        'movement_visualizer.show_orientation': True,
        'movement_visualizer.show_grid': True,
        'movement_visualizer.show_stats': True
    }
    
    # Create visualizer
    visualizer = MovementVisualizer(config)
    
    # Mock SLAM results for demonstration
    for i in range(100):
        t = i * 0.1
        # Simulate a spiral movement
        x = 2 * np.cos(t) + 0.1 * t
        y = 2 * np.sin(t) + 0.1 * t
        z = 0
        
        pose = np.eye(4)
        pose[0, 3] = x
        pose[1, 3] = y
        pose[2, 3] = z
        
        slam_result = {
            'position': [x, y, z],
            'pose': pose,
            'tracking_quality': 0.9,
            'num_matches': 50,
            'is_keyframe': False,
            'num_map_points': 100
        }
        
        # Update visualizer
        visualizer.update(slam_result, time.time() - 10 + t)  # Simulate time progression
        
        # Get visualization
        vis_img = visualizer.visualize()
        
        # Display (in a real app, you'd show this in a window)
        if i % 20 == 0:  # Print stats every 20 iterations
            stats = visualizer.get_stats()
            print(f"Iteration {i}: Distance={stats['total_distance']:.2f}m, "
                  f"Speed={stats['current_speed']:.2f}m/s")
    
    print("Movement visualization example completed")


if __name__ == "__main__":
    example_usage()