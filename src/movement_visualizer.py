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
        Update movement visualization with new SLAM data using enhanced accuracy methods.
        
        Args:
            slam_result: Dictionary containing SLAM results including pose, position, and tracking quality
            current_time: Optional timestamp for velocity calculation
        """
        if not self.enabled:
            return
            
        if current_time is None:
            current_time = time.time()
        
        # Extract position and tracking quality from slam result
        position = np.array(slam_result.get('position', [0, 0, 0]))
        tracking_quality = slam_result.get('tracking_quality', 0.5)  # Default to medium quality
        
        # Calculate velocity with quality-based filtering
        dt = current_time - self.last_update_time
        if dt > 0.001:  # Avoid division by zero
            displacement = position - self.last_position
            
            # Calculate raw velocity
            raw_velocity = displacement / dt
            raw_speed = np.linalg.norm(raw_velocity)
            
            # Apply quality-based filtering to velocity
            if len(self.velocity_history) > 0:
                # Weight current measurement by tracking quality
                last_velocity_entry = self.velocity_history[-1]
                
                # Handle both old and new velocity history formats properly
                if len(last_velocity_entry) >= 4:  # New format: (velocity, speed, time, quality)
                    prev_velocity, prev_speed, _, _ = last_velocity_entry
                elif len(last_velocity_entry) >= 3:  # Old format: (velocity, speed, time)
                    prev_velocity, prev_speed, _ = last_velocity_entry
                else:  # Very old format: (velocity, speed)
                    prev_velocity, prev_speed = last_velocity_entry
                
                # Check for velocity outliers based on tracking quality
                velocity_diff = np.linalg.norm(raw_velocity - prev_velocity)
                max_reasonable_change = prev_speed * 2.0  # Reasonable acceleration limit
                
                if velocity_diff > max_reasonable_change and tracking_quality < 0.5:
                    # Use previous velocity if current measurement is noisy
                    raw_velocity = prev_velocity
                    raw_speed = prev_speed
            
            # Store velocity with quality weighting
            self.velocity_history.append((raw_velocity, raw_speed, current_time, tracking_quality))
            
            # Update movement statistics with quality filtering
            # Use exponentially weighted moving average for smoother results
            if 'current_speed' not in self.movement_stats or self.movement_stats['current_speed'] == 0:
                self.movement_stats['current_speed'] = raw_speed
            else:
                # Use quality-weighted smoothing
                alpha = min(0.3, tracking_quality)  # Higher quality = more weight to new value
                self.movement_stats['current_speed'] = (
                    alpha * raw_speed + (1 - alpha) * self.movement_stats['current_speed']
                )
                
            self.movement_stats['max_speed'] = max(self.movement_stats['max_speed'], raw_speed)
            
            # Calculate average speed with quality weighting
            if len(self.velocity_history) > 0:
                # Calculate weighted average based on tracking quality
                total_weighted_speed = 0
                total_weight = 0
                for entry in self.velocity_history:
                    # Handle both old and new formats
                    if len(entry) >= 4:
                        _, speed, _, quality = entry
                    elif len(entry) >= 3:
                        _, speed, _ = entry
                        quality = 0.5  # Default quality
                    else:
                        _, speed = entry
                        quality = 0.5  # Default quality
                    
                    weight = max(0.1, quality)  # Minimum weight to prevent zero division
                    total_weighted_speed += speed * weight
                    total_weight += weight
                
                if total_weight > 0:
                    self.movement_stats['average_speed'] = total_weighted_speed / total_weight

        # Apply quality-based filtering to position before adding to trail
        if len(self.position_trail) > 0:
            # Check for sudden position jumps that might indicate tracking errors
            last_pos = self.position_trail[-1]
            pos_diff = np.linalg.norm(position - last_pos)
            
            # Reasonable movement threshold (assuming ~1m/s max speed, 30fps = ~0.03m/frame)
            max_reasonable_move = 0.3 / (1.0 + tracking_quality * 2)  # Lower threshold for lower quality
            
            if pos_diff > max_reasonable_move and tracking_quality < 0.4:
                # Position jump is too large for low tracking quality, use prediction
                if len(self.velocity_history) >= 2:
                    # Use velocity prediction to fill in the gap
                    est_position = last_pos + self.velocity_history[-1][0] * dt
                    position = est_position
        
        # Store position in trail
        self.position_trail.append(position.copy())
        self.last_position = position.copy()
        self.last_update_time = current_time
        
        # Store orientation with quality filtering if available
        if 'pose' in slam_result:
            pose = slam_result['pose']
            # Extract orientation (simplified as yaw angle)
            rotation = pose[:3, :3]
            # Extract yaw angle (rotation around Z-axis)
            yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
            self.orientation_history.append((yaw, current_time, tracking_quality))

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
        """Draw the movement trajectory with enhanced accuracy indicators."""
        if len(self.position_trail) < 2:
            return
        
        # Convert positions to canvas coordinates with quality information if available
        points = []
        
        # Use SLAM results from elsewhere to get quality info, or use a fixed approach
        # For now, we'll assume positions are added in order and create a quality-based visualization
        for i, pos in enumerate(self.position_trail):
            x, y = self._world_to_canvas(pos)
            # Only add points that are valid integers within canvas bounds
            if isinstance(x, (int, np.integer)) and isinstance(y, (int, np.integer)):
                if 0 <= x < self.canvas_size[0] and 0 <= y < self.canvas_size[1]:
                    points.append((int(x), int(y), i))  # Include index for quality estimation
        
        # Draw trajectory line with quality indicators
        if len(points) >= 2:
            for i in range(len(points) - 1):
                pt1 = (points[i][0], points[i][1])
                pt2 = (points[i+1][0], points[i+1][1])
                
                # Ensure points are valid for OpenCV
                if (isinstance(pt1, tuple) and len(pt1) == 2 and 
                    isinstance(pt2, tuple) and len(pt2) == 2 and
                    all(isinstance(coord, (int, np.integer)) for coord in pt1 + pt2)):
                    
                    # Color based on position age (older to newer)
                    age_factor = i / len(points) if len(points) > 0 else 1.0
                    color = (int(200 * age_factor), 100, int(255 * (1 - age_factor)))  # From blue to red
                    
                    # Line thickness based on reliability - recent points are more reliable
                    thickness = 2 if i < len(points) * 0.7 else 3  # Thicker for more recent
                    
                    cv2.line(canvas, pt1, pt2, color, thickness)
        
        # Draw trajectory dots with size based on recency and quality
        for i, point in enumerate(points):
            x, y, pos_idx = point
            if (isinstance((x, y), tuple) and len((x, y)) == 2 and
                all(isinstance(coord, (int, np.integer)) for coord in (x, y)) and
                0 <= x < self.canvas_size[0] and 0 <= y < self.canvas_size[1]):
                
                age_factor = pos_idx / len(self.position_trail) if len(self.position_trail) > 1 else 1.0
                
                # Size based on recency - larger for more recent positions
                if pos_idx >= len(self.position_trail) * 0.8:  # Most recent 20%
                    size = 5
                elif pos_idx >= len(self.position_trail) * 0.5:  # Middle 30% 
                    size = 3
                else:  # Older positions
                    size = 2
                
                # Color based on recency (greener for more recent)
                color = (0, int(255 * age_factor), int(255 * (1 - age_factor)))  # Green for recent
                cv2.circle(canvas, (x, y), size, color, -1)
                
                # Add subtle trail fading for older points
                if pos_idx < len(self.position_trail) * 0.3:  # Oldest 30%
                    # Draw a smaller, lighter circle to show fading trail
                    faded_color = (int(100 * (1 - age_factor)), int(200 * (1 - age_factor)), int(200 * (1 - age_factor)))
                    cv2.circle(canvas, (x, y), size//2, faded_color, -1)

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
        
        # Get most recent velocity (with quality information)
        velocity_data = self.velocity_history[-1]
        if len(velocity_data) >= 3:
            velocity, speed, _, tracking_quality = velocity_data
        else:
            velocity, speed, _ = velocity_data
            tracking_quality = 0.5  # Default quality
        
        current_pos = self.position_trail[-1]
        
        start_x, start_y = self._world_to_canvas(current_pos)
        
        # Scale velocity vector for visualization
        scale_factor = 50  # pixels per unit of speed
        end_x = int(start_x + velocity[0] * scale_factor)
        end_y = int(start_y - velocity[1] * scale_factor)  # Flip Y axis
        
        if not (0 <= end_x < self.canvas_size[0] and 0 <= end_y < self.canvas_size[1]):
            return
        
        # Draw velocity vector with quality-based coloring
        if tracking_quality > 0.7:
            color = (0, 100, 255)  # Bright orange for high quality
        elif tracking_quality > 0.4:
            color = (0, 165, 255)  # Orange for medium quality
        else:
            color = (0, 0, 255)    # Red for low quality
        
        cv2.arrowedLine(canvas, (start_x, start_y), (end_x, end_y), color, 3, tipLength=0.2)
        
        # Draw speed information with quality indicator
        speed_text = f"Speed: {speed:.2f} m/s (Q: {tracking_quality:.1f})"
        cv2.putText(canvas, speed_text, (end_x + 10, end_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def _draw_orientation(self, canvas: np.ndarray):
        """Draw orientation indicator."""
        if len(self.orientation_history) < 1:
            return
        
        orientation_data = self.orientation_history[-1]
        if len(orientation_data) >= 3:
            orientation, _, tracking_quality = orientation_data
        else:
            orientation, _ = orientation_data
            tracking_quality = 0.5  # Default quality
        
        current_pos = self.position_trail[-1]
        pos_x, pos_y = self._world_to_canvas(current_pos)
        
        # Draw orientation arrow (length scaled for visibility)
        arrow_length = 30
        end_x = int(pos_x + arrow_length * np.cos(orientation))
        end_y = int(pos_y - arrow_length * np.sin(orientation))  # Flip Y axis
        
        # Draw orientation arrow with quality-based coloring
        if tracking_quality > 0.7:
            color = (0, 0, 200)    # Darker red for high quality
        elif tracking_quality > 0.4:
            color = (0, 0, 255)    # Red for medium quality
        else:
            color = (100, 100, 255) # Light red for low quality
            
        cv2.arrowedLine(canvas, (pos_x, pos_y), (end_x, end_y), color, 2, tipLength=0.3)
        
        # Draw direction label with quality indicator
        direction_text = f"Dir: {np.degrees(orientation):.1f}° (Q: {tracking_quality:.1f})"
        cv2.putText(canvas, direction_text, 
                   (end_x + 10, end_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _draw_statistics(self, canvas: np.ndarray):
        """Draw movement statistics with accuracy indicators."""
        stats_y = 50  # Starting Y position for stats
        line_height = 25
        
        # Draw background for stats
        overlay = canvas.copy()
        cv2.rectangle(overlay, (self.canvas_size[0] - 300, 40), 
                     (self.canvas_size[0] - 10, 230), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
        
        # Draw stats text with enhanced information
        cv2.putText(canvas, f"Distance: {self.movement_stats['total_distance']:.2f}m", 
                   (self.canvas_size[0] - 290, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        stats_y += line_height
        cv2.putText(canvas, f"Current Speed: {self.movement_stats['current_speed']:.2f}m/s", 
                   (self.canvas_size[0] - 290, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        stats_y += line_height
        cv2.putText(canvas, f"Average Speed: {self.movement_stats['average_speed']:.2f}m/s", 
                   (self.canvas_size[0] - 290, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        stats_y += line_height
        cv2.putText(canvas, f"Max Speed: {self.movement_stats['max_speed']:.2f}m/s", 
                   (self.canvas_size[0] - 290, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        stats_y += line_height
        cv2.putText(canvas, f"Trajectory: {len(self.position_trail)} pts", 
                   (self.canvas_size[0] - 290, stats_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        stats_y += line_height
        # Calculate smoothness indicator based on velocity variance
        if len(self.velocity_history) > 5:
            # Calculate velocity variance for smoothness estimation
            speeds = []
            for item in self.velocity_history:
                if len(item) >= 3:
                    speeds.append(item[1])  # speed value
            if speeds:
                speed_std = np.std(speeds[-10:]) if len(speeds) >= 10 else np.std(speeds)
                smoothness = 1.0 - min(1.0, speed_std * 2.0)  # Lower std = smoother
                cv2.putText(canvas, f"Smoothness: {smoothness:.2f}", 
                           (self.canvas_size[0] - 290, stats_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1)
        else:
            cv2.putText(canvas, f"Smoothness: --", 
                       (self.canvas_size[0] - 290, stats_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        stats_y += line_height
        # Show trajectory confidence based on position consistency
        if len(self.position_trail) > 2:
            # Calculate average distance between consecutive positions
            distances = []
            for i in range(1, min(len(self.position_trail), 10)):  # Check last 10 positions
                dist = np.linalg.norm(self.position_trail[-i] - self.position_trail[-i-1])
                distances.append(dist)
            
            if distances:
                avg_step = np.mean(distances)
                # Lower average step size indicates more consistent tracking
                consistency = max(0.0, min(1.0, 0.3 / (avg_step + 0.01))) 
                cv2.putText(canvas, f"Consistency: {consistency:.2f}", 
                           (self.canvas_size[0] - 290, stats_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)
        else:
            cv2.putText(canvas, f"Consistency: --", 
                       (self.canvas_size[0] - 290, stats_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

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

    def get_velocity_profile(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get velocity profile over time with quality information."""
        if len(self.velocity_history) == 0:
            return np.zeros(0), np.zeros(0), np.zeros(0)
        
        # Handle both old and new velocity history formats
        times = []
        speeds = []
        qualities = []
        
        for item in self.velocity_history:
            if len(item) >= 4:  # New format with quality
                _, speed, time_val, quality = item
            elif len(item) >= 3:  # Old format with time but no quality
                _, speed, time_val = item
                quality = 0.5  # Default quality
            else:  # Very old format
                _, speed = item
                time_val = time.time()
                quality = 0.5  # Default quality
            
            times.append(time_val)
            speeds.append(speed)
            qualities.append(quality)
        
        return np.array(times), np.array(speeds), np.array(qualities)

    def get_stats(self) -> Dict:
        """Get current movement statistics with accuracy metrics."""
        stats = self.movement_stats.copy()
        stats['position_count'] = len(self.position_trail)
        stats['velocity_samples'] = len(self.velocity_history)
        
        # Add accuracy-related metrics
        if len(self.velocity_history) > 1:
            # Calculate velocity consistency (lower variance = more consistent)
            speeds = []
            for item in self.velocity_history[-20:]:  # Use last 20 samples
                if len(item) >= 3:
                    speeds.append(item[1])  # speed value
            
            if speeds:
                speed_variance = np.var(speeds)
                stats['velocity_variance'] = speed_variance
                stats['speed_std'] = np.std(speeds)
                # Smoothness score: lower variance = higher smoothness
                stats['smoothness_score'] = 1.0 / (1.0 + speed_variance * 10)
        
        # Calculate trajectory consistency
        if len(self.position_trail) > 1:
            # Calculate average step size and its variance
            step_sizes = []
            for i in range(1, min(len(self.position_trail), 15)):  # Check last 15 positions
                dist = np.linalg.norm(self.position_trail[-i] - self.position_trail[-i-1])
                step_sizes.append(dist)
            
            if step_sizes:
                avg_step = np.mean(step_sizes)
                step_variance = np.var(step_sizes)
                stats['avg_step_size'] = avg_step
                stats['step_variance'] = step_variance
                stats['trajectory_consistency'] = 1.0 / (1.0 + step_variance * 100)
        
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