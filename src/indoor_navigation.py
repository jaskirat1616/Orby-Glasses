"""
OrbyGlasses - Indoor Navigation with Path Planning
Combines SLAM localization with A* path planning for goal-oriented navigation.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Set
from collections import deque
import heapq
from dataclasses import dataclass
import time


@dataclass
class NavigationGoal:
    """Represents a navigation goal/destination."""
    name: str
    position: np.ndarray  # [x, y, z] in world coordinates
    timestamp: float


class OccupancyGrid:
    """
    2D occupancy grid for path planning.
    Stores obstacles detected by SLAM and object detection.
    """

    def __init__(self, resolution: float = 0.1, size: Tuple[int, int] = (200, 200)):
        """
        Initialize occupancy grid.

        Args:
            resolution: Grid cell size in meters
            size: Grid dimensions (width, height)
        """
        self.resolution = resolution
        self.width, self.height = size
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)

        # Origin in world coordinates (center of grid)
        self.origin_x = -(self.width * self.resolution) / 2
        self.origin_y = -(self.height * self.resolution) / 2

        logging.info(f"Occupancy grid created: {self.width}x{self.height} @ {self.resolution}m/cell")

    def world_to_grid(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid indices."""
        if not np.all(np.isfinite(world_pos)):
            logging.warning(f"Received non-finite world position: {world_pos}. Returning default grid position.")
            return (self.width // 2, self.height // 2)  # Return center of the grid

        x = int((world_pos[0] - self.origin_x) / self.resolution)
        y = int((world_pos[1] - self.origin_y) / self.resolution)
        return (x, y)

    def grid_to_world(self, grid_pos: Tuple[int, int]) -> np.ndarray:
        """Convert grid indices to world coordinates."""
        x = grid_pos[0] * self.resolution + self.origin_x
        y = grid_pos[1] * self.resolution + self.origin_y
        return np.array([x, y, 0])

    def is_valid(self, grid_pos: Tuple[int, int]) -> bool:
        """Check if grid position is valid (within bounds)."""
        x, y = grid_pos
        return 0 <= x < self.width and 0 <= y < self.height

    def is_occupied(self, grid_pos: Tuple[int, int]) -> bool:
        """Check if grid cell is occupied."""
        if not self.is_valid(grid_pos):
            return True  # Out of bounds = occupied
        x, y = grid_pos
        return self.grid[y, x] > 127  # Threshold

    def set_occupied(self, grid_pos: Tuple[int, int], occupied: bool = True):
        """Mark grid cell as occupied or free."""
        if self.is_valid(grid_pos):
            x, y = grid_pos
            self.grid[y, x] = 255 if occupied else 0

    def update_from_detections(self, detections: List[Dict], camera_pos: np.ndarray):
        """
        Update occupancy grid from object detections.

        Args:
            detections: List of detected objects with positions
            camera_pos: Current camera position [x, y, z]
        """
        if not np.all(np.isfinite(camera_pos)):
            logging.warning(f"Invalid camera position: {camera_pos}. Skipping occupancy grid update.")
            return

        # Mark obstacles in grid
        for det in detections:
            depth = det.get('depth', 0)
            if depth == 0 or depth > 5.0 or not np.isfinite(depth):  # Skip invalid or far objects
                continue

            # Estimate object position (simplified - assumes object is in front of camera)
            obj_x = camera_pos[0] + depth  # Simplified: assumes facing +X
            obj_y = camera_pos[1]
            obj_pos = np.array([obj_x, obj_y, 0])

            if not np.all(np.isfinite(obj_pos)):
                continue

            # Mark as occupied
            grid_pos = self.world_to_grid(obj_pos)
            self.set_occupied(grid_pos, occupied=True)

            # Inflate obstacle (safety margin)
            inflation_radius = 2  # cells
            for dx in range(-inflation_radius, inflation_radius + 1):
                for dy in range(-inflation_radius, inflation_radius + 1):
                    neighbor = (grid_pos[0] + dx, grid_pos[1] + dy)
                    if self.is_valid(neighbor):
                        self.set_occupied(neighbor, occupied=True)

    def clear_around_position(self, world_pos: np.ndarray, radius: float = 0.5):
        """Clear grid cells around a position (assumed free space)."""
        if not np.all(np.isfinite(world_pos)):
            logging.warning(f"Invalid world position for clearing: {world_pos}. Skipping.")
            return

        grid_pos = self.world_to_grid(world_pos)
        cell_radius = int(radius / self.resolution)

        for dx in range(-cell_radius, cell_radius + 1):
            for dy in range(-cell_radius, cell_radius + 1):
                if dx * dx + dy * dy <= cell_radius * cell_radius:
                    neighbor = (grid_pos[0] + dx, grid_pos[1] + dy)
                    self.set_occupied(neighbor, occupied=False)

    def get_grid_image(self) -> np.ndarray:
        """Get occupancy grid as image (for visualization)."""
        return self.grid.copy()


class AStarPlanner:
    """A* path planner for grid-based navigation."""

    def __init__(self, occupancy_grid: OccupancyGrid):
        """
        Initialize A* planner.

        Args:
            occupancy_grid: OccupancyGrid instance
        """
        self.grid = occupancy_grid

    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Euclidean distance heuristic."""
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbors (8-connected)."""
        x, y = pos
        neighbors = []

        # 8 directions
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            neighbor = (x + dx, y + dy)
            if self.grid.is_valid(neighbor) and not self.grid.is_occupied(neighbor):
                neighbors.append(neighbor)

        return neighbors

    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* path planning.

        Args:
            start: Start grid position (x, y)
            goal: Goal grid position (x, y)

        Returns:
            List of grid positions forming path, or None if no path found
        """
        if not self.grid.is_valid(start) or not self.grid.is_valid(goal):
            logging.warning("Start or goal position invalid")
            return None

        if self.grid.is_occupied(start) or self.grid.is_occupied(goal):
            logging.warning("Start or goal position occupied")
            return None

        if start == goal:
            return [start]

        # Priority queue: (f_score, counter, position)
        open_set = [(0, 0, start)]
        came_from = {}
        g_score = {start: 0}
        counter = 0

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current):
                # Cost: 1 for straight, sqrt(2) for diagonal
                dx = abs(neighbor[0] - current[0])
                dy = abs(neighbor[1] - current[1])
                move_cost = 1.414 if (dx + dy) == 2 else 1.0

                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)

                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        logging.warning("No path found")
        return None


class IndoorNavigator:
    """
    High-level indoor navigation system.
    Combines SLAM localization with path planning.
    """

    def __init__(self, slam_system, config):
        """
        Initialize indoor navigator.

        Args:
            slam_system: MonocularSLAM instance
            config: ConfigManager instance
        """
        self.slam = slam_system
        self.config = config

        # Occupancy grid
        grid_size = config.get('slam.grid_size', (200, 200))
        grid_resolution = config.get('slam.grid_resolution', 0.1)
        self.occupancy_grid = OccupancyGrid(resolution=grid_resolution, size=grid_size)

        # Path planner
        self.planner = AStarPlanner(self.occupancy_grid)

        # Navigation state
        self.current_goal = None
        self.current_path = None
        self.path_index = 0

        # Saved locations
        self.saved_locations = {}  # name -> position

        logging.info("Indoor navigator initialized")

    def update(self, slam_result: Dict, detections: List[Dict]):
        """
        Update navigation system with new data.

        Args:
            slam_result: Result from SLAM.process_frame()
            detections: Object detections with depth
        """
        # Get current position
        position = np.array(slam_result['position'])

        if not np.all(np.isfinite(position)):
            logging.warning(f"Received non-finite position from SLAM: {position}. Skipping navigation update.")
            return

        # Update occupancy grid
        self.occupancy_grid.clear_around_position(position, radius=0.5)
        self.occupancy_grid.update_from_detections(detections, position)

        # Check if we need to replan (obstacles in path)
        if self.current_path:
            self._check_and_replan(position)

    def set_goal(self, goal_name: str, goal_position: np.ndarray = None):
        """
        Set navigation goal.

        Args:
            goal_name: Name of goal location
            goal_position: Target position [x, y, z] (if None, looks up saved location)
        """
        if goal_position is None:
            if goal_name not in self.saved_locations:
                logging.error(f"Unknown location: {goal_name}")
                return False
            goal_position = self.saved_locations[goal_name]

        self.current_goal = NavigationGoal(
            name=goal_name,
            position=goal_position,
            timestamp=time.time()
        )

        # Plan path
        success = self._plan_path()
        if success:
            logging.info(f"Goal set: {goal_name} at {goal_position}")
        else:
            logging.error(f"Failed to plan path to {goal_name}")

        return success

    def _plan_path(self) -> bool:
        """Plan path to current goal."""
        if self.current_goal is None:
            return False

        # Get current position from SLAM
        current_pos = self.slam.get_position()

        if not np.all(np.isfinite(current_pos)):
            logging.error(f"Cannot plan path with non-finite start position: {current_pos}")
            return False

        # Convert to grid coordinates
        start_grid = self.occupancy_grid.world_to_grid(current_pos)
        goal_grid = self.occupancy_grid.world_to_grid(self.current_goal.position)

        # Plan path
        path = self.planner.plan(start_grid, goal_grid)

        if path:
            self.current_path = path
            self.path_index = 0
            logging.info(f"Path planned: {len(path)} waypoints")
            return True
        else:
            self.current_path = None
            return False

    def _check_and_replan(self, current_pos: np.ndarray):
        """Check if current path is blocked and replan if needed."""
        if not self.current_path or self.path_index >= len(self.current_path):
            return

        if not np.all(np.isfinite(current_pos)):
            logging.warning("Cannot check for replan with non-finite position.")
            return

        # Check next few waypoints for obstacles
        lookahead = min(5, len(self.current_path) - self.path_index)
        for i in range(self.path_index, self.path_index + lookahead):
            if self.occupancy_grid.is_occupied(self.current_path[i]):
                logging.warning("Obstacle detected in path, replanning...")
                self._plan_path()
                break

    def get_navigation_guidance(self) -> Optional[str]:
        """
        Get navigation guidance for current goal.

        Returns:
            Guidance string or None
        """
        if self.current_goal is None or self.current_path is None:
            return None

        # Get current position
        current_pos = self.slam.get_position()
        if not np.all(np.isfinite(current_pos)):
            logging.warning("Cannot get navigation guidance with non-finite position.")
            return "Current position is unknown."

        current_grid = self.occupancy_grid.world_to_grid(current_pos)

        # Check if we reached goal
        goal_grid = self.occupancy_grid.world_to_grid(self.current_goal.position)
        distance_to_goal = self.occupancy_grid.resolution * np.linalg.norm(
            np.array(current_grid) - np.array(goal_grid)
        )

        if distance_to_goal < 0.5:  # Within 0.5m
            guidance = f"Arrived at {self.current_goal.name}"
            self.current_goal = None
            self.current_path = None
            return guidance

        # Get next waypoint
        if self.path_index >= len(self.current_path):
            return "Path completed"

        next_waypoint = self.current_path[min(self.path_index + 3, len(self.current_path) - 1)]
        next_world = self.occupancy_grid.grid_to_world(next_waypoint)

        # Calculate direction
        direction_vector = next_world[:2] - current_pos[:2]
        distance = np.linalg.norm(direction_vector)
        angle = np.arctan2(direction_vector[1], direction_vector[0])

        # Convert to guidance
        if abs(angle) < np.pi / 8:
            direction = "straight"
        elif abs(angle) < 3 * np.pi / 8:
            direction = "slight right" if angle < 0 else "slight left"
        elif abs(angle) < 5 * np.pi / 8:
            direction = "right" if angle < 0 else "left"
        else:
            direction = "sharp right" if angle < 0 else "sharp left"

        guidance = f"Continue {direction} for {distance:.1f} meters toward {self.current_goal.name}"

        return guidance

    def save_location(self, name: str, position: np.ndarray = None):
        """
        Save current location with a name.

        Args:
            name: Location name
            position: Position to save (if None, uses current SLAM position)
        """
        if position is None:
            position = self.slam.get_position()

        if not np.all(np.isfinite(position)):
            logging.error(f"Attempted to save non-finite position for location '{name}'. Location not saved.")
            return

        self.saved_locations[name] = position.copy()
        logging.info(f"Location saved: {name} at {position}")

    def get_saved_locations(self) -> Dict[str, np.ndarray]:
        """Get all saved locations."""
        return self.saved_locations.copy()

    def clear_goal(self):
        """Clear current navigation goal."""
        self.current_goal = None
        self.current_path = None
        self.path_index = 0
        logging.info("Navigation goal cleared")

    def visualize_grid(self) -> np.ndarray:
        """Get occupancy grid visualization."""
        grid_img = self.occupancy_grid.get_grid_image()

        # Convert to BGR for visualization
        grid_bgr = cv2.cvtColor(grid_img, cv2.COLOR_GRAY2BGR)

        # Draw path if exists
        if self.current_path:
            for i, (x, y) in enumerate(self.current_path):
                if i == self.path_index:
                    color = (0, 255, 255)  # Yellow for current waypoint
                else:
                    color = (255, 0, 0)  # Blue for path
                cv2.circle(grid_bgr, (x, y), 2, color, -1)

        # Draw current position
        current_pos = self.slam.get_position()
        if np.all(np.isfinite(current_pos)):
            current_grid = self.occupancy_grid.world_to_grid(current_pos)
            cv2.circle(grid_bgr, current_grid, 5, (0, 255, 0), -1)  # Green

        # Draw goal if exists
        if self.current_goal and np.all(np.isfinite(self.current_goal.position)):
            goal_grid = self.occupancy_grid.world_to_grid(self.current_goal.position)
            cv2.circle(grid_bgr, goal_grid, 5, (0, 0, 255), -1)  # Red

        return grid_bgr
