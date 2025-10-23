"""
Intelligent Object Tracker
Tracks objects across frames for better accuracy and context
"""
import numpy as np
from typing import List, Dict, Optional
from collections import deque
import time


class TrackedObject:
    """Represents an object tracked across multiple frames."""

    def __init__(self, obj_id: int, detection: Dict):
        """Initialize tracked object."""
        self.id = obj_id
        self.label = detection['label']
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.frames_tracked = 1
        self.frames_lost = 0

        # History
        self.position_history = deque(maxlen=10)
        self.depth_history = deque(maxlen=10)
        self.confidence_history = deque(maxlen=10)

        # Current state
        self.bbox = detection['bbox']
        self.center = detection['center']
        self.depth = detection.get('depth', 0)
        self.confidence = detection.get('confidence', 0)

        # Add to history
        self.position_history.append(self.center)
        self.depth_history.append(self.depth)
        self.confidence_history.append(self.confidence)

        # Movement
        self.velocity = [0, 0]  # pixels per frame
        self.depth_velocity = 0  # meters per frame

        # Importance
        self.is_priority = detection.get('is_priority', False)
        self.danger_level = 0  # 0-3: safe, caution, danger, immediate

    def update(self, detection: Dict):
        """Update with new detection."""
        self.last_seen = time.time()
        self.frames_tracked += 1
        self.frames_lost = 0

        # Update state
        old_center = self.center
        old_depth = self.depth

        self.bbox = detection['bbox']
        self.center = detection['center']
        self.depth = detection.get('depth', self.depth)
        self.confidence = detection.get('confidence', self.confidence)

        # Update history
        self.position_history.append(self.center)
        self.depth_history.append(self.depth)
        self.confidence_history.append(self.confidence)

        # Calculate velocity
        self.velocity = [
            self.center[0] - old_center[0],
            self.center[1] - old_center[1]
        ]
        self.depth_velocity = self.depth - old_depth

    def mark_lost(self):
        """Mark as lost this frame."""
        self.frames_lost += 1

    def get_smoothed_depth(self) -> float:
        """Get smoothed depth using median of history."""
        if not self.depth_history:
            return 0.0
        return float(np.median(list(self.depth_history)))

    def get_smoothed_position(self) -> List[float]:
        """Get smoothed position using average."""
        if not self.position_history:
            return self.center
        positions = np.array(list(self.position_history))
        return positions.mean(axis=0).tolist()

    def is_approaching(self) -> bool:
        """Check if object is approaching (getting closer)."""
        return self.depth_velocity < -0.05  # Moving closer

    def is_moving(self) -> bool:
        """Check if object is moving laterally."""
        speed = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        return speed > 5  # pixels per frame

    def time_since_seen(self) -> float:
        """Time since last seen."""
        return time.time() - self.last_seen

    def should_remove(self, max_lost_frames: int = 5) -> bool:
        """Check if object should be removed from tracking."""
        return self.frames_lost > max_lost_frames


class ObjectTracker:
    """Tracks objects across frames for better accuracy."""

    def __init__(self, max_distance: float = 50.0, max_depth_diff: float = 1.0):
        """
        Initialize object tracker.

        Args:
            max_distance: Max pixel distance to match objects
            max_depth_diff: Max depth difference to match objects
        """
        self.max_distance = max_distance
        self.max_depth_diff = max_depth_diff
        self.next_id = 0
        self.tracked_objects: Dict[int, TrackedObject] = {}

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracking with new detections.

        Args:
            detections: New detections from current frame

        Returns:
            Enhanced detections with tracking info
        """
        # Mark all as lost initially
        for obj in self.tracked_objects.values():
            obj.mark_lost()

        # Match detections to tracked objects
        matched = set()

        for detection in detections:
            best_match = None
            best_score = float('inf')

            for obj_id, tracked in self.tracked_objects.items():
                if obj_id in matched:
                    continue

                # Only match same label
                if tracked.label != detection['label']:
                    continue

                # Calculate distance
                center = detection['center']
                dist = np.sqrt(
                    (center[0] - tracked.center[0])**2 +
                    (center[1] - tracked.center[1])**2
                )

                # Check depth similarity
                depth_diff = abs(detection.get('depth', 0) - tracked.depth)

                # Combined score (lower is better)
                score = dist + (depth_diff * 20)  # Weight depth more

                if dist < self.max_distance and depth_diff < self.max_depth_diff:
                    if score < best_score:
                        best_score = score
                        best_match = obj_id

            # Update or create
            if best_match is not None:
                self.tracked_objects[best_match].update(detection)
                detection['track_id'] = best_match
                detection['frames_tracked'] = self.tracked_objects[best_match].frames_tracked
                detection['smoothed_depth'] = self.tracked_objects[best_match].get_smoothed_depth()
                detection['is_approaching'] = self.tracked_objects[best_match].is_approaching()
                detection['is_moving'] = self.tracked_objects[best_match].is_moving()
                matched.add(best_match)
            else:
                # New object
                new_id = self.next_id
                self.next_id += 1
                self.tracked_objects[new_id] = TrackedObject(new_id, detection)
                detection['track_id'] = new_id
                detection['frames_tracked'] = 1
                detection['smoothed_depth'] = detection.get('depth', 0)
                detection['is_approaching'] = False
                detection['is_moving'] = False

        # Remove old objects
        to_remove = [
            obj_id for obj_id, obj in self.tracked_objects.items()
            if obj.should_remove()
        ]
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]

        return detections

    def get_tracked_count(self) -> int:
        """Get number of currently tracked objects."""
        return len(self.tracked_objects)

    def get_priority_objects(self) -> List[TrackedObject]:
        """Get priority objects (close, approaching, or dangerous)."""
        priority = []
        for obj in self.tracked_objects.values():
            if obj.frames_lost > 0:
                continue
            if obj.depth < 2.0 or obj.is_approaching() or obj.is_priority:
                priority.append(obj)

        # Sort by depth (closest first)
        priority.sort(key=lambda x: x.get_smoothed_depth())
        return priority
