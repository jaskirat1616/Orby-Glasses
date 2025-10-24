"""
Smart Caching System for OrbyGlasses
Optimizes performance through intelligent caching and prediction.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging


class SmartCache:
    """
    Intelligent caching system that predicts what to compute and what to cache.
    Uses motion analysis to determine when full recomputation is needed.
    """

    def __init__(self, cache_size: int = 10):
        """
        Initialize smart cache.

        Args:
            cache_size: Number of frames to keep in cache
        """
        self.cache_size = cache_size

        # Frame history for motion analysis
        self.frame_history = deque(maxlen=cache_size)
        self.depth_cache = deque(maxlen=cache_size)
        self.detection_cache = deque(maxlen=cache_size)

        # Motion tracking
        self.last_motion_score = 0.0
        self.motion_threshold = 0.15  # Threshold for significant motion

        # Performance stats
        self.cache_hits = 0
        self.cache_misses = 0

        logging.info("Smart cache initialized")

    def compute_motion_score(self, current_frame: np.ndarray,
                            previous_frame: Optional[np.ndarray]) -> float:
        """
        Compute motion score between two frames.
        Higher score = more motion = need fresh computation.

        Args:
            current_frame: Current frame
            previous_frame: Previous frame

        Returns:
            Motion score (0-1, where 1 = maximum motion)
        """
        if previous_frame is None or current_frame.shape != previous_frame.shape:
            return 1.0  # Full motion on first frame or size change

        try:
            # Convert to grayscale for faster comparison
            if len(current_frame.shape) == 3:
                current_gray = np.mean(current_frame, axis=2).astype(np.uint8)
                prev_gray = np.mean(previous_frame, axis=2).astype(np.uint8)
            else:
                current_gray = current_frame
                prev_gray = previous_frame

            # Downsample for speed (4x4 blocks)
            current_small = current_gray[::4, ::4]
            prev_small = prev_gray[::4, ::4]

            # Compute absolute difference
            diff = np.abs(current_small.astype(np.float32) - prev_small.astype(np.float32))

            # Normalize to 0-1
            motion_score = np.mean(diff) / 255.0

            return float(motion_score)

        except Exception as e:
            logging.error(f"Motion score computation error: {e}")
            return 1.0  # Assume full motion on error

    def should_recompute_depth(self, motion_score: float, frame_count: int) -> bool:
        """
        Determine if depth map should be recomputed.

        Args:
            motion_score: Current motion score
            frame_count: Current frame number

        Returns:
            True if depth should be recomputed
        """
        # Always compute on first frame
        if frame_count == 0:
            return True

        # Recompute if significant motion detected
        if motion_score > self.motion_threshold:
            self.cache_misses += 1
            return True

        # Also recompute periodically even with low motion (every 15 frames)
        if frame_count % 15 == 0:
            return True

        # Cache hit - reuse previous depth
        self.cache_hits += 1
        return False

    def update(self, frame: np.ndarray, depth_map: Optional[np.ndarray],
               detections: List[Dict]):
        """
        Update cache with new data.

        Args:
            frame: Current frame
            depth_map: Current depth map
            detections: Current detections
        """
        self.frame_history.append(frame.copy())
        if depth_map is not None:
            self.depth_cache.append(depth_map.copy())
        if detections:
            self.detection_cache.append(detections.copy())

    def get_cached_depth(self) -> Optional[np.ndarray]:
        """Get most recent cached depth map."""
        if len(self.depth_cache) > 0:
            return self.depth_cache[-1]
        return None

    def get_previous_frame(self) -> Optional[np.ndarray]:
        """Get previous frame for motion analysis."""
        if len(self.frame_history) >= 2:
            return self.frame_history[-2]
        return None

    def predict_object_motion(self, current_detections: List[Dict]) -> List[Dict]:
        """
        Predict where objects will move based on history.

        Args:
            current_detections: Current frame detections

        Returns:
            Detections with predicted future positions
        """
        if len(self.detection_cache) < 2:
            return current_detections

        try:
            prev_detections = self.detection_cache[-2]

            # Match objects between frames by proximity
            for curr_det in current_detections:
                curr_center = curr_det.get('center', [0, 0])
                curr_label = curr_det.get('label', '')

                # Find matching object in previous frame
                best_match = None
                min_distance = float('inf')

                for prev_det in prev_detections:
                    if prev_det.get('label') != curr_label:
                        continue

                    prev_center = prev_det.get('center', [0, 0])
                    distance = np.sqrt((curr_center[0] - prev_center[0])**2 +
                                      (curr_center[1] - prev_center[1])**2)

                    if distance < min_distance and distance < 100:  # Max 100px movement
                        min_distance = distance
                        best_match = prev_det

                # Predict future position if match found
                if best_match:
                    prev_center = best_match.get('center', [0, 0])
                    velocity = [curr_center[0] - prev_center[0],
                               curr_center[1] - prev_center[1]]

                    # Predict 3 frames ahead (~0.15 seconds at 20 FPS)
                    predicted_center = [
                        curr_center[0] + velocity[0] * 3,
                        curr_center[1] + velocity[1] * 3
                    ]

                    curr_det['predicted_center'] = predicted_center
                    curr_det['velocity'] = velocity
                    curr_det['is_moving'] = np.linalg.norm(velocity) > 2.0

        except Exception as e:
            logging.error(f"Object motion prediction error: {e}")

        return current_detections

    def get_performance_stats(self) -> Dict[str, float]:
        """Get cache performance statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0.0

        return {
            'cache_hit_rate': hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': total
        }


class PredictiveEngine:
    """
    Predicts future state to enable proactive navigation guidance.
    """

    def __init__(self):
        """Initialize predictive engine."""
        self.collision_history = deque(maxlen=100)
        self.safe_path_history = deque(maxlen=100)

        logging.info("Predictive engine initialized")

    def predict_collision_risk(self, detections: List[Dict],
                               time_horizon: float = 0.5) -> List[Dict]:
        """
        Predict collision risk for moving objects.

        Args:
            detections: Current detections with velocities
            time_horizon: Time to look ahead (seconds, ~10 frames at 20 FPS)

        Returns:
            Detections with collision risk scores
        """
        for det in detections:
            if not det.get('is_moving', False):
                det['collision_risk'] = 0.0
                continue

            velocity = det.get('velocity', [0, 0])
            depth = det.get('depth', 10.0)
            center = det.get('center', [160, 160])

            # Predict future position
            speed = np.linalg.norm(velocity)

            # Risk factors:
            # 1. Moving toward center of frame (toward user)
            heading_to_center = (center[0] > 100 and velocity[0] < 0) or \
                               (center[0] < 220 and velocity[0] > 0)

            # 2. Close distance
            distance_risk = max(0, 1.0 - (depth / 3.0))

            # 3. High speed
            speed_risk = min(1.0, speed / 20.0)

            # Combined risk score
            collision_risk = 0.0
            if heading_to_center:
                collision_risk = (distance_risk * 0.5 + speed_risk * 0.5) * 0.8
            else:
                collision_risk = (distance_risk * 0.7 + speed_risk * 0.3) * 0.4

            det['collision_risk'] = float(collision_risk)

            # Log high-risk collisions
            if collision_risk > 0.6:
                self.collision_history.append({
                    'time': time.time(),
                    'label': det.get('label'),
                    'risk': collision_risk
                })

        return detections

    def suggest_safe_direction(self, detections: List[Dict],
                              frame_width: int = 416) -> str:
        """
        Suggest safest direction to move.

        Args:
            detections: Current detections
            frame_width: Frame width in pixels

        Returns:
            Direction suggestion ('left', 'right', 'forward', 'stop')
        """
        if not detections:
            return 'forward'

        # Analyze left, center, and right regions
        left_risk = 0.0
        center_risk = 0.0
        right_risk = 0.0

        left_bound = frame_width / 3
        right_bound = 2 * frame_width / 3

        for det in detections:
            center_x = det.get('center', [frame_width/2, 0])[0]
            depth = det.get('depth', 10.0)
            collision_risk = det.get('collision_risk', 0.0)

            # Weight by inverse distance (closer = higher risk)
            risk_weight = 1.0 / max(depth, 0.5)

            # Add to appropriate region
            if center_x < left_bound:
                left_risk += risk_weight * (1 + collision_risk)
            elif center_x > right_bound:
                right_risk += risk_weight * (1 + collision_risk)
            else:
                center_risk += risk_weight * (1 + collision_risk)

        # Determine safest direction
        if center_risk > 2.0:  # High danger straight ahead
            if left_risk < right_risk:
                return 'left'
            elif right_risk < left_risk:
                return 'right'
            else:
                return 'stop'  # Both sides equally risky
        elif center_risk > 1.0:  # Moderate danger
            if left_risk < right_risk and left_risk < 0.5:
                return 'slight_left'
            elif right_risk < left_risk and right_risk < 0.5:
                return 'slight_right'

        return 'forward'

    def analyze_pattern(self) -> Dict[str, any]:
        """
        Analyze patterns in collision history.

        Returns:
            Pattern analysis results
        """
        if len(self.collision_history) == 0:
            return {'frequent_risks': [], 'risk_rate': 0.0}

        # Count object types
        risk_counts = {}
        for event in self.collision_history:
            label = event['label']
            risk_counts[label] = risk_counts.get(label, 0) + 1

        # Sort by frequency
        frequent_risks = sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)

        return {
            'frequent_risks': frequent_risks[:5],
            'risk_rate': len(self.collision_history) / 100.0,
            'total_events': len(self.collision_history)
        }
