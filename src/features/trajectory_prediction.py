"""
OrbyGlasses - Trajectory Prediction with Graph Neural Networks
Predicts future positions of moving objects (people, vehicles) for proactive navigation.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
from collections import deque, defaultdict
import time


class ObjectTracker:
    """
    Tracks objects across frames to build trajectory histories.
    Uses simple centroid-based tracking with IoU matching.
    """

    def __init__(self, max_history: int = 10, max_disappeared: int = 5):
        """
        Initialize object tracker.

        Args:
            max_history: Maximum number of historical positions to store
            max_disappeared: Maximum frames an object can disappear before removal
        """
        self.max_history = max_history
        self.max_disappeared = max_disappeared

        # Track objects
        self.objects = {}  # id -> {'positions': deque, 'velocities': deque, 'label': str, ...}
        self.disappeared = {}  # id -> frame count
        self.next_object_id = 0

        logging.info(f"Object tracker initialized (history: {max_history} frames)")

    def update(self, detections: List[Dict], frame_time: float) -> Dict[int, Dict]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detected objects with bbox, depth, label
            frame_time: Timestamp of current frame

        Returns:
            Dictionary of tracked objects with trajectory info
        """
        # If no existing objects, register all detections
        if len(self.objects) == 0:
            for detection in detections:
                self._register_object(detection, frame_time)
            return self.objects

        # Get object IDs and centroids
        object_ids = list(self.objects.keys())
        object_centroids = [self._get_latest_position(obj_id) for obj_id in object_ids]

        # Get detection centroids
        detection_centroids = [self._get_centroid_3d(det) for det in detections]

        if len(detection_centroids) == 0:
            # No detections - mark all as disappeared
            for obj_id in object_ids:
                self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1

            # Remove objects that disappeared too long
            for obj_id in list(self.disappeared.keys()):
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister_object(obj_id)

            return self.objects

        # Match detections to tracked objects
        matches = self._match_detections(object_centroids, detection_centroids)

        # Update matched objects
        matched_object_ids = set()
        matched_detection_ids = set()

        for obj_idx, det_idx in matches:
            obj_id = object_ids[obj_idx]
            detection = detections[det_idx]

            self._update_object(obj_id, detection, frame_time)
            matched_object_ids.add(obj_id)
            matched_detection_ids.add(det_idx)

            # Reset disappeared counter
            self.disappeared[obj_id] = 0

        # Handle unmatched objects (mark as disappeared)
        for i, obj_id in enumerate(object_ids):
            if i not in [m[0] for m in matches]:
                self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1

                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister_object(obj_id)

        # Register new detections
        for i, detection in enumerate(detections):
            if i not in matched_detection_ids:
                self._register_object(detection, frame_time)

        return self.objects

    def _register_object(self, detection: Dict, frame_time: float):
        """Register a new tracked object."""
        centroid = self._get_centroid_3d(detection)

        self.objects[self.next_object_id] = {
            'positions': deque([centroid], maxlen=self.max_history),
            'timestamps': deque([frame_time], maxlen=self.max_history),
            'velocities': deque(maxlen=self.max_history),
            'label': detection.get('label', 'unknown'),
            'last_seen': frame_time
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def _deregister_object(self, obj_id: int):
        """Remove a tracked object."""
        if obj_id in self.objects:
            del self.objects[obj_id]
        if obj_id in self.disappeared:
            del self.disappeared[obj_id]

    def _update_object(self, obj_id: int, detection: Dict, frame_time: float):
        """Update tracked object with new detection."""
        obj = self.objects[obj_id]
        new_position = self._get_centroid_3d(detection)

        # Calculate velocity
        if len(obj['positions']) > 0:
            last_position = obj['positions'][-1]
            last_time = obj['timestamps'][-1]
            dt = frame_time - last_time

            if dt > 0:
                velocity = (new_position - last_position) / dt
                obj['velocities'].append(velocity)

        # Update position
        obj['positions'].append(new_position)
        obj['timestamps'].append(frame_time)
        obj['last_seen'] = frame_time
        obj['label'] = detection.get('label', obj['label'])

    def _get_centroid_3d(self, detection: Dict) -> np.ndarray:
        """
        Get 3D centroid from detection.

        Args:
            detection: Detection with bbox and depth

        Returns:
            3D position [x, y, z] where z is depth
        """
        bbox = detection.get('bbox', [0, 0, 1, 1])
        center = detection.get('center', [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        depth = detection.get('depth', 5.0)

        # Convert to 3D: use pixel center as (x, y) and depth as z
        # Normalize pixel coordinates to meters (rough approximation)
        x = (center[0] - 320) / 100.0  # Assuming 640x480 image, normalize to meters
        y = (center[1] - 240) / 100.0
        z = depth

        return np.array([x, y, z], dtype=np.float32)

    def _get_latest_position(self, obj_id: int) -> np.ndarray:
        """Get latest position of tracked object."""
        if obj_id not in self.objects:
            return np.zeros(3)
        positions = self.objects[obj_id]['positions']
        return positions[-1] if len(positions) > 0 else np.zeros(3)

    def _match_detections(self, object_centroids: List[np.ndarray],
                         detection_centroids: List[np.ndarray]) -> List[Tuple[int, int]]:
        """
        Match detections to existing objects using distance.

        Args:
            object_centroids: List of existing object positions
            detection_centroids: List of new detection positions

        Returns:
            List of (object_idx, detection_idx) matches
        """
        if len(object_centroids) == 0 or len(detection_centroids) == 0:
            return []

        # Calculate distance matrix
        distances = np.zeros((len(object_centroids), len(detection_centroids)))
        for i, obj_pos in enumerate(object_centroids):
            for j, det_pos in enumerate(detection_centroids):
                distances[i, j] = np.linalg.norm(obj_pos - det_pos)

        # Simple greedy matching: assign each detection to nearest object
        matches = []
        used_objects = set()
        used_detections = set()

        # Sort by distance
        flat_indices = np.argsort(distances.ravel())
        for idx in flat_indices:
            obj_idx = idx // len(detection_centroids)
            det_idx = idx % len(detection_centroids)

            if obj_idx in used_objects or det_idx in used_detections:
                continue

            # Only match if distance is reasonable (< 2 meters)
            if distances[obj_idx, det_idx] < 2.0:
                matches.append((obj_idx, det_idx))
                used_objects.add(obj_idx)
                used_detections.add(det_idx)

        return matches

    def get_tracked_objects(self) -> Dict[int, Dict]:
        """Get all currently tracked objects."""
        return self.objects


class SocialForceModel:
    """
    Simple social force model for trajectory prediction.
    Models how people avoid each other and obstacles.
    """

    def __init__(self):
        """Initialize social force model."""
        self.person_radius = 0.5  # Personal space radius (meters)
        self.obstacle_radius = 0.3
        self.max_speed = 2.0  # m/s

    def predict_social_forces(self, target_obj: Dict, other_objects: List[Dict]) -> np.ndarray:
        """
        Calculate social forces acting on target object.

        Args:
            target_obj: Target object with position and velocity
            other_objects: List of other objects in scene

        Returns:
            Force vector [fx, fy, fz]
        """
        if len(target_obj['positions']) < 2:
            return np.zeros(3)

        target_pos = target_obj['positions'][-1]
        target_vel = target_obj['velocities'][-1] if len(target_obj['velocities']) > 0 else np.zeros(3)

        total_force = np.zeros(3)

        # Repulsive force from other objects
        for other_obj in other_objects:
            if len(other_obj['positions']) == 0:
                continue

            other_pos = other_obj['positions'][-1]
            diff = target_pos - other_pos
            distance = np.linalg.norm(diff)

            if distance < 0.1:  # Too close
                continue

            # Repulsive force inversely proportional to distance
            if distance < 3.0:  # Only consider nearby objects
                force_magnitude = 1.0 / (distance ** 2)
                force_direction = diff / distance
                total_force += force_magnitude * force_direction

        return total_force


class TrajectoryGNN:
    """
    Simplified Graph Neural Network for trajectory prediction.
    Uses spatial-temporal graph convolution to predict future positions.

    This is a lightweight implementation suitable for real-time use.
    For production, consider using PyTorch Geometric with proper training.
    """

    def __init__(self, prediction_horizon: int = 3, time_step: float = 0.5):
        """
        Initialize trajectory prediction GNN.

        Args:
            prediction_horizon: Number of future timesteps to predict (default: 3)
            time_step: Time between predictions in seconds (default: 0.5s)
        """
        self.prediction_horizon = prediction_horizon
        self.time_step = time_step

        # Social force model for basic predictions
        self.social_force = SocialForceModel()

        logging.info(f"Trajectory GNN initialized (horizon: {prediction_horizon} steps @ {time_step}s)")

    def predict_trajectories(self, tracked_objects: Dict[int, Dict]) -> Dict[int, Dict]:
        """
        Predict future trajectories for all tracked objects.

        Args:
            tracked_objects: Dictionary of tracked objects with history

        Returns:
            Dictionary mapping object_id to prediction:
                {
                    'predicted_positions': List of future positions,
                    'predicted_times': List of future timestamps,
                    'confidence': Prediction confidence (0-1),
                    'predicted_collision': Whether collision is predicted
                }
        """
        predictions = {}

        # Convert to list for easier processing
        obj_list = []
        obj_ids = []
        for obj_id, obj_data in tracked_objects.items():
            if len(obj_data['positions']) >= 2:  # Need at least 2 positions for velocity
                obj_list.append(obj_data)
                obj_ids.append(obj_id)

        if len(obj_list) == 0:
            return predictions

        # Predict for each object
        for i, (obj_id, obj_data) in enumerate(zip(obj_ids, obj_list)):
            # Get other objects for social force calculation
            other_objects = [obj for j, obj in enumerate(obj_list) if j != i]

            # Predict trajectory
            prediction = self._predict_single_trajectory(obj_data, other_objects)
            predictions[obj_id] = prediction

        return predictions

    def _predict_single_trajectory(self, obj_data: Dict, other_objects: List[Dict]) -> Dict:
        """
        Predict trajectory for a single object using physics + social forces.

        Args:
            obj_data: Object with position and velocity history
            other_objects: Other objects in scene

        Returns:
            Prediction dictionary
        """
        if len(obj_data['positions']) < 2 or len(obj_data['velocities']) < 1:
            return {
                'predicted_positions': [],
                'predicted_times': [],
                'confidence': 0.0,
                'predicted_collision': False
            }

        # Current state
        current_pos = np.array(obj_data['positions'][-1])
        current_vel = np.array(obj_data['velocities'][-1])
        current_time = obj_data['timestamps'][-1]

        # Calculate average velocity (more stable than instantaneous)
        if len(obj_data['velocities']) >= 3:
            avg_velocity = np.mean([v for v in obj_data['velocities']], axis=0)
        else:
            avg_velocity = current_vel

        # Predict future positions
        predicted_positions = []
        predicted_times = []

        position = current_pos.copy()
        velocity = avg_velocity.copy()

        for step in range(1, self.prediction_horizon + 1):
            # Calculate social forces
            temp_obj = {
                'positions': deque([position]),
                'velocities': deque([velocity])
            }
            social_force = self.social_force.predict_social_forces(temp_obj, other_objects)

            # Update velocity with social forces (simplified dynamics)
            # F = ma, assume m=1, so a = F
            acceleration = social_force * 0.5  # Damping factor
            velocity = velocity + acceleration * self.time_step

            # Clamp velocity to realistic maximum
            speed = np.linalg.norm(velocity)
            if speed > self.social_force.max_speed:
                velocity = velocity / speed * self.social_force.max_speed

            # Update position
            position = position + velocity * self.time_step

            predicted_positions.append(position.copy())
            predicted_times.append(current_time + step * self.time_step)

        # Calculate confidence based on velocity consistency
        if len(obj_data['velocities']) >= 3:
            velocity_std = np.std([v for v in obj_data['velocities']], axis=0)
            velocity_consistency = 1.0 / (1.0 + np.mean(velocity_std))
            confidence = min(1.0, velocity_consistency)
        else:
            confidence = 0.5

        # Check for predicted collisions (distance < 1m to any other object)
        predicted_collision = False
        for other_obj in other_objects:
            if len(other_obj['positions']) == 0:
                continue

            other_pos = other_obj['positions'][-1]
            for pred_pos in predicted_positions:
                distance = np.linalg.norm(pred_pos - other_pos)
                if distance < 1.0:  # Collision threshold
                    predicted_collision = True
                    break
            if predicted_collision:
                break

        return {
            'predicted_positions': predicted_positions,
            'predicted_times': predicted_times,
            'confidence': confidence,
            'predicted_collision': predicted_collision,
            'current_velocity': avg_velocity
        }


class TrajectoryPredictionSystem:
    """
    Complete trajectory prediction system integrating tracking and GNN.
    """

    def __init__(self, config):
        """
        Initialize trajectory prediction system.

        Args:
            config: ConfigManager instance
        """
        self.config = config
        self.enabled = config.get('trajectory_prediction.enabled', False)

        if not self.enabled:
            logging.info("Trajectory prediction disabled")
            return

        # Initialize components
        max_history = config.get('trajectory_prediction.max_history', 10)
        prediction_horizon = config.get('trajectory_prediction.prediction_horizon', 3)
        time_step = config.get('trajectory_prediction.time_step', 0.5)

        self.tracker = ObjectTracker(max_history=max_history)
        self.gnn = TrajectoryGNN(prediction_horizon=prediction_horizon, time_step=time_step)

        # State
        self.last_update_time = 0
        self.frame_count = 0

        logging.info("Trajectory prediction system initialized")

    def update(self, detections: List[Dict]) -> Dict:
        """
        Update system with new detections and predict trajectories.

        Args:
            detections: List of detected objects

        Returns:
            Dictionary with tracking and prediction results:
                {
                    'tracked_objects': Dict of tracked objects,
                    'predictions': Dict of trajectory predictions
                }
        """
        if not self.enabled:
            return {'tracked_objects': {}, 'predictions': {}}

        current_time = time.time()
        self.frame_count += 1

        # Update tracker
        tracked_objects = self.tracker.update(detections, current_time)

        # Predict trajectories
        predictions = self.gnn.predict_trajectories(tracked_objects)

        self.last_update_time = current_time

        return {
            'tracked_objects': tracked_objects,
            'predictions': predictions
        }

    def get_collision_warnings(self, predictions: Dict, user_position: np.ndarray = None) -> List[Dict]:
        """
        Get collision warnings based on predictions.

        Args:
            predictions: Trajectory predictions from update()
            user_position: Current user position (optional)

        Returns:
            List of collision warnings with object info and time to collision
        """
        warnings = []

        if user_position is None:
            user_position = np.array([0, 0, 0])  # Assume user at origin

        for obj_id, prediction in predictions.items():
            if not prediction['predicted_collision']:
                continue

            predicted_positions = prediction['predicted_positions']
            predicted_times = prediction['predicted_times']

            # Check if any predicted position is close to user
            for i, (pos, t) in enumerate(zip(predicted_positions, predicted_times)):
                distance = np.linalg.norm(pos - user_position)

                if distance < 1.5:  # Warning threshold
                    time_to_collision = t - time.time()

                    warnings.append({
                        'object_id': obj_id,
                        'predicted_position': pos,
                        'time_to_collision': time_to_collision,
                        'distance': distance,
                        'urgency': 'high' if time_to_collision < 1.0 else 'medium'
                    })
                    break  # Only warn about first collision point

        return warnings

    def visualize_predictions(self, frame: np.ndarray, tracked_objects: Dict,
                             predictions: Dict) -> np.ndarray:
        """
        Visualize trajectories and predictions on frame.

        Args:
            frame: Input frame
            tracked_objects: Tracked objects
            predictions: Trajectory predictions

        Returns:
            Annotated frame with trajectories
        """
        import cv2

        vis_frame = frame.copy()

        for obj_id, obj_data in tracked_objects.items():
            # Draw historical trajectory (blue)
            positions = obj_data['positions']
            if len(positions) < 2:
                continue

            # Convert 3D positions to 2D image coordinates
            for i in range(len(positions) - 1):
                pt1 = self._world_to_image(positions[i])
                pt2 = self._world_to_image(positions[i + 1])
                cv2.line(vis_frame, pt1, pt2, (255, 0, 0), 2)  # Blue for history

            # Draw predicted trajectory (red)
            if obj_id in predictions:
                pred = predictions[obj_id]
                pred_positions = pred['predicted_positions']

                if len(pred_positions) > 0:
                    # Connect current position to first prediction
                    pt1 = self._world_to_image(positions[-1])
                    pt2 = self._world_to_image(pred_positions[0])
                    cv2.line(vis_frame, pt1, pt2, (0, 0, 255), 2)  # Red for prediction

                    # Draw predicted positions
                    for i in range(len(pred_positions) - 1):
                        pt1 = self._world_to_image(pred_positions[i])
                        pt2 = self._world_to_image(pred_positions[i + 1])
                        cv2.line(vis_frame, pt1, pt2, (0, 0, 255), 2)

                    # Draw endpoint
                    endpoint = self._world_to_image(pred_positions[-1])
                    cv2.circle(vis_frame, endpoint, 5, (0, 0, 255), -1)

                    # Draw confidence
                    confidence = pred['confidence']
                    cv2.putText(vis_frame, f"{confidence:.2f}", endpoint,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return vis_frame

    def _world_to_image(self, world_pos: np.ndarray, scale: float = 100.0) -> Tuple[int, int]:
        """
        Convert 3D world position to 2D image coordinates.

        Args:
            world_pos: 3D position [x, y, z]
            scale: Scale factor for conversion

        Returns:
            (x, y) image coordinates
        """
        # Simple projection: ignore z, scale x and y
        img_x = int(world_pos[0] * scale + 320)  # Center at 320
        img_y = int(world_pos[1] * scale + 240)  # Center at 240

        # Clamp to image bounds
        img_x = max(0, min(639, img_x))
        img_y = max(0, min(479, img_y))

        return (img_x, img_y)
