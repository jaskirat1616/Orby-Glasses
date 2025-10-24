"""
Simple Visual SLAM
Tracks position indoors using camera
"""

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple


class SimpleSLAM:
    """
    Simple visual SLAM for indoor tracking.
    Uses ORB features for robust tracking.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize SLAM system.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # ORB feature detector
        n_features = self.config.get('slam', {}).get('orb_features', 2000)
        self.orb = cv2.ORB_create(nfeatures=n_features)

        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # State
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, z in meters
        self.pose = np.eye(4)  # 4x4 transformation matrix

        # Map points
        self.map_points = []

        # Tracking quality
        self.tracking_quality = 0.0
        self.num_matches = 0

    def process_frame(self, frame: np.ndarray, depth_map: Optional[np.ndarray] = None) -> Dict:
        """
        Process frame and update position.

        Args:
            frame: Input image (BGR)
            depth_map: Optional depth map for scale

        Returns:
            SLAM result dictionary
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        # First frame initialization
        if self.prev_frame is None:
            self.prev_frame = gray
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            return {
                'position': self.position.copy(),
                'pose': self.pose.copy(),
                'tracking_quality': 1.0,
                'num_matches': len(keypoints),
                'num_map_points': len(self.map_points),
                'is_keyframe': True
            }

        # Match features
        if descriptors is not None and self.prev_descriptors is not None:
            matches = self.matcher.knnMatch(self.prev_descriptors, descriptors, k=2)

            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            self.num_matches = len(good_matches)

            # Estimate motion if enough matches
            if len(good_matches) >= 10:
                # Get matched keypoints
                pts1 = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches])
                pts2 = np.float32([keypoints[m.trainIdx].pt for m in good_matches])

                # Estimate essential matrix
                E, mask = cv2.findEssentialMat(pts1, pts2, focal=500.0, pp=(frame.shape[1]/2, frame.shape[0]/2))

                # Recover pose
                if E is not None:
                    _, R, t, mask = cv2.recoverPose(E, pts1, pts2)

                    # Update position (simple integration)
                    # Scale translation using depth if available
                    if depth_map is not None:
                        scale = self._estimate_scale(pts1, pts2, depth_map)
                    else:
                        scale = 0.1  # Default scale

                    t_scaled = t * scale

                    # Update pose
                    self.position += (self.pose[:3, :3] @ t_scaled).flatten()
                    self.pose[:3, :3] = self.pose[:3, :3] @ R
                    self.pose[:3, 3] = self.position

                    # Update tracking quality
                    self.tracking_quality = min(len(good_matches) / 100.0, 1.0)

        # Update previous frame
        self.prev_frame = gray
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        # Add to map if enough motion
        if len(keypoints) > 50:
            self.map_points.append({
                'keypoints': keypoints,
                'descriptors': descriptors,
                'position': self.position.copy()
            })

            # Keep only recent map points
            if len(self.map_points) > 100:
                self.map_points.pop(0)

        return {
            'position': self.position.copy(),
            'pose': self.pose.copy(),
            'tracking_quality': self.tracking_quality,
            'num_matches': self.num_matches,
            'num_map_points': len(self.map_points),
            'is_keyframe': len(keypoints) > 100
        }

    def _estimate_scale(self, pts1: np.ndarray, pts2: np.ndarray, depth_map: np.ndarray) -> float:
        """
        Estimate scale from depth map.

        Args:
            pts1: Points in frame 1
            pts2: Points in frame 2
            depth_map: Depth map

        Returns:
            Scale factor
        """
        scales = []

        for p1, p2 in zip(pts1, pts2):
            # Get depth at point 1
            x1, y1 = int(p1[0]), int(p1[1])
            if 0 <= y1 < depth_map.shape[0] and 0 <= x1 < depth_map.shape[1]:
                depth = depth_map[y1, x1]
                if depth > 0:
                    # Calculate pixel displacement
                    displacement = np.linalg.norm(p2 - p1)
                    if displacement > 1:
                        scale = depth / (displacement * 100)  # Rough scale
                        scales.append(scale)

        if len(scales) > 0:
            return np.median(scales)
        return 0.1

    def reset(self):
        """Reset SLAM state."""
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.position = np.array([0.0, 0.0, 0.0])
        self.pose = np.eye(4)
        self.map_points = []
        self.tracking_quality = 0.0
        self.num_matches = 0


# Example usage
if __name__ == "__main__":
    # Initialize SLAM
    slam = SimpleSLAM()

    # Test with webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        result = slam.process_frame(frame)

        # Draw position
        text = f"Position: ({result['position'][0]:.2f}, {result['position'][1]:.2f}, {result['position'][2]:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        text = f"Quality: {result['tracking_quality']:.2f} | Matches: {result['num_matches']}"
        cv2.putText(frame, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display
        cv2.imshow('Simple SLAM', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
