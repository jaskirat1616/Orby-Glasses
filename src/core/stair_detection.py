"""
Stair and Curb Detection for OrbyGlasses
Uses depth discontinuities to detect vertical drops and elevation changes.

Critical safety feature to prevent falls - the #1 injury cause for blind users.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional


class StairCurbDetector:
    """
    Detects stairs, curbs, and other vertical discontinuities using depth analysis.

    Detection methods:
    1. Depth gradient analysis - Find sudden depth changes
    2. Horizontal edge detection - Look for horizontal lines at ground level
    3. Step pattern matching - Detect repeating stair patterns
    """

    def __init__(self, config: Dict):
        """Initialize stair/curb detector."""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Detection thresholds
        self.min_drop_height = config.get('stair_detection.min_drop_height', 0.15)  # 15cm minimum drop
        self.stair_height_range = config.get('stair_detection.stair_height_range', (0.15, 0.25))  # Typical stair: 15-25cm
        self.detection_distance = config.get('stair_detection.detection_distance', 2.5)  # Detect up to 2.5m ahead
        self.ground_roi_bottom = config.get('stair_detection.ground_roi_bottom', 0.7)  # Bottom 30% of frame
        self.ground_roi_top = config.get('stair_detection.ground_roi_top', 0.4)  # Middle 30% for ground

        # Gradient thresholds
        self.depth_gradient_threshold = config.get('stair_detection.gradient_threshold', 0.3)  # 30cm/meter change
        self.edge_confidence_threshold = config.get('stair_detection.edge_confidence', 0.6)  # 60% confidence minimum

        # State
        self.last_detection = None
        self.detection_count = 0
        self.false_positive_filter = 3  # Require N consecutive detections

        self.logger.info("✓ Stair/Curb detector initialized")
        self.logger.info(f"  → Min drop height: {self.min_drop_height}m")
        self.logger.info(f"  → Detection distance: {self.detection_distance}m")

    def detect(self, depth_map: np.ndarray, frame: np.ndarray) -> Dict:
        """
        Detect stairs and curbs in the depth map.

        Args:
            depth_map: Depth map (HxW) in meters
            frame: Original BGR frame for visualization

        Returns:
            Dictionary with detection results:
            {
                'has_stair': bool,
                'has_curb': bool,
                'drop_detected': bool,
                'distance_to_hazard': float,
                'hazard_type': str,  # 'stair_down', 'stair_up', 'curb', 'drop', 'none'
                'confidence': float,
                'warning_level': str,  # 'danger', 'caution', 'safe'
                'visualization': np.ndarray  # Annotated frame
            }
        """
        try:
            if depth_map is None or depth_map.size == 0:
                return self._empty_result(frame)

            h, w = depth_map.shape

            # Focus on ground area (bottom-middle of frame where ground should be)
            roi_top = int(h * self.ground_roi_top)
            roi_bottom = int(h * self.ground_roi_bottom)
            roi_left = int(w * 0.2)  # Focus on center 60% horizontally
            roi_right = int(w * 0.8)

            ground_depth = depth_map[roi_top:roi_bottom, roi_left:roi_right]

            if ground_depth.size == 0:
                return self._empty_result(frame)

            # Method 1: Depth gradient analysis (detect sudden drops)
            drop_result = self._detect_depth_discontinuity(ground_depth, (roi_top, roi_bottom, roi_left, roi_right))

            # Method 2: Horizontal edge detection (detect stair edges)
            edge_result = self._detect_horizontal_edges(depth_map, frame)

            # Method 3: Step pattern matching
            pattern_result = self._detect_step_pattern(ground_depth)

            # Combine results
            combined_result = self._combine_detection_results(drop_result, edge_result, pattern_result)

            # Add visualization
            combined_result['visualization'] = self._visualize_detection(frame, combined_result, depth_map)

            # Update state for temporal filtering
            if combined_result['drop_detected']:
                self.detection_count += 1
            else:
                self.detection_count = max(0, self.detection_count - 1)

            # Require multiple consecutive detections to reduce false positives
            if self.detection_count < self.false_positive_filter:
                combined_result['drop_detected'] = False
                combined_result['has_stair'] = False
                combined_result['has_curb'] = False
                combined_result['warning_level'] = 'safe'

            self.last_detection = combined_result

            return combined_result

        except Exception as e:
            self.logger.error(f"Stair detection error: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._empty_result(frame)

    def _detect_depth_discontinuity(self, ground_depth: np.ndarray, roi_coords: Tuple) -> Dict:
        """
        Detect sudden depth changes indicating drops or elevation changes.

        Args:
            ground_depth: Depth map of ground region
            roi_coords: (top, bottom, left, right) coordinates of ROI

        Returns:
            Detection result dictionary
        """
        roi_top, roi_bottom, roi_left, roi_right = roi_coords

        # Calculate vertical gradient (depth change from top to bottom of ROI)
        if ground_depth.shape[0] < 5:
            return {'detected': False, 'distance': None, 'confidence': 0.0}

        # Compute row-wise median depth (more robust than mean)
        row_depths = np.median(ground_depth, axis=1)

        # Find maximum depth gradient between consecutive rows
        depth_gradients = np.diff(row_depths)

        # Positive gradient = depth increasing = ground going away (stairs down or drop)
        # Negative gradient = depth decreasing = ground coming closer (stairs up or obstacle)

        max_positive_grad_idx = np.argmax(depth_gradients)
        max_positive_grad = depth_gradients[max_positive_grad_idx]

        max_negative_grad_idx = np.argmin(depth_gradients)
        max_negative_grad = depth_gradients[max_negative_grad_idx]

        # Check for significant drop (positive gradient)
        if max_positive_grad > self.min_drop_height:
            # Depth suddenly increased = drop detected
            drop_distance = row_depths[max_positive_grad_idx]
            if drop_distance < self.detection_distance:
                return {
                    'detected': True,
                    'distance': float(drop_distance),
                    'confidence': min(1.0, max_positive_grad / 0.5),  # Normalize by 50cm max expected
                    'type': 'drop',
                    'gradient': float(max_positive_grad)
                }

        # Check for significant rise (negative gradient - stairs up)
        if abs(max_negative_grad) > self.min_drop_height:
            rise_distance = row_depths[max_negative_grad_idx]
            if rise_distance < self.detection_distance:
                return {
                    'detected': True,
                    'distance': float(rise_distance),
                    'confidence': min(1.0, abs(max_negative_grad) / 0.5),
                    'type': 'rise',
                    'gradient': float(max_negative_grad)
                }

        return {'detected': False, 'distance': None, 'confidence': 0.0}

    def _detect_horizontal_edges(self, depth_map: np.ndarray, frame: np.ndarray) -> Dict:
        """
        Detect horizontal edges in depth map that might indicate stair edges.

        Args:
            depth_map: Full depth map
            frame: Original frame for texture analysis

        Returns:
            Detection result dictionary
        """
        try:
            # Convert depth to uint8 for edge detection
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Apply Sobel filter to detect horizontal edges
            sobelx = cv2.Sobel(depth_normalized, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(depth_normalized, cv2.CV_64F, 0, 1, ksize=3)

            # We care more about vertical gradients (horizontal edges)
            sobely_abs = np.abs(sobely)

            # Threshold to find strong horizontal edges
            edge_threshold = np.percentile(sobely_abs, 95)  # Top 5% of gradients
            edges = sobely_abs > edge_threshold

            # Focus on ground region
            h, w = edges.shape
            ground_edges = edges[int(h*0.5):int(h*0.8), int(w*0.2):int(w*0.8)]

            # Count horizontal edge pixels
            edge_density = np.sum(ground_edges) / ground_edges.size

            if edge_density > 0.05:  # More than 5% of ground area has strong edges
                return {
                    'detected': True,
                    'confidence': min(1.0, edge_density / 0.2),
                    'type': 'edge_detected'
                }

            return {'detected': False, 'confidence': 0.0}

        except Exception as e:
            self.logger.debug(f"Edge detection error: {e}")
            return {'detected': False, 'confidence': 0.0}

    def _detect_step_pattern(self, ground_depth: np.ndarray) -> Dict:
        """
        Detect repeating step patterns indicative of stairs.

        Args:
            ground_depth: Depth map of ground region

        Returns:
            Detection result dictionary
        """
        try:
            if ground_depth.shape[0] < 10:
                return {'detected': False, 'confidence': 0.0}

            # Calculate row-wise median depths
            row_depths = np.median(ground_depth, axis=1)

            # Look for periodic depth changes
            # Stairs typically have depth increasing in steps
            depth_diff = np.diff(row_depths)

            # Find sequences of positive gradients (depth increasing)
            positive_steps = depth_diff > self.stair_height_range[0]

            # Count consecutive step patterns
            step_count = 0
            max_step_sequence = 0
            for is_step in positive_steps:
                if is_step:
                    step_count += 1
                    max_step_sequence = max(max_step_sequence, step_count)
                else:
                    step_count = 0

            # If we found 3+ consecutive steps, likely stairs
            if max_step_sequence >= 3:
                return {
                    'detected': True,
                    'confidence': min(1.0, max_step_sequence / 5.0),
                    'type': 'stair_pattern',
                    'step_count': int(max_step_sequence)
                }

            return {'detected': False, 'confidence': 0.0}

        except Exception as e:
            self.logger.debug(f"Pattern detection error: {e}")
            return {'detected': False, 'confidence': 0.0}

    def _combine_detection_results(self, drop_result: Dict, edge_result: Dict, pattern_result: Dict) -> Dict:
        """Combine results from multiple detection methods."""

        # Check if any method detected something
        drop_detected = drop_result.get('detected', False)
        edge_detected = edge_result.get('detected', False)
        pattern_detected = pattern_result.get('detected', False)

        has_detection = drop_detected or edge_detected or pattern_detected

        if not has_detection:
            return {
                'has_stair': False,
                'has_curb': False,
                'drop_detected': False,
                'distance_to_hazard': None,
                'hazard_type': 'none',
                'confidence': 0.0,
                'warning_level': 'safe'
            }

        # Determine hazard type and distance
        hazard_type = 'none'
        distance = None
        confidence = 0.0

        # Pattern detected = stairs
        if pattern_detected:
            hazard_type = 'stair_down'
            confidence = max(confidence, pattern_result.get('confidence', 0.0))
            if drop_detected:
                distance = drop_result.get('distance')

        # Drop detected without pattern = curb or single drop
        elif drop_detected:
            drop_type = drop_result.get('type', 'drop')
            if drop_type == 'drop':
                hazard_type = 'curb' if drop_result.get('gradient', 0) < 0.3 else 'drop'
            else:  # rise
                hazard_type = 'obstacle'  # Elevation increase ahead

            distance = drop_result.get('distance')
            confidence = drop_result.get('confidence', 0.0)

        # Edge detected without drop/pattern = possible stair edge
        elif edge_detected:
            hazard_type = 'possible_stair'
            confidence = edge_result.get('confidence', 0.0) * 0.5  # Lower confidence for edge-only

        # Boost confidence if multiple methods agree
        if sum([drop_detected, edge_detected, pattern_detected]) >= 2:
            confidence = min(1.0, confidence * 1.3)

        # Determine warning level
        warning_level = 'safe'
        if distance is not None:
            if distance < 1.0:
                warning_level = 'danger'
            elif distance < 2.0:
                warning_level = 'caution'

        return {
            'has_stair': 'stair' in hazard_type,
            'has_curb': hazard_type == 'curb',
            'drop_detected': has_detection,
            'distance_to_hazard': distance,
            'hazard_type': hazard_type,
            'confidence': confidence,
            'warning_level': warning_level,
            'detection_details': {
                'drop': drop_result,
                'edge': edge_result,
                'pattern': pattern_result
            }
        }

    def _visualize_detection(self, frame: np.ndarray, result: Dict, depth_map: np.ndarray) -> np.ndarray:
        """Add visualization overlay for detected hazards."""
        vis_frame = frame.copy()

        if not result.get('drop_detected', False):
            return vis_frame

        h, w = vis_frame.shape[:2]

        # Draw warning banner
        hazard_type = result.get('hazard_type', 'unknown')
        warning_level = result.get('warning_level', 'safe')

        if warning_level == 'danger':
            color = (0, 0, 255)  # Red
            text = f"DANGER: {hazard_type.upper()} AHEAD!"
        elif warning_level == 'caution':
            color = (0, 165, 255)  # Orange
            text = f"CAUTION: {hazard_type.replace('_', ' ').title()}"
        else:
            color = (0, 255, 255)  # Yellow
            text = f"Notice: {hazard_type.replace('_', ' ').title()}"

        # Draw warning banner at top
        cv2.rectangle(vis_frame, (0, 0), (w, 60), color, -1)
        cv2.putText(vis_frame, text, (10, 40),
                    cv2.FONT_HERSHEY_BOLD, 1.2, (255, 255, 255), 3)

        # Draw distance if available
        distance = result.get('distance_to_hazard')
        if distance is not None:
            dist_text = f"Distance: {distance:.1f}m"
            cv2.putText(vis_frame, dist_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Draw ground ROI for debugging
        roi_top = int(h * self.ground_roi_top)
        roi_bottom = int(h * self.ground_roi_bottom)
        cv2.rectangle(vis_frame, (int(w*0.2), roi_top), (int(w*0.8), roi_bottom),
                      (0, 255, 0), 2)

        return vis_frame

    def _empty_result(self, frame: np.ndarray) -> Dict:
        """Return empty detection result."""
        return {
            'has_stair': False,
            'has_curb': False,
            'drop_detected': False,
            'distance_to_hazard': None,
            'hazard_type': 'none',
            'confidence': 0.0,
            'warning_level': 'safe',
            'visualization': frame.copy() if frame is not None else None
        }
