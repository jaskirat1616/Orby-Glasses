"""
Blind Navigation - Solving Real Pain Points
Focus on what blind people actually need to navigate safely and independently
"""
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging


class PathFinder:
    """Finds safe walking paths - THE most important thing for blind people."""

    def __init__(self, frame_width: int = 416, frame_height: int = 416):
        """Initialize path finder."""
        self.frame_width = frame_width
        self.frame_height = frame_height

    def find_clear_path(self, detections: List[Dict], depth_map: Optional[np.ndarray]) -> Dict:
        """
        Find WHERE the user can safely walk.

        This is critical - blind people need to know:
        - Where is safe to walk RIGHT NOW
        - Which direction to move
        - What obstacles are blocking the path

        Args:
            detections: Detected objects
            depth_map: Depth map

        Returns:
            Path info with clear directions
        """
        # Divide view into 3 zones: LEFT, CENTER, RIGHT
        left_zone = []
        center_zone = []
        right_zone = []

        for det in detections:
            center_x = det['center'][0]
            depth = det.get('depth', 10)

            # Only care about close obstacles (< 3 meters)
            if depth > 3.0:
                continue

            # Categorize by position
            if center_x < self.frame_width / 3:
                left_zone.append(det)
            elif center_x > 2 * self.frame_width / 3:
                right_zone.append(det)
            else:
                center_zone.append(det)

        # Find clearest path
        left_clear = len(left_zone) == 0 or min([d.get('depth', 10) for d in left_zone]) > 2.0
        center_clear = len(center_zone) == 0 or min([d.get('depth', 10) for d in center_zone]) > 2.0
        right_clear = len(right_zone) == 0 or min([d.get('depth', 10) for d in right_zone]) > 2.0

        # Generate simple, clear guidance
        if center_clear:
            direction = "straight ahead"
            confidence = "high"
        elif left_clear and not right_clear:
            direction = "to your left"
            confidence = "high"
        elif right_clear and not left_clear:
            direction = "to your right"
            confidence = "high"
        elif left_clear and right_clear:
            direction = "left or right, not straight"
            confidence = "medium"
        else:
            direction = "blocked - stop and assess"
            confidence = "low"

        return {
            'direction': direction,
            'confidence': confidence,
            'left_clear': left_clear,
            'center_clear': center_clear,
            'right_clear': right_clear,
            'obstacles_left': len(left_zone),
            'obstacles_center': len(center_zone),
            'obstacles_right': len(right_zone)
        }


class ImportantObjectDetector:
    """Detects objects that MATTER to blind people."""

    CRITICAL_OBJECTS = {
        # Navigation
        'door': 'Door',
        'doorway': 'Doorway',
        'stairs': 'Stairs',
        'step': 'Step',
        'curb': 'Curb',
        'elevator': 'Elevator',
        'escalator': 'Escalator',

        # Hazards
        'pole': 'Pole',
        'fire hydrant': 'Fire hydrant',
        'bench': 'Bench',
        'chair': 'Chair',
        'table': 'Table',

        # People & vehicles
        'person': 'Person',
        'bicycle': 'Bicycle',
        'car': 'Car',
        'motorcycle': 'Motorcycle',
        'bus': 'Bus',
        'truck': 'Truck',

        # Landmarks
        'traffic light': 'Traffic light',
        'stop sign': 'Stop sign',
        'street sign': 'Street sign'
    }

    def filter_important_objects(self, detections: List[Dict]) -> List[Dict]:
        """
        Keep only objects that matter for navigation.

        Args:
            detections: All detected objects

        Returns:
            Filtered list of important objects only
        """
        important = []
        for det in detections:
            label = det.get('label', '').lower()

            # Check if it's a critical object
            for key in self.CRITICAL_OBJECTS:
                if key in label:
                    det['importance'] = 'critical'
                    det['user_friendly_name'] = self.CRITICAL_OBJECTS[key]
                    important.append(det)
                    break

        return important

    def find_hazards(self, detections: List[Dict]) -> List[Dict]:
        """
        Find objects that are hazards (head-height, trip hazards, etc).

        Args:
            detections: All detections

        Returns:
            List of hazards
        """
        hazards = []

        HAZARD_OBJECTS = ['pole', 'fire hydrant', 'sign', 'branch', 'low ceiling']
        TRIP_HAZARDS = ['step', 'stairs', 'curb', 'hole', 'cable']

        for det in detections:
            label = det.get('label', '').lower()
            depth = det.get('depth', 10)

            # Head-height hazards (poles, signs, branches)
            for hazard in HAZARD_OBJECTS:
                if hazard in label and depth < 2.0:
                    det['hazard_type'] = 'head'
                    det['warning'] = f"{det['label']} at head height"
                    hazards.append(det)
                    break

            # Trip hazards (stairs, steps, curbs)
            for trip in TRIP_HAZARDS:
                if trip in label and depth < 2.0:
                    det['hazard_type'] = 'trip'
                    det['warning'] = f"{det['label']} ahead - trip hazard"
                    hazards.append(det)
                    break

        return hazards


class BlindNavigationAssistant:
    """
    Main assistant that provides CLEAR, ACTIONABLE guidance.

    Key principles:
    1. Be CLEAR - no jargon, simple language
    2. Be SPECIFIC - exact directions, not vague
    3. Be TIMELY - warn before it's too late
    4. Be HELPFUL - tell them what to DO, not just what's there
    """

    def __init__(self, frame_width: int = 416, frame_height: int = 416):
        """Initialize navigation assistant."""
        self.path_finder = PathFinder(frame_width, frame_height)
        self.object_detector = ImportantObjectDetector()
        self.last_guidance = ""
        self.guidance_count = 0

    def get_navigation_guidance(self, detections: List[Dict], depth_map: Optional[np.ndarray] = None) -> str:
        """
        Get CLEAR, ACTIONABLE navigation guidance.

        This is what blind users need to hear:
        - Where to walk
        - What to avoid
        - When to stop
        - How to proceed

        Args:
            detections: All detections
            depth_map: Depth map

        Returns:
            Clear, simple guidance message
        """
        # Filter to important objects only
        important = self.object_detector.filter_important_objects(detections)

        # Find immediate hazards
        hazards = self.object_detector.find_hazards(important)

        # Check for immediate dangers (< 1 meter)
        immediate_danger = [d for d in important if d.get('depth', 10) < 1.0]

        # PRIORITY 1: Immediate danger - STOP
        if immediate_danger:
            closest = min(immediate_danger, key=lambda x: x.get('depth', 10))
            label = closest.get('user_friendly_name', closest['label'])
            depth = closest['depth']
            position = self._get_simple_position(closest['center'][0])

            return f"STOP! {label} {position}, {depth:.1f} meters. Do not move forward."

        # PRIORITY 2: Hazards - WARN
        if hazards:
            hazard = hazards[0]
            warning = hazard.get('warning', hazard['label'])
            position = self._get_simple_position(hazard['center'][0])
            return f"Warning: {warning} {position}."

        # PRIORITY 3: Path guidance - WHERE TO WALK
        path_info = self.path_finder.find_clear_path(important, depth_map)

        if path_info['confidence'] == 'high':
            direction = path_info['direction']

            # Mention what's nearby if relevant
            nearby = [d for d in important if 1.0 < d.get('depth', 10) < 2.5]
            if nearby:
                obj = nearby[0]
                label = obj.get('user_friendly_name', obj['label'])
                obj_position = self._get_simple_position(obj['center'][0])
                return f"Path clear {direction}. {label} {obj_position} at {obj['depth']:.1f} meters."
            else:
                return f"Path clear {direction}."

        elif path_info['confidence'] == 'medium':
            return f"Multiple obstacles. Safe path {path_info['direction']}."

        else:
            # Path blocked
            return "Path blocked ahead. Stop and listen for guidance."

    def get_landmark_guidance(self, detections: List[Dict]) -> Optional[str]:
        """
        Help user understand landmarks around them.

        This helps with:
        - "Where am I?"
        - "Am I at the door?"
        - "Is there a crosswalk?"

        Args:
            detections: All detections

        Returns:
            Landmark description or None
        """
        LANDMARKS = ['door', 'traffic light', 'stop sign', 'crosswalk', 'bench', 'stairs']

        for det in detections:
            label = det.get('label', '').lower()
            for landmark in LANDMARKS:
                if landmark in label:
                    depth = det.get('depth', 10)
                    position = self._get_simple_position(det['center'][0])

                    if depth < 3.0:  # Only mention if close enough
                        return f"{det['label'].title()} {position}, {depth:.1f} meters away."

        return None

    def _get_simple_position(self, x: float) -> str:
        """Get simple position description."""
        if x < self.path_finder.frame_width / 3:
            return "on your left"
        elif x > 2 * self.path_finder.frame_width / 3:
            return "on your right"
        else:
            return "directly ahead"

    def should_speak_now(self, guidance: str, min_interval: float = 2.0) -> bool:
        """
        Decide if we should speak this guidance now.

        Avoid:
        - Repeating the same thing
        - Speaking too frequently
        - Information overload

        Args:
            guidance: The guidance message
            min_interval: Minimum seconds between messages

        Returns:
            True if should speak
        """
        import time

        # Different message - speak it
        if guidance != self.last_guidance:
            self.last_guidance = guidance
            self.guidance_count = 0
            return True

        # Same message - only repeat occasionally
        self.guidance_count += 1
        if self.guidance_count >= 5:  # Repeat every 5 times (about 10 seconds)
            self.guidance_count = 0
            return True

        return False
