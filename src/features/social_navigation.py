"""
OrbyGlasses - Social Navigation AI
Implements navigation in crowded areas using social norms and conventions
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SocialNorm:
    """Represents a social navigation norm"""
    region: str
    convention: str
    description: str
    implementation: str


@dataclass
class CrowdAnalysis:
    """Result of crowd analysis"""
    crowd_density: str  # 'sparse', 'moderate', 'dense'
    social_gaps: List[Tuple[float, float]]  # (x, y) coordinates of available gaps
    people_positions: List[Dict]  # positions and trajectories of detected people
    suggested_path: str  # 'left', 'right', 'center', 'wait'
    social_norm_advice: str  # specific advice based on social norms


class SocialNavigationAI:
    """
    Social Navigation AI for navigating crowded areas using social norms and conventions.
    """

    def __init__(self):
        """Initialize social navigation system."""
        self.logger = logging.getLogger(__name__)
        
        # Define social navigation norms by region
        self.social_norms = {
            'us': SocialNorm(
                region='us',
                convention='right_side',
                description='In the US, people generally stay to the right in hallways',
                implementation='favor_right_side'
            ),
            'uk': SocialNorm(
                region='uk',
                convention='left_side',
                description='In the UK, people generally stay to the left',
                implementation='favor_left_side'
            ),
            'japan': SocialNorm(
                region='japan',
                convention='left_side',
                description='In Japan, people generally stay to the left',
                implementation='favor_left_side'
            ),
            'default': SocialNorm(
                region='default',
                convention='right_side',
                description='Default: stay to the right in hallways',
                implementation='favor_right_side'
            )
        }
        
        # Current active norm
        self.active_norm = self.social_norms['us']
        self.logger.info("Social Navigation AI initialized")

    def set_region(self, region: str):
        """Set the region-specific social norm."""
        if region in self.social_norms:
            self.active_norm = self.social_norms[region]
            self.logger.info(f"Social norm set to {region}: {self.social_norms[region].description}")
        else:
            self.active_norm = self.social_norms['default']
            self.logger.info(f"Unknown region {region}, using default social norm")

    def analyze_crowd(self, detections: List[Dict]) -> CrowdAnalysis:
        """
        Analyze crowd to provide social navigation guidance.
        
        Args:
            detections: List of detected objects with position and depth info
            
        Returns:
            CrowdAnalysis object with navigation suggestions
        """
        # Filter for people detections
        people_detections = [det for det in detections if det.get('label') == 'person']
        
        if not people_detections:
            return CrowdAnalysis(
                crowd_density='sparse',
                social_gaps=[],
                people_positions=[],
                suggested_path='center',
                social_norm_advice="No people detected, path is clear."
            )
        
        # Analyze crowd density
        crowd_density = self._analyze_crowd_density(people_detections)
        
        # Identify gaps in crowd
        social_gaps = self._find_social_gaps(people_detections)
        
        # Analyze people positions and movement
        people_positions = self._analyze_people_positions(people_detections)
        
        # Determine suggested path based on social norms and crowd
        suggested_path = self._determine_suggested_path(people_detections, social_gaps)
        
        # Generate social norm advice
        social_norm_advice = self._generate_social_advice(
            people_detections, crowd_density, suggested_path
        )
        
        return CrowdAnalysis(
            crowd_density=crowd_density,
            social_gaps=social_gaps,
            people_positions=people_positions,
            suggested_path=suggested_path,
            social_norm_advice=social_norm_advice
        )

    def _analyze_crowd_density(self, people_detections: List[Dict]) -> str:
        """Analyze crowd density based on number and proximity of people."""
        if len(people_detections) == 0:
            return 'sparse'
        
        # Count people in different distance ranges (handle None depth)
        close_people = [p for p in people_detections if p.get('depth') is not None and p.get('depth') < 1.5]
        medium_people = [p for p in people_detections if p.get('depth') is not None and 1.5 <= p.get('depth') < 3.0]
        
        if len(close_people) >= 3 or len(people_detections) >= 5:
            return 'dense'
        elif len(close_people) >= 1 or len(people_detections) >= 2:
            return 'moderate'
        else:
            return 'sparse'

    def _find_social_gaps(self, people_detections: List[Dict]) -> List[Tuple[float, float]]:
        """Find available gaps in crowd for safe passage."""
        if not people_detections:
            return [(0.5, 0.5)]  # Center position if no people
        
        # Extract x positions of people (normalized to 0-1 range)
        x_positions = []
        for person in people_detections:
            center_x = person.get('center', [0.5, 0.5])[0]
            # Assuming frame width of 320 (from config), normalize to 0-1
            x_normalized = center_x / 320.0
            x_positions.append(max(0.1, min(0.9, x_normalized)))  # Keep within bounds
        
        # Find gaps between people (simplified approach)
        gaps = []
        x_positions = sorted(x_positions)
        
        # Check for gaps on the sides and between people
        if x_positions:
            # Left side
            if x_positions[0] > 0.25:
                gaps.append((0.1, 0.5))
            # Right side  
            if (1.0 - x_positions[-1]) > 0.25:
                gaps.append((0.9, 0.5))
            # Between people if spacing allows
            for i in range(len(x_positions) - 1):
                mid_point = (x_positions[i] + x_positions[i + 1]) / 2
                if abs(x_positions[i] - x_positions[i + 1]) > 0.3:
                    gaps.append((mid_point, 0.5))
        else:
            gaps.append((0.5, 0.5))
        
        return gaps

    def _analyze_people_positions(self, people_detections: List[Dict]) -> List[Dict]:
        """Analyze positions of detected people."""
        positions = []
        for person in people_detections:
            pos_info = {
                'x': person.get('center', [0, 0])[0],
                'y': person.get('center', [0, 0])[1],
                'depth': person.get('depth', 0),
                'bbox': person.get('bbox', [0, 0, 0, 0])
            }
            positions.append(pos_info)
        return positions

    def _determine_suggested_path(self, people_detections: List[Dict], 
                                 social_gaps: List[Tuple[float, float]]) -> str:
        """Determine the best path based on social norms and available gaps."""
        if not people_detections:
            return 'center'
        
        # Get crowd density
        crowd_density = self._analyze_crowd_density(people_detections)
        
        # Apply social norm based on region
        if self.active_norm.implementation == 'favor_right_side':
            # In US, favor the right side
            if social_gaps:
                # Find the rightmost gap if available
                rightmost_gap = max(social_gaps, key=lambda g: g[0])
                if rightmost_gap[0] > 0.6:  # Right side
                    return 'right'
                elif rightmost_gap[0] < 0.4:  # Left side
                    return 'left'
                else:
                    return 'center'
        elif self.active_norm.implementation == 'favor_left_side':
            # In UK/Japan, favor the left side
            if social_gaps:
                leftmost_gap = min(social_gaps, key=lambda g: g[0])
                if leftmost_gap[0] < 0.4:  # Left side
                    return 'left'
                elif leftmost_gap[0] > 0.6:  # Right side
                    return 'right'
                else:
                    return 'center'
        
        # Default: choose the first available gap or center
        if social_gaps:
            gap = social_gaps[0]
            if gap[0] < 0.4:
                return 'left'
            elif gap[0] > 0.6:
                return 'right'
            else:
                return 'center'
        
        return 'wait'

    def _generate_social_advice(self, people_detections: List[Dict], 
                               crowd_density: str, suggested_path: str) -> str:
        """Generate social navigation advice string."""
        if not people_detections:
            return "Path is clear, no people detected ahead."
        
        advice_parts = []
        
        # Add social norm reminder based on region
        if self.active_norm.region in ['us']:
            advice_parts.append(f"Following US convention: stay to the {self.active_norm.convention.replace('_side', '')} in hallways.")
        elif self.active_norm.region in ['uk', 'japan']:
            advice_parts.append(f"Following {self.active_norm.region.upper()} convention: stay to the {self.active_norm.convention.replace('_side', '')}.")
        else:
            advice_parts.append(f"Following standard convention: stay to the {self.active_norm.convention.replace('_side', '')}.")
        
        # Describe crowd density
        if crowd_density == 'dense':
            advice_parts.append("Crowd is dense ahead.")
        elif crowd_density == 'moderate':
            advice_parts.append("Moderate number of people ahead.")
        else:
            advice_parts.append("Few people ahead.")
        
        # Suggest navigation based on detected gaps
        if suggested_path == 'right' and crowd_density != 'sparse':
            if self.active_norm.implementation == 'favor_right_side':
                advice_parts.append("Gap opening in crowd ahead on your right. Safe to proceed to the right.")
            else:
                advice_parts.append("Gap on your right, safe to move right while respecting social norms.")
        elif suggested_path == 'left' and crowd_density != 'sparse':
            if self.active_norm.implementation == 'favor_left_side':
                advice_parts.append("Gap opening in crowd ahead on your left. Safe to proceed to the left.")
            else:
                advice_parts.append("Gap on your left, safe to move left while respecting social norms.")
        elif suggested_path == 'center':
            advice_parts.append("Path is clear ahead in the center.")
        elif suggested_path == 'wait':
            advice_parts.append("Wait for gap in crowd to open.")
        
        # Additional context about people behavior
        if people_detections:
            closest_person = min(people_detections, key=lambda x: x.get('depth') if x.get('depth') is not None else float('inf'))
            closest_depth = closest_person.get('depth')
            if closest_depth is not None and closest_depth < 1.5:
                advice_parts.append(f"Person {closest_depth:.1f}m ahead.")

                # If people are yielding space
                if len(people_detections) == 1 and closest_depth > 1.0:
                    advice_parts.append("Person appears to be yielding space, safe to proceed.")
        
        return " ".join(advice_parts)

    def get_social_navigation_guidance(self, detections: List[Dict], 
                                     user_request: str = "") -> str:
        """
        Get social navigation guidance based on current detections and optional user request.
        
        Args:
            detections: List of detected objects
            user_request: Optional user request for specific guidance
            
        Returns:
            Social navigation guidance string
        """
        analysis = self.analyze_crowd(detections)
        
        # Return the social norm advice generated
        return analysis.social_norm_advice

    def update_social_context(self, detections: List[Dict]) -> Dict:
        """
        Update social navigation context and return detailed information.
        
        Args:
            detections: List of detected objects
            
        Returns:
            Dictionary with social navigation context
        """
        analysis = self.analyze_crowd(detections)
        
        return {
            'crowd_density': analysis.crowd_density,
            'social_gaps': analysis.social_gaps,
            'people_positions': analysis.people_positions,
            'suggested_path': analysis.suggested_path,
            'social_advice': analysis.social_norm_advice,
            'social_norm_region': self.active_norm.region,
            'social_norm_convention': self.active_norm.convention
        }