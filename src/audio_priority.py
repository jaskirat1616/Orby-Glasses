"""
OrbyGlasses - Intelligent Audio Prioritization System
Manages audio feedback to prevent overwhelming blind users while ensuring critical info is delivered.
"""

import time
from typing import List, Dict, Optional
from collections import deque
from dataclasses import dataclass
import logging


@dataclass
class AudioMessage:
    """Represents an audio message with priority."""
    message: str
    priority: int  # 0-10, where 10 is most critical
    category: str  # 'danger', 'warning', 'info', 'navigation'
    timestamp: float
    urgency_decay: float = 1.0  # How fast urgency decreases


class AudioPriorityManager:
    """
    Manages audio message prioritization for blind navigation.
    Ensures critical safety messages are delivered while preventing information overload.
    """

    # Priority levels
    PRIORITY_CRITICAL = 10  # Immediate danger - always play
    PRIORITY_HIGH = 8  # High importance - play soon
    PRIORITY_MEDIUM = 5  # Medium importance - play when possible
    PRIORITY_LOW = 3  # Low importance - play only if queue empty
    PRIORITY_INFO = 1  # Information only - can be skipped

    def __init__(self, max_queue_size: int = 5, min_message_interval: float = 0.5):
        """
        Initialize audio priority manager.

        Args:
            max_queue_size: Maximum number of queued messages
            min_message_interval: Minimum time between messages (seconds)
        """
        self.max_queue_size = max_queue_size
        self.min_message_interval = min_message_interval

        self.message_queue = deque(maxlen=max_queue_size)
        self.last_message_time = 0
        self.last_message_content = ""
        self.message_history = deque(maxlen=20)  # Track recent messages

        self.logger = logging.getLogger(__name__)

    def add_message(self, message: str, priority: int, category: str = 'info') -> bool:
        """
        Add message to priority queue.

        Args:
            message: Message text
            priority: Priority level (0-10)
            category: Message category

        Returns:
            True if message was added, False if rejected
        """
        # Don't add duplicate of last message
        if message == self.last_message_content:
            return False

        # Check if same message was recently spoken
        recent_similar = any(
            msg.message == message and (time.time() - msg.timestamp) < 3.0
            for msg in self.message_history
        )
        if recent_similar and priority < self.PRIORITY_HIGH:
            return False

        # Create audio message
        audio_msg = AudioMessage(
            message=message,
            priority=priority,
            category=category,
            timestamp=time.time()
        )

        # Critical messages always go first
        if priority >= self.PRIORITY_CRITICAL:
            # Clear queue and add critical message
            self.message_queue.clear()
            self.message_queue.append(audio_msg)
            return True

        # Check if queue is full
        if len(self.message_queue) >= self.max_queue_size:
            # Replace lowest priority message if new message has higher priority
            lowest = min(self.message_queue, key=lambda x: x.priority)
            if priority > lowest.priority:
                self.message_queue.remove(lowest)
                self.message_queue.append(audio_msg)
                return True
            return False

        # Add to queue
        self.message_queue.append(audio_msg)
        return True

    def get_next_message(self) -> Optional[str]:
        """
        Get next message to speak based on priority and timing.

        Returns:
            Message string or None if not ready to speak
        """
        current_time = time.time()

        # Check if enough time has passed
        if current_time - self.last_message_time < self.min_message_interval:
            # Unless it's critical
            if self.message_queue and self.message_queue[0].priority >= self.PRIORITY_CRITICAL:
                pass  # Allow critical messages to bypass interval
            else:
                return None

        if not self.message_queue:
            return None

        # Get highest priority message
        highest_priority_msg = max(self.message_queue, key=lambda x: x.priority)
        self.message_queue.remove(highest_priority_msg)

        # Update state
        self.last_message_time = current_time
        self.last_message_content = highest_priority_msg.message
        self.message_history.append(highest_priority_msg)

        return highest_priority_msg.message

    def create_safety_message(self, warnings: List[Dict], detections: List[Dict] = None) -> Optional[Dict]:
        """
        Create prioritized safety message from warnings with tracking intelligence.

        Args:
            warnings: List of safety warnings
            detections: Full detections for context (optional)

        Returns:
            Message dict with priority or None
        """
        if not warnings:
            return None

        # Get highest priority warning
        top_warning = warnings[0]
        level = top_warning['level']
        obj = top_warning['object']
        dist = top_warning['distance']

        # Get tracking info if available
        is_approaching = False
        is_moving = False
        if detections:
            for det in detections:
                if det.get('label') == obj and abs(det.get('depth', 10) - dist) < 0.5:
                    is_approaching = det.get('is_approaching', False)
                    is_moving = det.get('is_moving', False)
                    break

        # Determine priority and message
        if level == 'IMMEDIATE_DANGER':
            # Critical - immediate threat
            position = self._get_position_description(top_warning['position'])
            msg = f"STOP! {obj} {position} at {dist:.1f} meters"
            if is_approaching:
                msg += " - approaching!"
            return {
                'message': msg,
                'priority': self.PRIORITY_CRITICAL,
                'category': 'danger'
            }

        elif level == 'DANGER':
            # High priority - close obstacle
            position = self._get_position_description(top_warning['position'])
            msg = f"Caution: {obj} {position}, {dist:.1f} meters"
            if is_approaching:
                msg = f"Warning: {obj} approaching {position}, {dist:.1f} meters"
                priority = self.PRIORITY_CRITICAL  # Upgrade if approaching
            else:
                priority = self.PRIORITY_HIGH

            return {
                'message': msg,
                'priority': priority,
                'category': 'warning'
            }

        elif level == 'CAUTION':
            # Medium priority - obstacle ahead but not immediate
            msg = f"{obj} detected {dist:.1f} meters ahead"
            if is_moving:
                msg = f"{obj} moving, {dist:.1f} meters away"

            return {
                'message': msg,
                'priority': self.PRIORITY_MEDIUM,
                'category': 'warning'
            }

        return None

    def _get_position_description(self, position: List[float], frame_width: int = 416) -> str:
        """
        Get position description (left/center/right).

        Args:
            position: [x, y] position in frame
            frame_width: Frame width in pixels

        Returns:
            Position description string
        """
        x = position[0]

        # Divide frame into thirds
        left_third = frame_width / 3
        right_third = 2 * frame_width / 3

        if x < left_third:
            return "on your left"
        elif x > right_third:
            return "on your right"
        else:
            return "directly ahead"

    def create_navigation_message(self, nav_summary: Dict, detections: List[Dict]) -> Optional[Dict]:
        """
        Create navigation guidance message.

        Args:
            nav_summary: Navigation summary
            detections: Current detections

        Returns:
            Message dict or None
        """
        # Check for dangers first
        if nav_summary.get('danger_objects'):
            closest_danger = min(nav_summary['danger_objects'], key=lambda x: x.get('depth', 10))
            return {
                'message': f"{closest_danger['label']} blocking path at {closest_danger['depth']:.1f} meters",
                'priority': self.PRIORITY_HIGH,
                'category': 'navigation'
            }

        # Check for cautions
        if nav_summary.get('caution_objects'):
            closest_caution = min(nav_summary['caution_objects'], key=lambda x: x.get('depth', 10))
            return {
                'message': f"{closest_caution['label']} {closest_caution['depth']:.1f} meters ahead",
                'priority': self.PRIORITY_MEDIUM,
                'category': 'navigation'
            }

        # Path clear
        if nav_summary.get('path_clear'):
            # Low priority - only inform if user might want to know
            if detections:
                # Mention what's nearby but safe
                nearby = [d for d in detections if d.get('depth', 10) < 5.0]
                if nearby and len(nearby) <= 2:
                    objects = ", ".join([d['label'] for d in nearby[:2]])
                    return {
                        'message': f"Path clear. {objects} nearby",
                        'priority': self.PRIORITY_LOW,
                        'category': 'info'
                    }

            return {
                'message': "Path clear",
                'priority': self.PRIORITY_LOW,
                'category': 'info'
            }

        return None

    def queue_size(self) -> int:
        """Get current queue size."""
        return len(self.message_queue)

    def clear_queue(self):
        """Clear all queued messages."""
        self.message_queue.clear()

    def get_statistics(self) -> Dict:
        """Get queue statistics."""
        if not self.message_queue:
            return {
                'queue_size': 0,
                'highest_priority': 0,
                'average_priority': 0
            }

        priorities = [msg.priority for msg in self.message_queue]
        return {
            'queue_size': len(self.message_queue),
            'highest_priority': max(priorities),
            'average_priority': sum(priorities) / len(priorities),
            'oldest_message_age': time.time() - min(msg.timestamp for msg in self.message_queue)
        }
