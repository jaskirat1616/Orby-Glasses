"""
Simple Conversation System
Uses Gemma 3:4b for BOTH conversation AND scene understanding
Replaces heavy VLM with single lightweight model
"""

import ollama
import time
from typing import Dict, List, Optional
import threading
import queue


class SimpleConversation:
    """
    Lightweight conversation system using Gemma 3:4b.
    Handles both:
    1. User questions ("What's ahead?")
    2. Scene understanding (replaces VLM)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize conversation system.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Use same model for everything
        self.model = self.config.get('conversation', {}).get('model', 'gemma3:4b')

        # System prompt for navigation assistant
        self.system_prompt = """You are a helpful navigation assistant for blind users.
You analyze the environment and provide clear, concise directions.
Keep responses under 20 words. Focus on immediate, actionable guidance.
Examples:
- "Person ahead 2 meters. Path clear on left."
- "Door on your right. Go straight."
- "Stairs detected. Stop."""

        # Response queue (non-blocking)
        self.response_queue = queue.Queue()

        # Last scene analysis
        self.last_scene_analysis = None
        self.last_analysis_time = 0

        print(f"✓ Simple conversation initialized with {self.model}")

    def ask_question(self, question: str, context: Dict) -> str:
        """
        Ask a question about the environment.

        Args:
            question: User's question
            context: Current environment context (detections, etc.)

        Returns:
            Response string
        """
        # Build context description
        detections = context.get('detections', [])
        obstacle_count = len([d for d in detections if d.get('depth', 10) < 2.0])

        context_text = f"""Current environment:
- {len(detections)} objects detected
- {obstacle_count} obstacles within 2 meters
- Objects: {', '.join([d['label'] for d in detections[:5]])}
"""

        # Ask Gemma 3
        try:
            response = ollama.generate(
                model=self.model,
                prompt=f"{self.system_prompt}\n\n{context_text}\n\nUser: {question}\nAssistant:",
                options={
                    'temperature': 0.7,
                    'num_predict': 50  # Short responses
                }
            )

            return response['response'].strip()

        except Exception as e:
            return f"Error: {e}"

    def analyze_scene(self, detections: List[Dict]) -> str:
        """
        Analyze current scene (replaces VLM).

        Args:
            detections: List of detected objects

        Returns:
            Scene description and guidance
        """
        # Rate limit (analyze every 3 seconds)
        current_time = time.time()
        if current_time - self.last_analysis_time < 3.0:
            return self.last_scene_analysis or "Clear path"

        if not detections:
            analysis = "Path clear"
            self.last_scene_analysis = analysis
            self.last_analysis_time = current_time
            return analysis

        # Build scene description
        danger_objects = [d for d in detections if d.get('depth', 10) < 1.0]
        close_objects = [d for d in detections if 1.0 <= d.get('depth', 10) < 2.5]

        # Simple rule-based analysis (faster than LLM)
        if danger_objects:
            closest = min(danger_objects, key=lambda x: x['depth'])
            # Determine direction
            center_x = closest.get('center', [160])[0]
            if center_x < 106:
                direction = "Go right"
            elif center_x > 213:
                direction = "Go left"
            else:
                direction = "Stop"

            analysis = f"{closest['label']} at {closest['depth']:.1f}m. {direction}"

        elif close_objects:
            closest = min(close_objects, key=lambda x: x['depth'])
            analysis = f"{closest['label']} ahead {closest['depth']:.1f}m. Proceed with caution"

        else:
            analysis = "Path clear. Safe to proceed"

        self.last_scene_analysis = analysis
        self.last_analysis_time = current_time

        return analysis

    def analyze_scene_with_llm(self, detections: List[Dict]) -> str:
        """
        Use Gemma 3 for complex scene analysis (when needed).

        Args:
            detections: List of detected objects

        Returns:
            LLM-generated scene description
        """
        # Build scene description
        scene_text = "Environment:\n"
        for det in detections[:5]:  # Top 5 objects
            scene_text += f"- {det['label']} at {det.get('depth', 'unknown')}m\n"

        prompt = f"""{self.system_prompt}

{scene_text}

Provide navigation guidance in one sentence."""

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': 0.6,
                    'num_predict': 30
                }
            )

            return response['response'].strip()

        except Exception as e:
            # Fallback to rule-based
            return self.analyze_scene(detections)

    def handle_common_questions(self, question: str, context: Dict) -> Optional[str]:
        """
        Handle common questions with pre-defined answers (faster).

        Args:
            question: User question
            context: Environment context

        Returns:
            Pre-defined answer or None
        """
        question_lower = question.lower()

        detections = context.get('detections', [])

        if 'clear' in question_lower or 'safe' in question_lower:
            obstacles = [d for d in detections if d.get('depth', 10) < 2.5]
            if obstacles:
                return f"{len(obstacles)} obstacles within 2.5 meters"
            else:
                return "Path is clear"

        elif 'what' in question_lower and ('ahead' in question_lower or 'front' in question_lower):
            if not detections:
                return "Nothing detected ahead"
            closest = min(detections, key=lambda x: x.get('depth', 10))
            return f"{closest['label']} at {closest.get('depth', 'unknown')}m"

        elif 'how many' in question_lower:
            return f"{len(detections)} objects detected"

        elif 'where' in question_lower:
            if detections:
                objects_str = ', '.join([d['label'] for d in detections[:3]])
                return f"Detected: {objects_str}"
            return "No objects nearby"

        return None  # Use LLM for complex questions


# Example usage
if __name__ == "__main__":
    # Initialize
    conv = SimpleConversation()

    # Test scene analysis
    test_detections = [
        {'label': 'person', 'depth': 2.3, 'center': [160, 240]},
        {'label': 'chair', 'depth': 1.5, 'center': [80, 240]},
        {'label': 'wall', 'depth': 5.0, 'center': [320, 240]}
    ]

    print("\n=== Scene Analysis (Rule-based) ===")
    analysis = conv.analyze_scene(test_detections)
    print(f"Analysis: {analysis}")

    print("\n=== Scene Analysis (LLM-based) ===")
    llm_analysis = conv.analyze_scene_with_llm(test_detections)
    print(f"LLM Analysis: {llm_analysis}")

    print("\n=== Question Answering ===")
    context = {'detections': test_detections}

    questions = [
        "What's ahead?",
        "Is the path clear?",
        "How many objects?",
        "Where should I go?"
    ]

    for q in questions:
        # Try pre-defined first (faster)
        answer = conv.handle_common_questions(q, context)
        if answer is None:
            # Use LLM for complex questions
            answer = conv.ask_question(q, context)

        print(f"Q: {q}")
        print(f"A: {answer}\n")
