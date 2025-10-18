"""
OrbyGlasses - AI Narrative Generation
Uses Ollama with Gemma/Moondream for contextual navigation guidance.
"""

import base64
import cv2
import numpy as np
import logging
import ollama
from typing import List, Dict, Optional
import json


class NarrativeGenerator:
    """
    Generates contextual navigation narratives using LLMs.
    Uses Moondream for vision understanding and Gemma for narrative generation.
    """

    def __init__(self, config):
        """
        Initialize narrative generator.

        Args:
            config: ConfigManager instance
        """
        self.config = config

        # Model configuration
        self.primary_model = config.get('models.llm.primary', 'gemma2:2b')
        self.vision_model = config.get('models.llm.vision', 'moondream')
        self.temperature = config.get('models.llm.temperature', 0.7)
        self.max_tokens = config.get('models.llm.max_tokens', 150)

        # Check available models
        self.available_models = self._check_available_models()

        logging.info(f"Narrative generator initialized with models: {self.available_models}")

        # Context window for recent detections
        self.context_history = []
        self.max_context_length = 5

    def _check_available_models(self) -> List[str]:
        """Check which Ollama models are available."""
        try:
            models = ollama.list()
            available = [m['name'] for m in models.get('models', [])]
            logging.info(f"Available Ollama models: {available}")
            return available
        except Exception as e:
            logging.warning(f"Could not check Ollama models: {e}")
            return []

    def _encode_image(self, frame: np.ndarray) -> str:
        """
        Encode frame to base64 for multimodal LLM input.

        Args:
            frame: Input frame (BGR)

        Returns:
            Base64 encoded JPEG string
        """
        # Resize for efficiency
        small_frame = cv2.resize(frame, (320, 240))

        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])

        # Convert to base64
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return base64_image

    def describe_scene_with_vision(self, frame: np.ndarray) -> str:
        """
        Use vision model (Moondream) to describe the scene.

        Args:
            frame: Input frame

        Returns:
            Scene description
        """
        try:
            # Check if vision model is available
            if self.vision_model not in str(self.available_models):
                logging.warning(f"Vision model {self.vision_model} not available")
                return ""

            # Encode frame
            image_base64 = self._encode_image(frame)

            # Query vision model
            response = ollama.generate(
                model=self.vision_model,
                prompt="Describe this scene for navigation purposes. Focus on obstacles, paths, and safety. Be concise.",
                images=[image_base64],
                options={
                    'temperature': 0.5,
                    'num_predict': 100
                }
            )

            description = response.get('response', '').strip()
            logging.debug(f"Vision description: {description}")

            return description

        except Exception as e:
            logging.error(f"Vision model error: {e}")
            return ""

    def generate_narrative(self, detections: List[Dict],
                          frame: Optional[np.ndarray] = None,
                          navigation_summary: Optional[Dict] = None) -> str:
        """
        Generate navigation narrative from detections.

        Args:
            detections: List of detections with depth
            frame: Optional frame for vision model
            navigation_summary: Optional summary from detection pipeline

        Returns:
            Narrative guidance text
        """
        # Build context from detections
        context = self._build_context(detections, navigation_summary)

        # Get vision description if frame is provided
        vision_context = ""
        if frame is not None:
            vision_context = self.describe_scene_with_vision(frame)

        # Generate narrative using primary LLM
        narrative = self._generate_with_llm(context, vision_context)

        # Update context history
        self._update_context_history(detections)

        return narrative

    def _build_context(self, detections: List[Dict], summary: Optional[Dict] = None) -> str:
        """
        Build text context from detections and summary.

        Args:
            detections: List of detections
            summary: Navigation summary

        Returns:
            Context string
        """
        context_parts = []

        # Add summary info
        if summary:
            total = summary.get('total_objects', 0)
            danger = len(summary.get('danger_objects', []))
            caution = len(summary.get('caution_objects', []))

            context_parts.append(f"Detected {total} objects.")

            if danger > 0:
                context_parts.append(f"{danger} immediate obstacles.")

            if caution > 0:
                context_parts.append(f"{caution} objects requiring caution.")

            closest = summary.get('closest_object')
            if closest:
                label = closest.get('label', 'object')
                depth = closest.get('depth', 0)
                context_parts.append(f"Closest: {label} at {depth:.1f}m.")

        # Add detection details (top 3)
        for i, det in enumerate(detections[:3]):
            label = det.get('label', 'object')
            depth = det.get('depth', 0)
            conf = det.get('confidence', 0)

            # Determine position (left, center, right)
            center = det.get('center', [0, 0])
            # Assuming frame width ~640, center ~320
            if center[0] < 213:
                position = "left"
            elif center[0] > 427:
                position = "right"
            else:
                position = "ahead"

            context_parts.append(f"{label} {position} at {depth:.1f}m")

        context = " ".join(context_parts)
        return context

    def _generate_with_llm(self, context: str, vision_context: str = "") -> str:
        """
        Generate narrative using primary LLM.

        Args:
            context: Detection context
            vision_context: Vision model description

        Returns:
            Generated narrative
        """
        try:
            # Build prompt
            prompt = self._build_prompt(context, vision_context)

            # Check if model is available
            if self.primary_model not in str(self.available_models):
                logging.warning(f"Primary model {self.primary_model} not available, using fallback")
                return self._fallback_narrative(context)

            # Generate with Ollama
            response = ollama.generate(
                model=self.primary_model,
                prompt=prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens
                }
            )

            narrative = response.get('response', '').strip()

            # Clean up and validate
            narrative = self._clean_narrative(narrative)

            if not narrative:
                return self._fallback_narrative(context)

            return narrative

        except Exception as e:
            logging.error(f"LLM generation error: {e}")
            return self._fallback_narrative(context)

    def _build_prompt(self, context: str, vision_context: str = "") -> str:
        """Build prompt for LLM."""
        base_prompt = """You are a navigation assistant for visually impaired users.
Provide clear, concise, and actionable navigation guidance.

Current situation:
{context}

{vision_info}

Provide brief guidance (1-2 sentences). Focus on safety and next steps.
Guidance:"""

        vision_info = f"\nScene description: {vision_context}" if vision_context else ""

        prompt = base_prompt.format(context=context, vision_info=vision_info)
        return prompt

    def _clean_narrative(self, narrative: str) -> str:
        """Clean and validate narrative output."""
        # Remove common artifacts
        narrative = narrative.replace("Guidance:", "").strip()
        narrative = narrative.replace("**", "").strip()

        # Limit length
        sentences = narrative.split('.')
        if len(sentences) > 2:
            narrative = '. '.join(sentences[:2]) + '.'

        return narrative

    def _fallback_narrative(self, context: str) -> str:
        """
        Generate simple rule-based narrative as fallback.

        Args:
            context: Detection context

        Returns:
            Simple narrative
        """
        if "immediate obstacles" in context:
            return "Caution. Obstacles detected ahead. Please slow down and verify path."
        elif "Closest" in context:
            return "Objects detected nearby. Proceed with caution."
        elif "Detected 0 objects" in context:
            return "Path appears clear. Proceed forward."
        else:
            return "Multiple objects in vicinity. Maintain awareness."

    def _update_context_history(self, detections: List[Dict]):
        """
        Update context history for temporal awareness.

        Args:
            detections: Current detections
        """
        # Store simplified detection summary
        summary = {
            'count': len(detections),
            'labels': [d.get('label', '') for d in detections[:3]],
            'min_depth': min([d.get('depth', 10) for d in detections]) if detections else 10
        }

        self.context_history.append(summary)

        # Maintain max length
        if len(self.context_history) > self.max_context_length:
            self.context_history.pop(0)

    def generate_predictive_guidance(self, detections: List[Dict],
                                    predicted_path: Optional[Dict] = None) -> str:
        """
        Generate predictive guidance based on RL predictions.

        Args:
            detections: Current detections
            predicted_path: Predicted path from RL agent

        Returns:
            Predictive guidance
        """
        if predicted_path is None:
            return ""

        try:
            # Extract prediction info
            action = predicted_path.get('action', 'forward')
            confidence = predicted_path.get('confidence', 0.5)

            if confidence < 0.5:
                return ""

            # Build prompt with prediction
            context = f"Predicted safe action: {action}. Detected objects: {len(detections)}"

            prompt = f"""Based on learned navigation patterns:
{context}

Provide brief guidance on whether to {action}. Be concise (1 sentence).
Guidance:"""

            response = ollama.generate(
                model=self.primary_model,
                prompt=prompt,
                options={'temperature': 0.5, 'num_predict': 50}
            )

            guidance = response.get('response', '').strip()
            return self._clean_narrative(guidance)

        except Exception as e:
            logging.error(f"Predictive guidance error: {e}")
            return ""


class ContextualAssistant:
    """
    High-level contextual assistant combining narrative, audio, and predictions.
    """

    def __init__(self, config):
        """Initialize contextual assistant."""
        self.config = config
        self.narrative_gen = NarrativeGenerator(config)

    def get_guidance(self, detections: List[Dict],
                    frame: Optional[np.ndarray] = None,
                    navigation_summary: Optional[Dict] = None,
                    predicted_path: Optional[Dict] = None) -> Dict[str, str]:
        """
        Get complete guidance package.

        Args:
            detections: Detections
            frame: Optional frame
            navigation_summary: Navigation summary
            predicted_path: RL prediction

        Returns:
            Dict with 'narrative', 'predictive', and 'combined' guidance
        """
        # Generate main narrative
        narrative = self.narrative_gen.generate_narrative(
            detections, frame, navigation_summary
        )

        # Generate predictive guidance
        predictive = ""
        if predicted_path:
            predictive = self.narrative_gen.generate_predictive_guidance(
                detections, predicted_path
            )

        # Combine if both exist
        combined = narrative
        if predictive:
            combined = f"{narrative} {predictive}"

        return {
            'narrative': narrative,
            'predictive': predictive,
            'combined': combined
        }
