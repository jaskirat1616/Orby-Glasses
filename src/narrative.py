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
import time


class NarrativeGenerator:
    """
    Generates contextual navigation narratives using LLMs.
    Uses Moondream for vision understanding and Gemma for narrative generation.
    """

    def __init__(self, config):
        """
        Initialize narrative generator with single model to prevent duplicate calls.

        Args:
            config: ConfigManager instance
        """
        self.config = config

        # Model configuration - use single model for both vision and narrative
        self.primary_model = config.get('models.llm.primary', 'gemma3:4b')
        self.vision_model = self.primary_model  # Use same model to prevent duplicate calls
        self.temperature = config.get('models.llm.temperature', 0.7)
        self.max_tokens = config.get('models.llm.max_tokens', 150)

        # Concurrency control to prevent duplicate model calls
        self._last_call_time = 0
        self._min_call_interval = 2.0  # Reduced from 4s to 2s for better responsiveness
        self._is_call_in_progress = False  # Flag to prevent concurrent calls

        # Check available models
        self.available_models = self._check_available_models()

        logging.info(f"Narrative generator initialized with single model: {self.primary_model}")

        # Context window for recent detections
        self.context_history = []
        self.max_context_length = 5

    def _is_model_available(self, model_name: str) -> bool:
        """Check if a model is available in Ollama."""
        if not self.available_models:
            return False
        # Check exact match or partial match (e.g., "gemma3:4b" matches "gemma3:4b")
        for available_model in self.available_models:
            if model_name in available_model or available_model in model_name:
                return True
        return False

    def _check_available_models(self) -> List[str]:
        """Check which Ollama models are available."""
        try:
            models = ollama.list()
            # Handle both dict and object response types
            if isinstance(models, dict):
                available = [m.get('name', m.get('model', '')) for m in models.get('models', [])]
            else:
                # If it's an object with models attribute
                available = [m.name if hasattr(m, 'name') else m.model for m in getattr(models, 'models', [])]
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
        Use the same model for vision description to prevent duplicate calls.

        Args:
            frame: Input frame

        Returns:
            Scene description
        """
        try:
            # Concurrency control - check if a call is already in progress or too recent
            current_time = time.time()
            wait_start = current_time
            original_call_time = self._last_call_time
            original_in_progress = self._is_call_in_progress
            time_since_last_call = current_time - self._last_call_time
            
            if self._is_call_in_progress or time_since_last_call < self._min_call_interval:
                wait_duration = 0
                # Wait for the previous call to complete or minimum interval to pass
                while (self._is_call_in_progress or 
                       (time.time() - self._last_call_time < self._min_call_interval)):
                    time.sleep(0.05)  # Sleep 50ms before checking again
                    wait_duration = time.time() - wait_start
                    if wait_duration > 10.0:  # Don't wait more than 10 seconds
                        logging.warning("Waited too long for model availability, skipping vision call")
                        return ""
                
                actual_wait = time.time() - wait_start
                if actual_wait > 0.1:  # Log if wait was significant
                    logging.info(f"Waited {actual_wait:.2f}s for vision model availability (prev_in_progress: {original_in_progress}, prev_call: {original_call_time:.2f}s ago, min_interval: {self._min_call_interval}s)")
            
            # Set the in-progress flag
            self._is_call_in_progress = True
            start_time = time.time()
            actual_model_start = time.time()

            # Check if model is available
            if not self._is_model_available(self.primary_model):  # Use primary model instead of vision_model
                logging.warning(f"Primary model {self.primary_model} not available")
                self._is_call_in_progress = False
                return ""

            logging.debug(f"Starting vision description with primary model at {start_time:.2f}...")
            
            # Encode frame
            image_base64 = self._encode_image(frame)
            logging.debug(f"Frame encoded to base64 ({len(image_base64)} chars)")

            # Improved prompt for better scene understanding
            vision_prompt = """Analyze this image for a visually impaired person navigating.

Focus on:
1. Obstacles in the path (people, objects, furniture)
2. Clear pathways or directions to move
3. Potential hazards (stairs, edges, obstacles at foot level)
4. Spatial layout (left, right, ahead)

Provide 1-2 concise sentences. Be specific about distances and directions when possible."""

            # Query the same model to prevent duplicate calls
            response = ollama.generate(
                model=self.primary_model,  # Use same model to prevent duplicate calls
                prompt=vision_prompt,
                images=[image_base64],
                options={
                    'temperature': 0.3,  # Lower temperature for more focused responses
                    'num_predict': 120
                }
            )

            description = response.get('response', '').strip()
            model_time = time.time() - actual_model_start
            total_elapsed = time.time() - start_time

            # Update call tracking
            self._last_call_time = time.time()
            self._is_call_in_progress = False

            logging.info(f"Vision description: model_time={model_time:.2f}s, total_time={total_elapsed:.2f}s, desc='{description[:100]}...'")
            logging.debug(f"Full vision description: {description}")

            return description

        except Exception as e:
            logging.error(f"Vision model error: {e}")
            import traceback
            traceback.print_exc()
            # Make sure to reset the flag even if there's an error
            self._is_call_in_progress = False
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
        logging.debug("=" * 60)
        logging.debug("NARRATIVE GENERATION STARTED")
        start_time = time.time()

        # Build context from detections
        logging.debug(f"Building context from {len(detections)} detections...")
        context = self._build_context(detections, navigation_summary)
        logging.info(f"Context built: {context}")

        # Get vision description if frame is provided
        vision_context = ""
        if frame is not None:
            logging.debug("Vision model available, analyzing scene...")
            vision_context = self.describe_scene_with_vision(frame)
            if vision_context:
                logging.info(f"Vision context: {vision_context[:100]}...")
        else:
            logging.debug("No frame provided, skipping vision analysis")

        # Generate narrative using primary LLM
        logging.debug("Generating narrative with LLM...")
        narrative = self._generate_with_llm(context, vision_context)

        # Update context history
        self._update_context_history(detections)

        elapsed = time.time() - start_time
        logging.info(f"✓ Narrative generated in {elapsed:.2f}s: \"{narrative}\"")
        logging.debug("=" * 60)

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
        Generate narrative using primary LLM with concurrency control.

        Args:
            context: Detection context
            vision_context: Vision model description

        Returns:
            Generated narrative
        """
        try:
            # Concurrency control - check if a call is already in progress or too recent
            current_time = time.time()
            wait_start = current_time
            original_call_time = self._last_call_time
            original_in_progress = self._is_call_in_progress
            time_since_last_call = current_time - self._last_call_time
            
            if self._is_call_in_progress or time_since_last_call < self._min_call_interval:
                wait_duration = 0
                # Wait for the previous call to complete or minimum interval to pass
                while (self._is_call_in_progress or 
                       (time.time() - self._last_call_time < self._min_call_interval)):
                    time.sleep(0.05)  # Sleep 50ms before checking again
                    wait_duration = time.time() - wait_start
                    if wait_duration > 10.0:  # Don't wait more than 10 seconds
                        logging.warning("Waited too long for model availability, using fallback")
                        return self._fallback_narrative(context)
                
                actual_wait = time.time() - wait_start
                if actual_wait > 0.1:  # Log if wait was significant
                    logging.info(f"Waited {actual_wait:.2f}s for LLM model availability (prev_in_progress: {original_in_progress}, prev_call: {original_call_time:.2f}s ago, min_interval: {self._min_call_interval}s)")
            
            # Set the in-progress flag
            self._is_call_in_progress = True
            start_time = time.time()
            actual_model_start = time.time()

            # Build prompt
            prompt = self._build_prompt(context, vision_context)
            logging.debug(f"LLM Prompt:\n{prompt}")

            # Check if model is available
            if not self._is_model_available(self.primary_model):
                logging.warning(f"Primary model {self.primary_model} not available, using fallback")
                self._is_call_in_progress = False
                return self._fallback_narrative(context)

            # Generate with Ollama
            logging.debug(f"Calling Ollama {self.primary_model} at {start_time:.2f}...")

            response = ollama.generate(
                model=self.primary_model,
                prompt=prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                    'top_p': 0.9,  # Nucleus sampling for better quality
                    'top_k': 40     # Limit vocabulary for more focused output
                }
            )

            model_time = time.time() - actual_model_start
            total_time = time.time() - start_time
            narrative = response.get('response', '').strip()

            # Update call tracking
            self._last_call_time = time.time()
            self._is_call_in_progress = False

            logging.info(f"LLM response: model_time={model_time:.2f}s, total_time={total_time:.2f}s, response='{narrative[:100]}...'")
            logging.debug(f"Raw LLM response: {narrative}")

            # Clean up and validate
            narrative = self._clean_narrative(narrative)
            logging.debug(f"Cleaned narrative: {narrative}")

            if not narrative or len(narrative) < 10:
                logging.warning(f"Narrative too short ({len(narrative)} chars), using fallback")
                return self._fallback_narrative(context)

            logging.info(f"✓ LLM narrative generated successfully in {total_time:.2f}s")
            return narrative

        except Exception as e:
            logging.error(f"LLM generation error: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_narrative(context)

    def _build_prompt(self, context: str, vision_context: str = "") -> str:
        """Build prompt for LLM with engineering for better outputs."""

        # Prompt with specific instructions and examples
        if vision_context:
            # Vision-based prompt
            base_prompt = """You are a navigation guide for BLIND users. Give clear, actionable directions.

DETECTION DATA:
{context}

VISUAL ANALYSIS:
{vision_context}

RULES:
1. Use relatable distances: "arm's length" (0.5m), "one step" (0.7m), "two steps" (1.5m), "few steps" (2-3m)
2. Always give direction to move: "keep left", "step right", "continue straight", "bear right"
3. Focus on WHERE to go, not just WHAT is there
4. ONE concise sentence - make it actionable
5. Only warn if truly close (under 1.5m)

EXAMPLES:
- "Chair one step ahead on right, keep left"
- "Person arm's length ahead, step to your left"
- "Path clear, continue straight"
- "Table two steps ahead left, bear right"

YOUR GUIDANCE:"""

            prompt = base_prompt.format(
                context=context,
                vision_context=vision_context
            )
        else:
            # Detection-only prompt
            base_prompt = """You are a navigation guide for BLIND users. Give clear, actionable directions.

DETECTED OBJECTS:
{context}

RULES:
1. Use relatable distances: "arm's length" (0.5m), "one step" (0.7m), "two steps" (1.5m), "few steps" (2-3m)
2. Always give direction to move: "keep left", "step right", "continue straight", "bear right"
3. Focus on WHERE to go, not just WHAT is there
4. ONE concise sentence - make it actionable
5. Only warn if truly close (under 1.5m)

EXAMPLES:
- "Person one step ahead, move to your right"
- "Chair arm's length on left, keep right"
- "Path clear, continue forward"
- "Wall two steps right, stay center"

YOUR GUIDANCE:"""

            prompt = base_prompt.format(context=context)

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
        start_time = time.time()
        logging.debug(f"get_guidance called with {len(detections)} detections, has_frame: {frame is not None}, has_pred_path: {predicted_path is not None}")
        
        # Generate main narrative
        narrative_start = time.time()
        narrative = self.narrative_gen.generate_narrative(
            detections, frame, navigation_summary
        )
        narrative_time = time.time() - narrative_start
        logging.debug(f"Narrative generation completed in {narrative_time:.2f}s")

        # Generate predictive guidance
        predictive = ""
        if predicted_path:
            predictive_start = time.time()
            predictive = self.narrative_gen.generate_predictive_guidance(
                detections, predicted_path
            )
            predictive_time = time.time() - predictive_start
            logging.debug(f"Predictive guidance generated in {predictive_time:.2f}s")

        # Combine if both exist
        combined = narrative
        if predictive:
            combined = f"{narrative} {predictive}"

        total_time = time.time() - start_time
        logging.info(f"get_guidance completed: total_time={total_time:.2f}s, narrative_len={len(narrative)}, predictive_len={len(predictive)}")
        
        return {
            'narrative': narrative,
            'predictive': predictive,
            'combined': combined
        }
