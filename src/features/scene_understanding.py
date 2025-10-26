"""
OrbyGlasses - Enhanced Scene Understanding
Vision Language Model integration for better navigation assistance.
"""

import os
import cv2
import numpy as np
import base64
import requests
import json
import time
from typing import Dict, List, Optional, Tuple
import logging


class VisionLanguageModel:
    """Vision Language Model for enhanced scene understanding - supports Moondream and LLaVA."""

    def __init__(self, config):
        """
        Initialize VLM for scene understanding.

        Args:
            config: Configuration manager
        """
        self.config = config
        self.model_name = config.get('models.llm.vision', 'moondream')
        self.temperature = config.get('models.llm.temperature', 0.6)
        self.max_tokens = config.get('models.llm.max_tokens', 150)
        self.use_moondream = self.model_name.lower() in ['moondream', 'moondream2']

        # Scene analysis settings - OPTIMIZED FOR BLIND USERS
        self.analysis_interval = config.get('models.llm.scene_analysis_interval', 5)
        self.last_analysis_time = 0
        self.scene_cache = {}
        self.scene_history = []

        # Force more frequent updates when danger detected
        self.danger_analysis_interval = 2.0  # Faster updates in dangerous situations

        # Initialize Moondream if selected
        if self.use_moondream:
            self._init_moondream()
        else:
            # Fallback to Ollama-based VLM
            self.ollama_url = "http://localhost:11434/api/generate"
            logging.info(f"VLM initialized: {self.model_name} (Ollama)")

    def _init_moondream(self):
        """Initialize Moondream VLM."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from PIL import Image
            import torch

            logging.info("Loading Moondream2 VLM...")

            # Load Moondream2 with latest revision
            model_id = "vikhyatk/moondream2"
            revision = "2025-06-21"  # Latest release

            self.moondream_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.moondream_tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                revision=revision
            )

            # Move to MPS if available
            if torch.backends.mps.is_available():
                self.moondream_model = self.moondream_model.to("mps")
                logging.info("✓ Moondream2 loaded on MPS (Apple Silicon)")
            elif torch.cuda.is_available():
                self.moondream_model = self.moondream_model.to("cuda")
                logging.info("✓ Moondream2 loaded on CUDA")
            else:
                logging.info("✓ Moondream2 loaded on CPU")

            self.moondream_available = True

        except Exception as e:
            logging.warning(f"Failed to load Moondream: {e}")
            logging.warning("Falling back to Ollama-based VLM")
            self.use_moondream = False
            self.moondream_available = False
            self.ollama_url = "http://localhost:11434/api/generate"
    
    def encode_image(self, frame: np.ndarray) -> str:
        """
        Encode frame to base64 for VLM input with enhanced preprocessing for accuracy.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Base64 encoded image string
        """
        # ACCURACY IMPROVEMENT: Higher resolution for better detail recognition
        # Use 768x768 instead of 512x512 for better object/scene recognition
        h, w = frame.shape[:2]
        target_size = 768  # Increased from 512

        if h > target_size or w > target_size:
            scale = min(target_size/h, target_size/w)
            new_h, new_w = int(h * scale), int(w * scale)
            # Use LANCZOS interpolation for better quality
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # ACCURACY IMPROVEMENT: Enhance contrast and brightness for better VLM perception
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced_frame = cv2.merge([l, a, b])

        # Convert LAB back to BGR, then to RGB
        enhanced_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_LAB2BGR)
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)

        # ACCURACY IMPROVEMENT: Higher JPEG quality (95 instead of 85)
        _, buffer = cv2.imencode('.jpg', rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return img_base64
    
    def analyze_scene(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """
        Analyze scene using Vision Language Model (Moondream or Ollama).

        Args:
            frame: Input frame
            detections: List of detected objects

        Returns:
            Scene analysis dictionary
        """
        current_time = time.time()

        # Check for danger objects - force more frequent analysis if danger present
        has_danger = any(d.get('depth', 10) < 1.5 for d in detections)
        active_interval = self.danger_analysis_interval if has_danger else self.analysis_interval

        # Check if we should analyze this frame
        if current_time - self.last_analysis_time < active_interval:
            return self.scene_cache.get('last_analysis', {})

        try:
            if self.use_moondream and self.moondream_available:
                # Use Moondream VLM
                analysis = self._analyze_with_moondream(frame, detections)
            else:
                # Use Ollama-based VLM
                analysis = self._analyze_with_ollama(frame, detections)

            # Update cache and history
            self.scene_cache['last_analysis'] = analysis
            self.scene_cache['timestamp'] = current_time
            self.last_analysis_time = current_time

            # Add to history (keep last 5 analyses)
            self.scene_history.append({
                'timestamp': current_time,
                'analysis': analysis,
                'detections': len(detections)
            })
            if len(self.scene_history) > 5:
                self.scene_history.pop(0)

            logging.info(f"Scene analysis completed: {analysis.get('scene_type', 'unknown')}")
            return analysis

        except Exception as e:
            logging.error(f"VLM scene analysis error: {e}")
            return self._fallback_scene_analysis(detections)

    def _analyze_with_moondream(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """Analyze scene using Moondream VLM."""
        from PIL import Image
        import torch

        # Convert frame to PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # ACCURACY IMPROVEMENT: Higher resolution for Moondream (768x768)
        pil_image.thumbnail((768, 768), Image.Resampling.LANCZOS)

        # Create detection context
        detection_context = self._create_detection_context(detections)

        # ACCURACY IMPROVEMENT: Multi-step reasoning for better VLM accuracy
        # First pass: Identify what's in the scene
        question1 = f"""Look at this image carefully. Detected objects: {detection_context}

What do you see in this scene? Describe:
1. The environment type (indoor room, outdoor street, hallway, etc.)
2. All visible objects and their locations (left/right/center, near/far)
3. Any potential hazards (stairs, drop-offs, obstacles, traffic, narrow spaces)
4. Clear pathways or open areas

Be specific and detailed."""

        with torch.no_grad():
            scene_analysis = self.moondream_model.query(
                image=pil_image,
                question=question1
            )["answer"]

        # Second pass: Generate navigation guidance based on analysis
        question2 = f"""Based on this scene: {scene_analysis}

You are guiding a BLIND person using FRIENDLY, natural language.

CRITICAL - NO NUMBERS:
- Say "very close", "nearby", "ahead", "in the distance" (NOT "2 meters", "5 feet")
- Say "to your left", "to your right", "in front" (NOT "90 degrees")
- Be calm and conversational

Examples:
- "Stop! There's a chair very close ahead, step to your right"
- "Path is clear ahead, you can continue walking"
- "There's a table nearby on your left"

Maximum 2 sentences. What should they do RIGHT NOW?"""

        with torch.no_grad():
            navigation_guidance = self.moondream_model.query(
                image=pil_image,
                question=question2
            )["answer"]

        # Combine both for structured response
        scene_description = f"{scene_analysis} NAVIGATION: {navigation_guidance}"

        # Parse and structure the response
        analysis = self._parse_scene_analysis(scene_description, detections)
        analysis['vlm_raw_response'] = scene_description
        analysis['vlm_model'] = 'moondream2'

        return analysis

    def _analyze_with_ollama(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """Analyze scene using Ollama-based VLM."""
        # Encode image
        img_base64 = self.encode_image(frame)

        # Create detection context
        detection_context = self._create_detection_context(detections)

        # ACCURACY IMPROVEMENT: Enhanced multi-step prompting for Ollama VLMs
        # First query: Scene understanding
        scene_prompt = f"""Analyze this image carefully for a blind person navigation system.

DETECTED OBJECTS WITH POSITIONS: {detection_context}

Provide detailed analysis:
1. Environment type: Is this indoor (room/hallway/office) or outdoor (street/sidewalk/park)?
2. Spatial layout: Describe object positions using left/center/right and near/far
3. Hazards: Identify ANY potential dangers (stairs, ledges, obstacles blocking path, traffic, narrow passages)
4. Clear areas: Where can they safely walk?
5. Key landmarks: What can help them orient?

Be thorough and precise. This person cannot see."""

        scene_payload = {
            "model": self.model_name,
            "prompt": scene_prompt,
            "images": [img_base64],
            "stream": False,
            "options": {
                "temperature": 0.3,  # Lower temperature for more accurate scene understanding
                "num_predict": 200  # More tokens for detailed analysis
            }
        }

        scene_response = requests.post(self.ollama_url, json=scene_payload, timeout=30)
        scene_response.raise_for_status()
        scene_analysis = scene_response.json().get('response', '')

        # Second query: Navigation guidance based on analysis
        nav_prompt = f"""Based on this scene analysis: {scene_analysis}

DETECTED OBJECTS: {detection_context}

You are guiding a BLIND person. Generate FRIENDLY navigation instructions:

CRITICAL RULES - NO NUMBERS:
- Use words: "very close", "nearby", "ahead", "in the distance" (NEVER "2m", "5 feet", etc.)
- Use words: "to your left", "to your right", "in front" (NEVER "90 degrees", angles)
- Be calm, clear, and conversational
- Examples:
  * "Stop! There's a wall very close ahead, step to your left"
  * "A chair is nearby on your right"
  * "Path is clear, you can walk forward"

Maximum 2 sentences. Be friendly and helpful."""

        nav_payload = {
            "model": self.model_name,
            "prompt": nav_prompt,
            "images": [img_base64],
            "stream": False,
            "options": {
                "temperature": 0.4,  # Slightly higher for natural language
                "num_predict": 100  # Concise navigation instructions
            }
        }

        # Make request to Ollama
        nav_response = requests.post(self.ollama_url, json=nav_payload, timeout=30)
        nav_response.raise_for_status()

        navigation_guidance = nav_response.json().get('response', '')

        # Combine scene analysis with navigation guidance
        scene_description = f"SCENE: {scene_analysis}\n\nNAVIGATION: {navigation_guidance}"

        # Parse and structure the response
        analysis = self._parse_scene_analysis(scene_description, detections)
        analysis['vlm_raw_response'] = scene_description
        analysis['vlm_model'] = self.model_name

        return analysis
    
    def _create_detection_context(self, detections: List[Dict]) -> str:
        """Create enriched context string from detections with spatial information."""
        if not detections:
            return "No objects detected"

        context_parts = []
        for det in detections[:8]:  # Increased from 5 to 8 for more context
            label = det.get('label', 'unknown')
            confidence = det.get('confidence', 0.0)
            depth = det.get('depth', 0.0)

            # ACCURACY IMPROVEMENT: Add spatial position context
            bbox = det.get('bbox', [0, 0, 0, 0])
            center = det.get('center', [(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])

            # Determine spatial position (left/center/right)
            frame_center = 160  # Assuming 320 width
            if center[0] < frame_center - 50:
                position = "left"
            elif center[0] > frame_center + 50:
                position = "right"
            else:
                position = "center"

            # Determine vertical position
            if center[1] < 80:  # Top third
                vertical = "upper"
            elif center[1] > 160:  # Bottom third
                vertical = "lower"
            else:
                vertical = "middle"

            context_parts.append(
                f"{label} at {depth:.1f}m ({position} {vertical}, conf: {confidence:.2f})"
            )

        return "; ".join(context_parts)
    
    def _parse_scene_analysis(self, description: str, detections: List[Dict]) -> Dict:
        """Parse VLM response into structured analysis."""
        analysis = {
            'scene_type': 'unknown',
            'hazards': [],
            'safe_areas': [],
            'landmarks': [],
            'recommendations': [],
            'navigation_guidance': '',
            'confidence': 0.8
        }
        
        # Extract scene type
        description_lower = description.lower()
        if 'indoor' in description_lower or 'room' in description_lower:
            analysis['scene_type'] = 'indoor'
        elif 'outdoor' in description_lower or 'street' in description_lower:
            analysis['scene_type'] = 'outdoor'
        
        # Extract hazards
        hazard_keywords = ['obstacle', 'hazard', 'danger', 'blocked', 'cluttered']
        for keyword in hazard_keywords:
            if keyword in description_lower:
                analysis['hazards'].append(f"Potential {keyword} detected")
        
        # Extract safe areas
        safe_keywords = ['clear', 'open', 'safe', 'path', 'corridor']
        for keyword in safe_keywords:
            if keyword in description_lower:
                analysis['safe_areas'].append(f"Clear {keyword} identified")
        
        # Extract landmarks
        landmark_keywords = ['door', 'window', 'wall', 'corner', 'entrance', 'exit']
        for keyword in landmark_keywords:
            if keyword in description_lower:
                analysis['landmarks'].append(f"{keyword.title()} reference point")
        
        # Set navigation guidance
        analysis['navigation_guidance'] = description[:200] + "..." if len(description) > 200 else description
        
        return analysis
    
    def _fallback_scene_analysis(self, detections: List[Dict]) -> Dict:
        """Fallback scene analysis when VLM is unavailable."""
        analysis = {
            'scene_type': 'unknown',
            'hazards': [],
            'safe_areas': [],
            'landmarks': [],
            'recommendations': [],
            'navigation_guidance': 'Scene analysis unavailable',
            'confidence': 0.3
        }
        
        # Basic analysis from detections
        if detections:
            danger_objects = [d for d in detections if d.get('depth', 10) < 1.5]
            if danger_objects:
                analysis['hazards'].append(f"{len(danger_objects)} objects in danger zone")
            
            safe_objects = [d for d in detections if d.get('depth', 10) > 3.0]
            if safe_objects:
                analysis['safe_areas'].append(f"{len(safe_objects)} objects at safe distance")
        
        return analysis
    
    def get_navigation_guidance(self, frame: np.ndarray, detections: List[Dict]) -> str:
        """
        Get enhanced navigation guidance using VLM.
        
        Args:
            frame: Current frame
            detections: Object detections
            
        Returns:
            Navigation guidance string
        """
        analysis = self.analyze_scene(frame, detections)
        
        # Generate priority-based guidance
        if analysis['hazards']:
            return f"⚠️ {analysis['navigation_guidance']}"
        elif analysis['safe_areas']:
            return f"✅ {analysis['navigation_guidance']}"
        else:
            return analysis['navigation_guidance']
    
    def get_scene_summary(self) -> Dict:
        """Get summary of recent scene analyses."""
        if not self.scene_history:
            return {'message': 'No scene analysis available'}
        
        recent_analyses = self.scene_history[-3:]  # Last 3 analyses
        scene_types = [a['analysis']['scene_type'] for a in recent_analyses]
        most_common_type = max(set(scene_types), key=scene_types.count)
        
        return {
            'scene_type': most_common_type,
            'analysis_count': len(self.scene_history),
            'last_analysis': self.scene_cache.get('timestamp', 0),
            'confidence': np.mean([a['analysis']['confidence'] for a in recent_analyses])
        }


class EnhancedSceneProcessor:
    """Enhanced scene processing with VLM integration."""
    
    def __init__(self, config):
        """Initialize enhanced scene processor."""
        self.config = config
        self.vlm = VisionLanguageModel(config)
        self.scene_context = {}
        
    def process_scene(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """
        Process scene with enhanced understanding.
        
        Args:
            frame: Input frame
            detections: Object detections
            
        Returns:
            Enhanced scene analysis
        """
        # Get VLM analysis
        vlm_analysis = self.vlm.analyze_scene(frame, detections)
        
        # Combine with detection data
        enhanced_analysis = {
            'detections': detections,
            'vlm_analysis': vlm_analysis,
            'navigation_guidance': self.vlm.get_navigation_guidance(frame, detections),
            'scene_summary': self.vlm.get_scene_summary(),
            'timestamp': time.time()
        }
        
        return enhanced_analysis
