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
    """Vision Language Model for enhanced scene understanding."""
    
    def __init__(self, config):
        """
        Initialize VLM for scene understanding.
        
        Args:
            config: Configuration manager
        """
        self.config = config
        self.model_name = config.get('models.llm.vision', 'llava:7b')
        self.temperature = config.get('models.llm.temperature', 0.6)
        self.max_tokens = config.get('models.llm.max_tokens', 150)
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Scene analysis settings
        self.analysis_interval = config.get('models.llm.scene_analysis_interval', 5)
        self.last_analysis_time = 0
        self.scene_cache = {}
        self.scene_history = []
        
        logging.info(f"VLM initialized: {self.model_name}")
    
    def encode_image(self, frame: np.ndarray) -> str:
        """
        Encode frame to base64 for VLM input.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Base64 encoded image string
        """
        # Resize frame for VLM processing (optimize for speed)
        h, w = frame.shape[:2]
        if h > 512 or w > 512:
            scale = min(512/h, 512/w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Encode to JPEG
        _, buffer = cv2.imencode('.jpg', rgb_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return img_base64
    
    def analyze_scene(self, frame: np.ndarray, detections: List[Dict]) -> Dict:
        """
        Analyze scene using Vision Language Model.
        
        Args:
            frame: Input frame
            detections: List of detected objects
            
        Returns:
            Scene analysis dictionary
        """
        current_time = time.time()
        
        # Check if we should analyze this frame
        if current_time - self.last_analysis_time < self.analysis_interval:
            return self.scene_cache.get('last_analysis', {})
        
        try:
            # Encode image
            img_base64 = self.encode_image(frame)
            
            # Create detection context
            detection_context = self._create_detection_context(detections)
            
            # VLM prompt for navigation assistance
            prompt = f"""You are a navigation assistant for visually impaired users. Analyze this scene and provide helpful guidance.

Detected objects: {detection_context}

Please describe:
1. The overall scene (indoor/outdoor, room type, etc.)
2. Navigation hazards or obstacles
3. Safe paths or clear areas
4. Important landmarks or reference points
5. Recommended actions for safe navigation

Keep your response concise and actionable for a visually impaired person."""

            # Prepare request
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [img_base64],
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            # Make request to Ollama
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            scene_description = result.get('response', '')
            
            # Parse and structure the response
            analysis = self._parse_scene_analysis(scene_description, detections)
            
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
    
    def _create_detection_context(self, detections: List[Dict]) -> str:
        """Create context string from detections."""
        if not detections:
            return "No objects detected"
        
        context_parts = []
        for det in detections[:5]:  # Limit to top 5 detections
            label = det.get('label', 'unknown')
            confidence = det.get('confidence', 0.0)
            depth = det.get('depth', 0.0)
            context_parts.append(f"{label} (confidence: {confidence:.2f}, distance: {depth:.1f}m)")
        
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
