#!/usr/bin/env python3
"""
Simplified OrbyGlasses - Bio-Mimetic Navigation System
A lightweight version without SLAM, 3D mapping, occupancy maps, and point clouds
for faster performance and simpler setup.
"""

import sys
import os
import cv2
import numpy as np
import argparse
import time
from typing import Optional
import queue
import threading
import logging

# Add src to path if running from root
if os.path.exists('src/utils.py'):
    sys.path.insert(0, 'src')

# Import required components only
try:
    from ultralytics import YOLO
    import torch
    from transformers import pipeline
    import ollama
    import base64
    from PIL import Image
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install ultralytics torch transformers ollama Pillow")
    sys.exit(1)

# Try to import audio components, with fallbacks
try:
    import pyttsx3
    import pyaudio
    from pydub import AudioSegment
    from pydub.playback import play
    SPEECH_AVAILABLE = True
except ImportError:
    SPEECH_AVAILABLE = False
    print("Audio libraries not available, using system 'say' command for macOS")


class DepthEstimator:
    """Depth estimation using Depth Anything V2"""
    
    def __init__(self, model_path='depth-anything/Depth-Anything-V2-Small-hf', device='mps'):
        self.model_path = model_path
        self.device = self._validate_device(device)
        
        try:
            from transformers import pipeline
            device_id = 0 if self.device in ["mps", "cuda"] else -1
            
            self.model = pipeline(
                task="depth-estimation",
                model=model_path,
                device=device_id
            )
            print(f"✓ Depth estimator loaded: {model_path}")
            self.model_available = True
        except Exception as e:
            print(f"⚠ Failed to load depth estimator: {e}")
            self.model = None
            self.model_available = False
    
    def _validate_device(self, device):
        if device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def estimate_depth(self, frame):
        """Estimate depth map from RGB frame"""
        if not self.model_available:
            # Return dummy depth map
            return np.ones((frame.shape[0], frame.shape[1]), dtype=np.float32) * 2.0
        
        try:
            from PIL import Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            
            result = self.model(pil_image)
            depth_map = np.array(result["depth"])
            
            # Normalize to 0-1 range
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            
            return depth_map
        except Exception as e:
            print(f"Depth estimation error: {e}")
            return np.ones((frame.shape[0], frame.shape[1]), dtype=np.float32) * 2.0
    
    def get_depth_at_bbox(self, depth_map, bbox):
        """Get estimated depth from depth map at bbox location"""
        if depth_map is None:
            return 2.0  # Default distance
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within bounds
        h, w = depth_map.shape
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))
        
        # Extract depth in bbox region
        depth_region = depth_map[y1:y2, x1:x2]
        
        if depth_region.size == 0:
            return 2.0
        
        # Get median depth in the region (more robust than mean)
        median_depth = np.median(depth_region)
        
        # Convert normalized depth (0-1) to meters
        # Map 0 (very close) -> 0.3m, 1 (far) -> 10m
        depth_meters = 0.3 + (median_depth * 9.7)
        
        return float(np.clip(depth_meters, 0.3, 10.0))


class ConfigManager:
    """Simplified configuration manager"""
    
    def __init__(self, config_dict=None):
        if config_dict is None:
            config_dict = self._default_config()
        self.config = config_dict
    
    def _default_config(self):
        return {
            'camera': {
                'source': 0,  # 0 for built-in webcam
                'width': 320,
                'height': 320,
                'fps': 20
            },
            'models': {
                'yolo': {
                    'path': 'yolo11n.pt',  # Will auto-download if not exists
                    'confidence': 0.65,
                    'iou_threshold': 0.45,
                    'device': self._get_best_device()
                },
                'depth': {
                    'path': 'depth-anything/Depth-Anything-V2-Small-hf',  # Hugging Face model
                    'device': self._get_best_device()
                },
                'llm': {
                    'primary': 'gemma3:4b',
                    'temperature': 0.7,
                    'max_tokens': 150
                }
            },
            'audio': {
                'tts_rate': 180,
                'tts_volume': 0.9,
                'beep_enabled': True
            },
            'safety': {
                'min_safe_distance': 1.5,  # meters
                'danger_distance': 0.4,
                'caution_distance': 1.5,
                'emergency_stop_key': 'q'
            },
            'performance': {
                'audio_update_interval': 3.0,  # seconds between audio updates
                'max_detections': 3,
                'depth_skip_frames': 2  # Calculate depth every Nth frame for performance
            }
        }
    
    def _get_best_device(self):
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def get(self, key_path, default=None):
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


class ObjectDetector:
    """Simplified YOLO object detector"""
    
    def __init__(self, model_path, confidence=0.5, iou_threshold=0.45, device="mps"):
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = self._validate_device(device)
        
        # Load YOLO model (auto-download if needed)
        try:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                print(f"Model not found at {model_path}, downloading YOLOv12n...")
                self.model = YOLO('yolo12n.pt')
            
            self.model.to(self.device)
            print(f"✓ YOLO model loaded on {self.device}")
        except Exception as e:
            print(f"✗ Failed to load YOLO model: {e}")
            raise
    
    def _validate_device(self, device):
        if device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif device == "cuda" and torch.cuda.is_available():
            return "cuda"
        else:
            print(f"Device {device} not available, falling back to CPU")
            return "cpu"
    
    def detect(self, frame):
        """Detect objects in a frame"""
        try:
            results = self.model(frame, conf=self.confidence, iou=self.iou_threshold, device=self.device)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    label = self.model.names[class_id]
                    
                    # Calculate center coordinates
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'label': label,
                        'confidence': confidence,
                        'class_id': class_id,
                        'center': [float(center_x), float(center_y)]
                    }
                    
                    detections.append(detection)
            
            # Sort by confidence, keep top detections
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            return detections[:3]  # Return top 3 detections
            
        except Exception as e:
            print(f"Detection error: {e}")
            return []


class SimpleAudioManager:
    """Simplified audio manager using macOS 'say' command"""
    
    def __init__(self, config):
        self.rate = config.get('audio.tts_rate', 180)
        self.volume = config.get('audio.tts_volume', 0.9)
        self.is_speaking = False
        self.speech_queue = queue.Queue()
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
        print(f"✓ Audio manager initialized with rate {self.rate} WPM")
    
    def _speech_worker(self):
        """Worker thread for speech output"""
        import subprocess
        while True:
            try:
                text = self.speech_queue.get(timeout=1)
                if text:
                    self.is_speaking = True
                    print(f"🎤 Speaking: {text[:50]}...")
                    
                    try:
                        subprocess.run(['say', '-r', str(self.rate), text], check=True)
                    except Exception as e:
                        print(f"Speech error: {e}")
                    finally:
                        self.is_speaking = False
                self.speech_queue.task_done()
            except queue.Empty:
                continue
    
    def speak(self, text, priority=False):
        """Queue text for speech output"""
        if priority:
            # Clear queue for urgent messages
            while not self.speech_queue.empty():
                try:
                    self.speech_queue.get_nowait()
                except queue.Empty:
                    break
        else:
            # Only queue if not already speaking
            if self.is_speaking:
                return
        
        self.speech_queue.put(text)
        print(f"🔊 Queued: {text[:50]}...")


class SimpleNarrativeGenerator:
    """Simplified narrative generation without vision model"""
    
    def __init__(self, config):
        self.primary_model = config.get('models.llm.primary', 'gemma3:4b')
        self.temperature = config.get('models.llm.temperature', 0.7)
        self.max_tokens = config.get('models.llm.max_tokens', 150)
        
        # Test if Ollama is available
        try:
            models = ollama.list()
            available = [m.get('name', m.get('model', '')) for m in models.get('models', [])]
            if self.primary_model in available:
                print(f"✓ Ollama model {self.primary_model} available")
                self.ollama_available = True
            else:
                print(f"⚠ Ollama model {self.primary_model} not available, using fallback")
                self.ollama_available = False
        except:
            print("⚠ Ollama not available, using fallback")
            self.ollama_available = False
    
    def generate_narrative(self, detections, navigation_summary):
        """Generate simple narrative guidance"""
        if not detections:
            return "Path is clear. No obstacles detected."
        
        # Build context from detections
        context_parts = []
        
        # Add summary info
        if navigation_summary:
            total = navigation_summary.get('total_objects', 0)
            danger = len(navigation_summary.get('danger_objects', []))
            caution = len(navigation_summary.get('caution_objects', []))
            
            if danger > 0:
                context_parts.append(f"{danger} immediate obstacles ahead.")
            elif caution > 0:
                context_parts.append(f"{caution} objects requiring caution.")
            else:
                context_parts.append(f"Detecting {total} objects around you.")
        
        # Add top detection details
        if detections:
            closest = navigation_summary.get('closest_object')
            if closest:
                label = closest.get('label', 'object')
                depth = closest.get('depth', 0)
                # Determine direction
                center = closest.get('center', [160, 160])
                if center[0] < 106:
                    direction = "on your left"
                elif center[0] > 213:
                    direction = "on your right"
                else:
                    direction = "straight ahead"
                
                if depth < 0.5:
                    distance = "arm's length away"
                elif depth < 1.0:
                    distance = "one step away"
                else:
                    distance = f"{depth:.1f} meters away"
                
                narrative = f"{label.title()} {distance} {direction}."
                return narrative
        
        return "Multiple objects detected. Proceed with caution."
    
    def get_guidance(self, detections, navigation_summary):
        """Get complete guidance"""
        narrative = self.generate_narrative(detections, navigation_summary)
        return {
            'narrative': narrative,
            'predictive': '',
            'combined': narrative
        }


class SimpleFrameProcessor:
    """Simplified frame processing with detection annotation"""
    
    @staticmethod
    def annotate_detections(frame, detections):
        """Draw bounding boxes and labels on frame"""
        annotated = frame.copy()
        
        for i, det in enumerate(detections):
            bbox = det.get('bbox', [])
            if len(bbox) != 4:
                continue
            
            x1, y1, x2, y2 = map(int, bbox)
            label = det.get('label', 'Unknown')
            confidence = det.get('confidence', 0.0)
            depth = det.get('depth', 0.0)
            
            # Color based on distance
            if depth < 1.5:
                color = (0, 0, 255)  # Red - danger
            elif depth < 3.0:
                color = (0, 165, 255)  # Orange - caution
            else:
                color = (0, 255, 0)  # Green - safe
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background and text
            label_text = f"{label} {confidence:.2f}"
            if depth > 0:
                label_text += f" {depth:.1f}m"
            
            (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated


class SimpleOrbyGlasses:
    """Simplified OrbyGlasses without SLAM, 3D mapping, etc."""
    
    def __init__(self, config_path=None):
        print("=" * 60)
        print("Simplified OrbyGlasses - Bio-Mimetic Navigation System")
        print("=" * 60)
        
        # Use default config if no path provided
        if config_path:
            # This would load from file if implemented, use defaults for now
            self.config = ConfigManager()
        else:
            self.config = ConfigManager()
        
        # Initialize components
        device = self.config.get('models.yolo.device', 'mps')
        print(f"Running on device: {device}")
        
        # Initialize detector
        self.detector = ObjectDetector(
            model_path=self.config.get('models.yolo.path', 'yolo11n.pt'),
            confidence=self.config.get('models.yolo.confidence', 0.5),
            iou_threshold=self.config.get('models.yolo.iou_threshold', 0.45),
            device=device
        )
        
        # Initialize depth estimator
        self.depth_estimator = DepthEstimator(
            model_path=self.config.get('models.depth.path', 'depth-anything/Depth-Anything-V2-Small-hf'),
            device=self.config.get('models.depth.device', 'mps')
        )
        
        # Audio system
        self.audio_manager = SimpleAudioManager(self.config)
        
        # Narrative generator
        self.narrative_generator = SimpleNarrativeGenerator(self.config)
        
        # Initialize camera
        self.camera = None
        self.frame_width = self.config.get('camera.width', 320)
        self.frame_height = self.config.get('camera.height', 320)
        
        # State variables
        self.running = False
        self.frame_count = 0
        self.last_audio_time = 0
        self.last_depth_time = 0
        self.audio_interval = self.config.get('performance.audio_update_interval', 3.0)
        self.depth_skip_frames = self.config.get('performance.depth_skip_frames', 2)
        self.last_depth_map = None  # Cache last depth map for performance
        
        # Safety thresholds
        self.min_safe_distance = self.config.get('safety.min_safe_distance', 1.5)
        self.danger_distance = self.config.get('safety.danger_distance', 0.4)
        self.caution_distance = self.config.get('safety.caution_distance', 1.5)
        
        print("✓ Simplified OrbyGlasses initialized!")
        print(f"  - Camera: {self.frame_width}x{self.frame_height}")
        print(f"  - Audio updates every {self.audio_interval}s")
        print(f"  - Safety distance: {self.min_safe_distance}m")
        print(f"  - Depth estimation: every {self.depth_skip_frames} frames")
        print("=" * 60)
    
    def initialize_camera(self):
        """Initialize camera"""
        camera_source = self.config.get('camera.source', 0)
        print(f"Initializing camera: {camera_source}")
        
        try:
            self.camera = cv2.VideoCapture(camera_source)
            if not self.camera.isOpened():
                print("✗ Failed to open camera")
                return False
            
            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.config.get('camera.fps', 20))
            
            print("✓ Camera initialized successfully")
            return True
        except Exception as e:
            print(f"✗ Camera initialization error: {e}")
            return False
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Object detection
        detections = self.detector.detect(frame)
        
        # Depth estimation - run every Nth frame to save performance
        depth_map = None
        if self.frame_count % (self.depth_skip_frames + 1) == 0:
            depth_map = self.depth_estimator.estimate_depth(frame)
            self.last_depth_map = depth_map
        else:
            depth_map = self.last_depth_map
        
        # Add depth estimation to detections
        if depth_map is not None:
            for detection in detections:
                bbox = detection['bbox']
                depth = self.depth_estimator.get_depth_at_bbox(depth_map, bbox)
                detection['depth'] = depth
                detection['is_danger'] = depth < self.danger_distance
        else:
            # Fallback depth estimates if depth estimation fails
            for detection in detections:
                # Simple depth estimation based on bounding box size
                bbox = detection['bbox']
                bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                # Larger bounding boxes = closer objects (rough estimation)
                # Normalized between 0.5m (large) and 5m (small)
                bbox_ratio = bbox_area / (self.frame_width * self.frame_height)
                # Map bbox_ratio [0,1] to distance [0.5, 5.0]
                depth = max(0.5, 5.0 - (bbox_ratio * 4.5))
                detection['depth'] = depth
                detection['is_danger'] = depth < self.danger_distance
        
        # Generate navigation summary
        summary = self._get_navigation_summary(detections)
        
        # Generate narrative guidance
        guidance = self.narrative_generator.get_guidance(detections, summary)
        
        # Annotate frame
        annotated_frame = SimpleFrameProcessor.annotate_detections(frame, detections)
        
        # Add performance and status info to frame
        danger_objects = [d for d in detections if d.get('depth', 10) < self.danger_distance]
        caution_objects = [d for d in detections if self.danger_distance <= d.get('depth', 10) < self.caution_distance]
        
        # Add status overlay
        overlay = annotated_frame.copy()
        bg_color = (0, 0, 100) if danger_objects else (0, 0, 0)
        cv2.rectangle(overlay, (5, 5), (250, 135), bg_color, -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
        
        # Add text info
        cv2.putText(annotated_frame, f"Danger: {len(danger_objects)} | Caution: {len(caution_objects)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        if detections:
            closest = min(detections, key=lambda x: x.get('depth', 10))
            dist_color = (0, 0, 255) if closest['depth'] < self.danger_distance else \
                        (0, 165, 255) if closest['depth'] < self.caution_distance else (0, 255, 0)
            cv2.putText(annotated_frame, f"Closest: {closest['label']} {closest['depth']:.1f}m", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dist_color, 1)
        
        # Status indicator
        status_text, status_color = self._get_status_text(danger_objects, caution_objects, detections)
        cv2.putText(annotated_frame, status_text, (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Add guidance to bottom of frame
        y_offset = self.frame_height - 30
        text = guidance.get('narrative', 'Processing...')
        # Wrap text if too long
        if len(text) > 50:
            text = text[:47] + "..."
        
        cv2.rectangle(annotated_frame, (0, y_offset - 5), (self.frame_width, self.frame_height), (0, 0, 0), -1)
        cv2.putText(annotated_frame, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame, detections, guidance, summary, self.last_depth_map
    
    def _get_navigation_summary(self, detections):
        """Generate navigation summary from detections"""
        summary = {
            'total_objects': len(detections),
            'danger_objects': [],
            'caution_objects': [],
            'safe_objects': [],
            'closest_object': None,
            'path_clear': True
        }
        
        min_distance = float('inf')
        
        for det in detections:
            depth = det.get('depth', 0.0)
            label = det.get('label', 'unknown')
            
            if depth < self.min_safe_distance:
                summary['danger_objects'].append(det)
                summary['path_clear'] = False
            elif depth < 3.0:
                summary['caution_objects'].append(det)
            else:
                summary['safe_objects'].append(det)
            
            if depth < min_distance and depth > 0:
                min_distance = depth
                summary['closest_object'] = det
        
        return summary
    
    def _get_status_text(self, danger_objects, caution_objects, detections):
        """Get status text and color"""
        if danger_objects:
            return "⚠ DANGER", (0, 0, 255)
        elif caution_objects:
            return "⚠ CAUTION", (0, 165, 255)
        elif detections:
            return "SAFE", (0, 255, 0)
        else:
            return "CLEAR", (0, 255, 0)
    
    def run(self, display=True):
        """Main run loop"""
        if not self.initialize_camera():
            print("Cannot start without camera")
            return
        
        self.running = True
        
        # Welcome message
        self.audio_manager.speak("Simplified OrbyGlasses navigation system activated", priority=True)
        print("Navigation system activated. Press 'q' to quit.")
        
        start_time = time.time()
        frame_times = []
        
        try:
            while self.running:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                frame_start = time.time()
                
                # Process frame
                annotated_frame, detections, guidance, summary, depth_map = self.process_frame(frame)
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                frame_times.append(frame_time)
                if len(frame_times) > 30:  # Keep last 30 frame times
                    frame_times.pop(0)
                
                # Calculate FPS
                if frame_times:
                    fps = 1.0 / np.mean(frame_times)
                else:
                    fps = 0
                
                # Add FPS to frame
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Audio output with timing control
                current_time = time.time()
                time_since_last_audio = current_time - self.last_audio_time
                
                # Check for danger
                danger_objects = [d for d in detections if d.get('depth', 10) < self.danger_distance]
                
                # Use shorter interval for danger situations
                active_interval = 1.0 if danger_objects else self.audio_interval
                
                if time_since_last_audio > active_interval and not self.audio_manager.is_speaking:
                    if danger_objects:
                        closest_danger = min(danger_objects, key=lambda x: x['depth'])
                        depth = closest_danger['depth']
                        
                        # Use relatable distance terms
                        if depth < 0.3:
                            distance_term = "immediately ahead"
                        elif depth < 0.5:
                            distance_term = "arm's length away"
                        else:
                            distance_term = "one step away"
                        
                        # Determine direction
                        center = closest_danger.get('center', [160, 160])
                        if center[0] < 106:
                            direction = "on your left, step right"
                        elif center[0] > 213:
                            direction = "on your right, step left"
                        else:
                            direction = "straight ahead, step aside"
                        
                        msg = f"Warning! {closest_danger['label']} {distance_term} {direction}"
                        print(f"🚨 DANGER ALERT: {msg}")
                        self.audio_manager.speak(msg, priority=True)
                    else:
                        # Normal guidance
                        msg = guidance.get('narrative', 'No objects detected')
                        print(f"🔊 Guidance: {msg}")
                        self.audio_manager.speak(msg, priority=False)
                    
                    self.last_audio_time = current_time
                
                # Display frame
                if display:
                    display_frame = cv2.resize(annotated_frame, (500, 500))
                    cv2.imshow('Simple OrbyGlasses', display_frame)
                    
                    # Show depth map in separate smaller window (only when freshly calculated based on frame count)
                    should_show_depth = (self.frame_count % (self.depth_skip_frames + 1) == 0) and self.last_depth_map is not None
                    if should_show_depth:
                        # Convert depth map to colormap for visualization
                        depth_colored = cv2.applyColorMap(
                            (self.last_depth_map * 255).astype(np.uint8),
                            cv2.COLORMAP_MAGMA
                        )
                        # Resize depth map to smaller size for display
                        depth_display = cv2.resize(depth_colored, (256, 256))
                        cv2.imshow('Depth Map', depth_display)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                emergency_key = self.config.get('safety.emergency_stop_key', 'q')
                
                if key == ord(emergency_key):
                    print("Emergency stop activated")
                    self.audio_manager.speak("Navigation stopped", priority=True)
                    break
                
                self.frame_count += 1
                
                # Show performance stats every 100 frames
                if self.frame_count % 100 == 0:
                    if frame_times:
                        avg_frame_time = np.mean(frame_times) * 1000  # ms
                        print(f"Frame {self.frame_count}: FPS={fps:.1f}, Avg frame time={avg_frame_time:.1f}ms")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nShutting down...")
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        print("Shutdown complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Simple OrbyGlasses - Bio-Mimetic Navigation System')
    parser.add_argument('--no-display', action='store_true', help='Run without video display')
    
    args = parser.parse_args()
    
    # Initialize system
    system = SimpleOrbyGlasses()
    
    # Run system
    system.run(display=not args.no_display)


if __name__ == "__main__":
    main()