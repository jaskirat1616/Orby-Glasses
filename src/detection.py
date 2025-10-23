"""
OrbyGlasses - Object Detection and Depth Estimation
Implements YOLOv12 for object detection and Depth Anything V2 for depth estimation.
"""

import os
import numpy as np
import cv2
import torch
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import logging


class ObjectDetector:
    """Optimized YOLOv12-based object detector for Apple Silicon."""

    def __init__(self, model_path: str = "models/yolo/yolo12n.pt",
                 confidence: float = 0.65,  # Higher confidence for fewer false positives
                 iou_threshold: float = 0.45,
                 device: str = "mps"):
        """
        Initialize optimized YOLOv12 detector.

        Args:
            model_path: Path to YOLO model weights
            confidence: Minimum confidence threshold (optimized for navigation)
            iou_threshold: IoU threshold for NMS
            device: Device to run on ('mps', 'cuda', or 'cpu')
        """
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = self._validate_device(device)

        # Load YOLOv12 model with optimizations
        try:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                self.model = YOLO('yolo12n.pt')

            # Set device and optimize for inference
            self.model.to(self.device)
            
            # Warm up the model for faster inference
            dummy_input = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model(dummy_input, verbose=False)
            
            logging.info("YOLOv12 model loaded and optimized")

        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            raise

        # Priority classes for navigation (optimized for visually impaired users)
        self.priority_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
            'traffic light', 'fire hydrant', 'stop sign', 'bench',
            'chair', 'potted plant', 'door', 'stairs', 'pole', 'fence'
        ]
        
        # Pre-compute class indices for faster filtering
        self.priority_class_ids = set()
        for class_name in self.priority_classes:
            if class_name in self.model.names.values():
                class_id = list(self.model.names.keys())[list(self.model.names.values()).index(class_name)]
                self.priority_class_ids.add(class_id)

    def _validate_device(self, device: str) -> str:
        """Validate and return appropriate device."""
        if device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif device == "cuda" and torch.cuda.is_available():
            return "cuda"
        else:
            logging.warning(f"Device {device} not available, falling back to CPU")
            return "cpu"

    def detect(self, frame: np.ndarray, verbose: bool = False) -> List[Dict]:
        """
        Perform optimized object detection on a frame.

        Args:
            frame: Input frame (BGR format)
            verbose: Enable verbose output

        Returns:
            List of detections, optimized for navigation
        """
        try:
            # Run inference with optimizations
            results = self.model(frame,
                               conf=self.confidence,
                               iou=self.iou_threshold,
                               verbose=False,  # Always False for performance
                               device=self.device,
                               half=True,  # Use half precision for speed
                               agnostic_nms=True)  # Class-agnostic NMS

            detections = []

            # Process results efficiently
            for result in results:
                boxes = result.boxes

                if boxes is None or len(boxes) == 0:
                    continue

                # Convert to numpy arrays for faster processing
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy().astype(int)

                # Process all boxes at once
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = xyxy[i]
                    confidence = float(conf[i])
                    class_id = int(cls[i])

                    # Fast priority check using pre-computed set
                    is_priority = class_id in self.priority_class_ids

                    # Only process high-confidence or priority detections
                    if confidence > 0.7 or is_priority:
                        detection = {
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'label': self.model.names[class_id],
                            'confidence': confidence,
                            'class_id': class_id,
                            'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                            'is_priority': is_priority
                        }
                        detections.append(detection)

            # Sort by priority and confidence (most important first)
            detections.sort(key=lambda x: (not x['is_priority'], -x['confidence']))
            
            # Limit to top 5 detections for performance
            return detections[:5]

        except Exception as e:
            logging.error(f"Detection error: {e}")
            return []

    def get_class_names(self) -> List[str]:
        """Return list of all class names."""
        return list(self.model.names.values())


class DepthEstimator:
    """Optimized depth estimation using Depth Anything V2."""

    def __init__(self, model_path: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 device: str = "mps"):
        """
        Initialize optimized depth estimator.

        Args:
            model_path: Hugging Face model name or path
            device: Device to run on
        """
        self.model_path = model_path
        self.device = self._validate_device(device)

        # Load Depth Anything V2 with optimizations
        try:
            from transformers import pipeline

            # Map device names to transformers format
            device_id = 0 if self.device in ["mps", "cuda"] else -1

            self.model = pipeline(
                task="depth-estimation",
                model=model_path,
                device=device_id,
                torch_dtype="float16" if device_id >= 0 else "float32"  # Use half precision
            )
            self.model_type = "depth_anything_v2"

            # Warm up the model
            dummy_image = np.zeros((320, 320, 3), dtype=np.uint8)
            self._estimate_depth_fast(dummy_image)
            
            logging.info(f"Depth Anything V2 loaded and optimized: {model_path}")

        except Exception as e:
            logging.error(f"Failed to load Depth Anything V2: {e}")
            # Fallback to simple depth estimation
            self.model = None
            self.model_type = "fallback"

    def _validate_device(self, device: str) -> str:
        """Validate device availability."""
        if device == "mps" and torch.backends.mps.is_available():
            return "mps"
        elif device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth map from RGB frame using optimized Depth Anything V2.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Depth map (normalized to 0-1) or None on failure
        """
        if self.model is None:
            return self._fallback_depth(frame)

        try:
            return self._estimate_depth_fast(frame)

        except Exception as e:
            logging.error(f"Depth estimation error: {e}")
            return self._fallback_depth(frame)
    
    def _estimate_depth_fast(self, frame: np.ndarray) -> np.ndarray:
        """Fast depth estimation with optimizations."""
        from PIL import Image

        # Resize frame for faster processing (maintain aspect ratio)
        # Use higher resolution for better accuracy (480 instead of 320)
        max_res = 480
        h, w = frame.shape[:2]
        if h > max_res or w > max_res:
            scale = min(max_res/h, max_res/w)
            new_h, new_w = int(h * scale), int(w * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        
        # Convert BGR to RGB efficiently
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        # Run Depth Anything V2 inference
        result = self.model(pil_image)

        # Extract depth map from result
        depth_map = np.array(result["depth"])

        # Fast normalization
        depth_min, depth_max = depth_map.min(), depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        else:
            depth_map = np.zeros_like(depth_map)

        return depth_map

    def _fallback_depth(self, frame: np.ndarray) -> np.ndarray:
        """Ultra-fast depth estimation using image brightness (objects closer = darker usually)."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Resize to small size for speed, then resize back
        small = cv2.resize(gray, (80, 80))
        # Simple blur
        blurred = cv2.GaussianBlur(small, (5, 5), 0)
        # Resize back to original size
        depth = cv2.resize(blurred, (frame.shape[1], frame.shape[0]))
        # Normalize
        depth = depth.astype(np.float32) / 255.0
        return depth

    def get_depth_at_bbox(self, depth_map: np.ndarray, bbox: List[float]) -> float:
        """
        Get estimated depth from depth map at bbox location with outlier filtering.

        Args:
            depth_map: Depth map (normalized 0-1, where higher values = farther)
            bbox: [x1, y1, x2, y2]

        Returns:
            Estimated depth in meters
        """
        if depth_map is None:
            return 10.0

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
            return 10.0

        # Use center region for more accurate depth (avoid edges)
        h_region, w_region = depth_region.shape
        if h_region > 10 and w_region > 10:
            h_pad = h_region // 4
            w_pad = w_region // 4
            center_region = depth_region[h_pad:-h_pad, w_pad:-w_pad]
            if center_region.size > 0:
                depth_region = center_region

        # Remove outliers using IQR method
        q1 = np.percentile(depth_region, 25)
        q3 = np.percentile(depth_region, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Filter outliers
        filtered = depth_region[(depth_region >= lower_bound) & (depth_region <= upper_bound)]

        # Get median depth (robust to remaining outliers)
        if filtered.size > 0:
            median_depth = np.median(filtered)
        else:
            median_depth = np.median(depth_region)

        # Convert normalized depth (0-1) to meters
        # Improved non-linear calibration for better real-world accuracy
        # Close objects need more precision, far objects less critical

        if median_depth < 0.3:
            # Very close: 0-0.3 maps to 0.3-1.5m (high precision near user)
            depth_meters = 0.3 + (median_depth / 0.3) * 1.2
        elif median_depth < 0.6:
            # Medium: 0.3-0.6 maps to 1.5-3.5m
            depth_meters = 1.5 + ((median_depth - 0.3) / 0.3) * 2.0
        else:
            # Far: 0.6-1.0 maps to 3.5-8.0m
            depth_meters = 3.5 + ((median_depth - 0.6) / 0.4) * 4.5

        return float(np.clip(depth_meters, 0.3, 8.0))


class DetectionPipeline:
    """Combined detection and depth estimation pipeline with safety system and tracking."""

    def __init__(self, config):
        """
        Initialize detection pipeline.

        Args:
            config: ConfigManager instance
        """
        self.config = config

        # Initialize YOLOv12 detector
        self.detector = ObjectDetector(
            model_path=config.get('models.yolo.path', 'models/yolo/yolo12n.pt'),
            confidence=config.get('models.yolo.confidence', 0.5),
            iou_threshold=config.get('models.yolo.iou_threshold', 0.45),
            device=config.get('models.yolo.device', 'mps')
        )

        # Initialize Depth Anything V2 estimator
        self.depth_estimator = DepthEstimator(
            model_path=config.get('models.depth.path', 'depth-anything/Depth-Anything-V2-Small-hf'),
            device=config.get('models.depth.device', 'mps')
        )

        self.min_safe_distance = config.get('safety.min_safe_distance', 1.5)

        # Import and initialize safety system
        try:
            from safety_system import SafetySystem
            camera_height = config.get('camera.height', 480)
            self.safety_system = SafetySystem(focal_length=500, frame_height=camera_height)
            logging.info("Safety system initialized")
        except Exception as e:
            logging.warning(f"Safety system initialization failed: {e}, using basic safety")
            self.safety_system = None

        # Initialize object tracker for temporal consistency
        try:
            from object_tracker import ObjectTracker
            self.object_tracker = ObjectTracker(max_distance=50.0, max_depth_diff=1.0)
            logging.info("Object tracker initialized")
        except Exception as e:
            logging.warning(f"Object tracker initialization failed: {e}")
            self.object_tracker = None

    def process_frame(self, frame: np.ndarray, current_fps: float = 15.0) -> Tuple[List[Dict], Optional[np.ndarray], List[Dict]]:
        """
        Process frame through detection and depth estimation with safety checks.

        Args:
            frame: Input frame
            current_fps: Current system FPS for health monitoring

        Returns:
            Tuple of (detections with calibrated depth, depth_map, safety_warnings)
        """
        # Object detection
        detections = self.detector.detect(frame)

        # Depth estimation
        depth_map = self.depth_estimator.estimate_depth(frame)

        # Add depth to detections
        if depth_map is not None:
            for detection in detections:
                bbox = detection['bbox']
                depth = self.depth_estimator.get_depth_at_bbox(depth_map, bbox)
                detection['depth'] = depth
                detection['is_danger'] = depth < self.min_safe_distance
        else:
            # No depth info
            for detection in detections:
                detection['depth'] = 0.0
                detection['is_danger'] = False

        # Apply object tracking for temporal consistency
        if self.object_tracker:
            try:
                detections = self.object_tracker.update(detections)
                # Use smoothed depth from tracking
                for det in detections:
                    if 'smoothed_depth' in det and det['frames_tracked'] > 2:
                        det['depth'] = det['smoothed_depth']  # More accurate
            except Exception as e:
                logging.error(f"Object tracking error: {e}")

        # Apply safety system calibration and get warnings
        safety_warnings = []
        if self.safety_system:
            try:
                detections, safety_warnings = self.safety_system.process_detections(detections, current_fps)
            except Exception as e:
                logging.error(f"Safety system error: {e}")

        return detections, depth_map, safety_warnings

    def get_navigation_summary(self, detections: List[Dict]) -> Dict:
        """
        Generate navigation summary from detections.

        Args:
            detections: List of detections with depth

        Returns:
            Summary dict with obstacle info
        """
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
