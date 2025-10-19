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
    """YOLOv12-based object detector optimized for Apple Silicon."""

    def __init__(self, model_path: str = "models/yolo/yolo12n.pt",
                 confidence: float = 0.5,
                 iou_threshold: float = 0.45,
                 device: str = "mps"):
        """
        Initialize YOLOv12 detector.

        Args:
            model_path: Path to YOLO model weights
            confidence: Minimum confidence threshold
            iou_threshold: IoU threshold for NMS
            device: Device to run on ('mps', 'cuda', or 'cpu')
        """
        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = self._validate_device(device)

        logging.info(f"Initializing YOLOv12 on device: {self.device}")

        # Load YOLOv12 model
        try:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                logging.warning(f"Model not found at {model_path}, downloading YOLOv12n...")
                self.model = YOLO('yolo12n.pt')

            # Set device
            self.model.to(self.device)
            logging.info("YOLOv12 model loaded successfully")

        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            raise

        # Priority classes for navigation (pedestrians, vehicles, obstacles)
        self.priority_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck',
            'traffic light', 'fire hydrant', 'stop sign', 'bench',
            'chair', 'potted plant', 'door', 'stairs'
        ]

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
        Perform object detection on a frame.

        Args:
            frame: Input frame (BGR format)
            verbose: Enable verbose output

        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - label: Class name
                - confidence: Detection confidence
                - class_id: Class ID
                - center: [x_center, y_center]
        """
        try:
            # Run inference
            results = self.model(frame,
                               conf=self.confidence,
                               iou=self.iou_threshold,
                               verbose=verbose,
                               device=self.device)

            detections = []

            # Process results
            for result in results:
                boxes = result.boxes

                if boxes is None or len(boxes) == 0:
                    continue

                for box in boxes:
                    # Extract box data
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    # Get class name
                    label = self.model.names[class_id]

                    # Calculate center
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2

                    detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'label': label,
                        'confidence': confidence,
                        'class_id': class_id,
                        'center': [float(center_x), float(center_y)],
                        'is_priority': label in self.priority_classes
                    }

                    detections.append(detection)

            # Sort by priority and confidence
            detections.sort(key=lambda x: (not x['is_priority'], -x['confidence']))

            return detections

        except Exception as e:
            logging.error(f"Detection error: {e}")
            return []

    def get_class_names(self) -> List[str]:
        """Return list of all class names."""
        return list(self.model.names.values())


class DepthEstimator:
    """Depth estimation using Depth Anything V2."""

    def __init__(self, model_path: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 device: str = "mps"):
        """
        Initialize depth estimator.

        Args:
            model_path: Hugging Face model name or path (e.g., "depth-anything/Depth-Anything-V2-Small-hf")
            device: Device to run on
        """
        self.model_path = model_path
        self.device = self._validate_device(device)

        logging.info(f"Initializing Depth Estimator on device: {self.device}")

        # Load Depth Anything V2 via transformers
        try:
            from transformers import pipeline

            # Map device names to transformers format
            device_id = 0 if self.device in ["mps", "cuda"] else -1

            self.model = pipeline(
                task="depth-estimation",
                model=model_path,
                device=device_id
            )
            self.model_type = "depth_anything_v2"

            logging.info(f"Depth Anything V2 loaded successfully: {model_path}")

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
        Estimate depth map from RGB frame using Depth Anything V2.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Depth map (normalized to 0-1) or None on failure
        """
        if self.model is None:
            return self._fallback_depth(frame)

        try:
            # Convert BGR to RGB for Depth Anything V2
            from PIL import Image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            # Run Depth Anything V2 inference
            result = self.model(pil_image)

            # Extract depth map from result
            depth_map = np.array(result["depth"])

            # Normalize to 0-1 range
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)

            return depth_map

        except Exception as e:
            logging.error(f"Depth estimation error: {e}")
            return self._fallback_depth(frame)

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
        Get estimated depth from depth map at bbox location.

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

        # Get median depth in the region (more robust than mean)
        median_depth = np.median(depth_region)

        # Convert normalized depth (0-1) to meters
        # Depth Anything V2 gives relative depth (higher = farther)
        # Map 0 (very close) -> 0.3m, 1 (far) -> 15m
        depth_meters = 0.3 + (median_depth * 14.7)

        return float(np.clip(depth_meters, 0.3, 15.0))


class DetectionPipeline:
    """Combined detection and depth estimation pipeline."""

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

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Dict], Optional[np.ndarray]]:
        """
        Process frame through detection and depth estimation.

        Args:
            frame: Input frame

        Returns:
            Tuple of (detections with depth, depth_map)
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

        return detections, depth_map

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
