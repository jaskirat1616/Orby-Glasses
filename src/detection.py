"""
OrbyGlasses - Object Detection and Depth Estimation
Implements YOLOv11 for object detection and Depth Pro for depth estimation.
"""

import os
import numpy as np
import cv2
import torch
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import logging


class ObjectDetector:
    """YOLOv11-based object detector optimized for Apple Silicon."""

    def __init__(self, model_path: str = "models/yolo/yolo11n.pt",
                 confidence: float = 0.5,
                 iou_threshold: float = 0.45,
                 device: str = "mps"):
        """
        Initialize YOLO detector.

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

        logging.info(f"Initializing YOLO on device: {self.device}")

        # Load YOLO model
        try:
            if os.path.exists(model_path):
                self.model = YOLO(model_path)
            else:
                logging.warning(f"Model not found at {model_path}, downloading YOLOv11n...")
                self.model = YOLO('yolo11n.pt')

            # Set device
            self.model.to(self.device)
            logging.info("YOLO model loaded successfully")

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
    """Depth estimation using Apple's Depth Pro or MiDaS."""

    def __init__(self, model_path: str = "models/depth/depth_pro.pt",
                 device: str = "mps"):
        """
        Initialize depth estimator.

        Args:
            model_path: Path to depth model
            device: Device to run on
        """
        self.model_path = model_path
        self.device = self._validate_device(device)

        logging.info(f"Initializing Depth Estimator on device: {self.device}")

        # Try to load Depth Pro, fall back to MiDaS if unavailable
        try:
            if os.path.exists(model_path):
                self.model = self._load_depth_pro(model_path)
                self.model_type = "depth_pro"
            else:
                logging.warning("Depth Pro not found, using MiDaS")
                self.model = self._load_midas()
                self.model_type = "midas"

            logging.info(f"Depth estimator loaded: {self.model_type}")

        except Exception as e:
            logging.error(f"Failed to load depth model: {e}")
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

    def _load_depth_pro(self, model_path: str):
        """Load Apple Depth Pro model."""
        # Placeholder for Depth Pro loading
        # In practice, this would load the actual Depth Pro model
        logging.warning("Depth Pro loading is placeholder, using MiDaS instead")
        return self._load_midas()

    def _load_midas(self):
        """Load MiDaS depth estimation model."""
        try:
            # Using MiDaS small for speed on M2
            model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
            model.to(self.device)
            model.eval()

            # Load transform ONCE during initialization
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform

            return model
        except Exception as e:
            logging.error(f"Failed to load MiDaS: {e}")
            return None

    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate depth map from RGB frame.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Depth map (normalized to 0-1) or None on failure
        """
        # Always use MiDaS for proper depth estimation
        if self.model is None:
            return self._fallback_depth(frame)

        try:
            # Convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Prepare input for MiDaS
            if self.model_type == "midas":
                # Use pre-loaded transform (loaded once during init)
                input_batch = self.transform(rgb).to(self.device)

                with torch.no_grad():
                    prediction = self.model(input_batch)
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=rgb.shape[:2],
                        mode="bicubic",
                        align_corners=False,
                    ).squeeze()

                depth_map = prediction.cpu().numpy()

                # Normalize to 0-1
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
            depth_map: Depth map (normalized 0-1, where lower = closer)
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
        # MiDaS gives inverse depth, so lower values = closer
        # Map 0 (very close) -> 0.5m, 1 (far) -> 15m
        depth_meters = 0.5 + (median_depth * 14.5)

        # Invert because MiDaS gives inverse depth
        depth_meters = 15.5 - depth_meters

        return float(np.clip(depth_meters, 0.5, 15.0))


class DetectionPipeline:
    """Combined detection and depth estimation pipeline."""

    def __init__(self, config):
        """
        Initialize detection pipeline.

        Args:
            config: ConfigManager instance
        """
        self.config = config

        # Initialize detector
        self.detector = ObjectDetector(
            model_path=config.get('models.yolo.path', 'models/yolo/yolo11n.pt'),
            confidence=config.get('models.yolo.confidence', 0.5),
            iou_threshold=config.get('models.yolo.iou_threshold', 0.45),
            device=config.get('models.yolo.device', 'mps')
        )

        # Initialize depth estimator
        self.depth_estimator = DepthEstimator(
            model_path=config.get('models.depth.path', 'models/depth/depth_pro.pt'),
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
