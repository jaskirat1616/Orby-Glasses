"""
YOLO-World Object Detection
Open-vocabulary detection for any object
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from ultralytics import YOLO
import cv2


class YOLOWorldDetector:
    """
    YOLO-World detector with open-vocabulary support.
    Can detect any object by text description.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize YOLO-World detector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Load YOLO-World model
        model_path = self.config.get('models', {}).get('yolo', {}).get('path', 'yolov8l-worldv2.pt')
        self.model = YOLO(model_path)

        # Set device
        self.device = self.config.get('models', {}).get('yolo', {}).get('device', 'cpu')
        if self.device == 'mps' and not torch.backends.mps.is_available():
            self.device = 'cpu'

        # Detection parameters
        self.confidence = self.config.get('models', {}).get('yolo', {}).get('confidence', 0.45)
        self.iou_threshold = self.config.get('models', {}).get('yolo', {}).get('iou_threshold', 0.45)

        # Custom classes for blind navigation
        self.set_navigation_classes()

    def set_navigation_classes(self):
        """Set custom classes for navigation."""
        navigation_classes = [
            'person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle',
            'chair', 'table', 'door', 'stairs', 'wall', 'floor',
            'bench', 'tree', 'pole', 'sign', 'traffic light',
            'sidewalk', 'curb', 'obstacle', 'barrier'
        ]
        self.model.set_classes(navigation_classes)

    def detect(self, frame: np.ndarray, custom_classes: Optional[List[str]] = None) -> List[Dict]:
        """
        Detect objects in frame.

        Args:
            frame: Input image (BGR)
            custom_classes: Optional list of custom classes to detect

        Returns:
            List of detections with bbox, label, confidence
        """
        # Set custom classes if provided
        if custom_classes:
            self.model.set_classes(custom_classes)

        # Run detection
        results = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )

        # Parse results
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                label = self.model.names[cls_id]

                # Calculate center point
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)

                detection = {
                    'bbox': bbox.tolist(),
                    'label': label,
                    'confidence': conf,
                    'center': [center_x, center_y],
                    'class_id': cls_id
                }
                detections.append(detection)

        return detections

    def detect_with_prompt(self, frame: np.ndarray, text_prompt: str) -> List[Dict]:
        """
        Detect objects matching text description.

        Args:
            frame: Input image
            text_prompt: Text description (e.g., "red door", "person walking")

        Returns:
            List of detections matching the prompt
        """
        # Set classes from prompt
        self.model.set_classes([text_prompt])

        return self.detect(frame)


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = YOLOWorldDetector()

    # Test with webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detections = detector.detect(frame)

        # Draw detections
        for det in detections:
            bbox = det['bbox']
            label = det['label']
            conf = det['confidence']

            # Draw box
            cv2.rectangle(frame,
                         (int(bbox[0]), int(bbox[1])),
                         (int(bbox[2]), int(bbox[3])),
                         (0, 255, 0), 2)

            # Draw label
            text = f"{label} {conf:.2f}"
            cv2.putText(frame, text,
                       (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display
        cv2.imshow('YOLO-World Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
