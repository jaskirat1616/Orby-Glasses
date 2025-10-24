"""
Depth Anything V2 Depth Estimation
Better depth maps for navigation
"""

import torch
import numpy as np
import cv2
from typing import Optional, Dict, Tuple
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image


class DepthAnythingV2:
    """
    Depth Anything V2 for metric depth estimation.
    Better accuracy than MiDaS for blind navigation.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Depth Anything V2.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Model selection (small/base/large)
        model_size = self.config.get('models', {}).get('depth', {}).get('size', 'small')
        model_name = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"

        # Load model and processor
        print(f"Loading {model_name}...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name)

        # Set device
        self.device = self.config.get('models', {}).get('depth', {}).get('device', 'cpu')
        if self.device == 'mps':
            if torch.backends.mps.is_available():
                self.model = self.model.to('mps')
            else:
                self.device = 'cpu'
        elif self.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.to('cuda')

        self.model.eval()

        # Cache for performance
        self.last_depth = None
        print(f"Depth Anything V2 ready on {self.device}")

    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth from RGB image.

        Args:
            frame: Input image (BGR from OpenCV)

        Returns:
            Depth map (meters, float32)
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # Prepare input
        inputs = self.processor(images=image, return_tensors="pt")

        # Move to device
        if self.device != 'cpu':
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Post-process
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        )

        # Convert to numpy
        depth = prediction.squeeze().cpu().numpy()

        # Normalize to meters (approximate)
        # Depth Anything V2 outputs relative depth, need to scale
        depth = self._normalize_depth(depth)

        self.last_depth = depth
        return depth

    def _normalize_depth(self, depth: np.ndarray) -> np.ndarray:
        """
        Normalize depth to metric scale (meters).

        Args:
            depth: Raw depth from model

        Returns:
            Depth in meters
        """
        # Simple normalization
        # Assume depth range 0-10m for indoor navigation
        depth_min = depth.min()
        depth_max = depth.max()

        # Normalize to 0-10m range
        normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        metric_depth = normalized * 10.0  # Scale to 10m max

        return metric_depth

    def get_depth_at_bbox(self, depth_map: np.ndarray, bbox: list, frame_size: Tuple[int, int]) -> float:
        """
        Get average depth within bounding box.

        Args:
            depth_map: Depth map
            bbox: Bounding box [x1, y1, x2, y2]
            frame_size: Original frame size (width, height)

        Returns:
            Average depth in meters
        """
        x1, y1, x2, y2 = [int(x) for x in bbox]

        # Clamp to image bounds
        h, w = depth_map.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        # Extract region
        if x2 > x1 and y2 > y1:
            region = depth_map[y1:y2, x1:x2]
            if region.size > 0:
                # Use median for robustness
                depth = float(np.median(region))
                return depth

        return 10.0  # Default far distance

    def visualize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Create colored depth visualization.

        Args:
            depth_map: Depth map in meters

        Returns:
            Colored depth image (BGR)
        """
        # Normalize for visualization
        depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        depth_normalized = (depth_normalized * 255).astype(np.uint8)

        # Apply colormap (TURBO is good for depth)
        colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)

        return colored


# Example usage
if __name__ == "__main__":
    # Initialize depth estimator
    estimator = DepthAnythingV2()

    # Test with webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Estimate depth
        depth_map = estimator.estimate_depth(frame)

        # Visualize
        depth_colored = estimator.visualize_depth(depth_map)

        # Show side by side
        combined = np.hstack([frame, depth_colored])
        cv2.imshow('RGB | Depth', combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
