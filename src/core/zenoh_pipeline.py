"""
Zenoh-based Pipeline for OrbyGlasses
High-performance pub-sub architecture for real-time navigation
"""

import zenoh
import json
import numpy as np
import cv2
import time
from typing import Dict, Any, Optional, Callable
import threading


class ZenohPipeline:
    """
    Zenoh-based communication pipeline.
    Much faster than Python threading for robotics applications.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Zenoh pipeline.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize Zenoh session
        zenoh_config = zenoh.Config()
        self.session = zenoh.open(zenoh_config)

        # Topic names
        self.topics = {
            'camera': 'orby/camera/frame',
            'depth': 'orby/depth/map',
            'detections': 'orby/vision/detections',
            'slam': 'orby/slam/pose',
            'audio': 'orby/audio/guidance',
            'haptic': 'orby/haptic/pattern'
        }

        # Subscribers
        self.subscribers = {}

        # Latest data cache
        self.latest_data = {
            'frame': None,
            'depth': None,
            'detections': None,
            'slam': None
        }

        print("✓ Zenoh pipeline initialized")

    def publish_frame(self, frame: np.ndarray):
        """
        Publish camera frame.

        Args:
            frame: Camera frame (numpy array)
        """
        # Encode frame to JPEG for efficient transmission
        _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self.session.put(self.topics['camera'], encoded.tobytes())

    def publish_depth(self, depth_map: np.ndarray):
        """
        Publish depth map.

        Args:
            depth_map: Depth map (numpy array)
        """
        # Compress depth map
        depth_bytes = depth_map.astype(np.float16).tobytes()
        shape = json.dumps(list(depth_map.shape))

        # Publish with metadata
        data = {
            'shape': shape,
            'data': depth_bytes.hex()
        }
        self.session.put(self.topics['depth'], json.dumps(data))

    def publish_detections(self, detections: list):
        """
        Publish object detections.

        Args:
            detections: List of detection dictionaries
        """
        # Convert numpy arrays to lists for JSON
        serializable = []
        for det in detections:
            det_copy = det.copy()
            if 'bbox' in det_copy and isinstance(det_copy['bbox'], np.ndarray):
                det_copy['bbox'] = det_copy['bbox'].tolist()
            if 'center' in det_copy and isinstance(det_copy['center'], np.ndarray):
                det_copy['center'] = det_copy['center'].tolist()
            serializable.append(det_copy)

        self.session.put(self.topics['detections'], json.dumps(serializable))

    def publish_slam(self, slam_result: Dict):
        """
        Publish SLAM result.

        Args:
            slam_result: SLAM result dictionary
        """
        # Convert numpy arrays to lists
        result_copy = slam_result.copy()
        if 'position' in result_copy:
            result_copy['position'] = result_copy['position'].tolist()
        if 'pose' in result_copy and isinstance(result_copy['pose'], np.ndarray):
            result_copy['pose'] = result_copy['pose'].tolist()

        self.session.put(self.topics['slam'], json.dumps(result_copy))

    def publish_audio_guidance(self, message: str):
        """
        Publish audio guidance message.

        Args:
            message: Audio message to speak
        """
        data = {
            'message': message,
            'timestamp': time.time()
        }
        self.session.put(self.topics['audio'], json.dumps(data))

    def publish_haptic_pattern(self, pattern: np.ndarray):
        """
        Publish haptic pattern.

        Args:
            pattern: Haptic pattern (num_motors x 2)
        """
        data = {
            'pattern': pattern.tolist(),
            'timestamp': time.time()
        }
        self.session.put(self.topics['haptic'], json.dumps(data))

    def subscribe_detections(self, callback: Callable):
        """
        Subscribe to object detections.

        Args:
            callback: Function to call with detections
        """
        def handler(sample):
            data = json.loads(sample.payload.decode('utf-8'))
            self.latest_data['detections'] = data
            callback(data)

        sub = self.session.declare_subscriber(self.topics['detections'], handler)
        self.subscribers['detections'] = sub

    def subscribe_depth(self, callback: Callable):
        """
        Subscribe to depth maps.

        Args:
            callback: Function to call with depth map
        """
        def handler(sample):
            data = json.loads(sample.payload.decode('utf-8'))
            shape = json.loads(data['shape'])
            depth_bytes = bytes.fromhex(data['data'])
            depth_map = np.frombuffer(depth_bytes, dtype=np.float16).reshape(shape)
            self.latest_data['depth'] = depth_map
            callback(depth_map)

        sub = self.session.declare_subscriber(self.topics['depth'], handler)
        self.subscribers['depth'] = sub

    def subscribe_slam(self, callback: Callable):
        """
        Subscribe to SLAM results.

        Args:
            callback: Function to call with SLAM result
        """
        def handler(sample):
            data = json.loads(sample.payload.decode('utf-8'))
            # Convert lists back to numpy arrays
            if 'position' in data:
                data['position'] = np.array(data['position'])
            if 'pose' in data:
                data['pose'] = np.array(data['pose'])
            self.latest_data['slam'] = data
            callback(data)

        sub = self.session.declare_subscriber(self.topics['slam'], handler)
        self.subscribers['slam'] = sub

    def get_latest(self, topic: str) -> Any:
        """
        Get latest data from a topic.

        Args:
            topic: Topic name ('frame', 'depth', 'detections', 'slam')

        Returns:
            Latest data or None
        """
        return self.latest_data.get(topic)

    def close(self):
        """Close Zenoh session."""
        for sub in self.subscribers.values():
            sub.undeclare()
        self.session.close()
        print("✓ Zenoh pipeline closed")


# Example: Separate processing nodes

class VisionNode:
    """Vision processing node (detection + depth)."""

    def __init__(self, pipeline: ZenohPipeline):
        self.pipeline = pipeline
        self.running = False

    def run(self):
        """Run vision processing loop."""
        self.running = True

        # Would subscribe to camera frames and process them
        # Then publish detections and depth

        print("Vision node running...")

        while self.running:
            # Process frames
            time.sleep(0.03)  # ~30 FPS

    def stop(self):
        self.running = False


class SLAMNode:
    """SLAM processing node."""

    def __init__(self, pipeline: ZenohPipeline):
        self.pipeline = pipeline
        self.running = False

    def run(self):
        """Run SLAM processing loop."""
        self.running = True

        print("SLAM node running...")

        while self.running:
            # Process SLAM
            time.sleep(0.03)  # ~30 FPS

    def stop(self):
        self.running = False


class AudioNode:
    """Audio output node."""

    def __init__(self, pipeline: ZenohPipeline):
        self.pipeline = pipeline
        self.running = False

        # Subscribe to audio guidance
        self.pipeline.subscribe_detections(self.on_detections)

    def on_detections(self, detections):
        """Handle new detections."""
        if detections:
            # Generate audio guidance
            message = f"{len(detections)} objects detected"
            print(f"Audio: {message}")

    def run(self):
        """Run audio node."""
        self.running = True
        print("Audio node running...")

        while self.running:
            time.sleep(0.1)

    def stop(self):
        self.running = False


# Example usage
if __name__ == "__main__":
    # Create pipeline
    pipeline = ZenohPipeline()

    # Create nodes
    vision_node = VisionNode(pipeline)
    slam_node = SLAMNode(pipeline)
    audio_node = AudioNode(pipeline)

    # Start nodes in separate threads
    vision_thread = threading.Thread(target=vision_node.run)
    slam_thread = threading.Thread(target=slam_node.run)
    audio_thread = threading.Thread(target=audio_node.run)

    vision_thread.start()
    slam_thread.start()
    audio_thread.start()

    try:
        # Test publishing
        for i in range(10):
            # Publish test data
            pipeline.publish_detections([
                {'label': 'person', 'confidence': 0.9, 'bbox': [100, 100, 200, 200]}
            ])
            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping...")

    # Stop nodes
    vision_node.stop()
    slam_node.stop()
    audio_node.stop()

    vision_thread.join()
    slam_thread.join()
    audio_thread.join()

    pipeline.close()

    print("Done!")
