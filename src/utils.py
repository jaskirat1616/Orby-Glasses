"""
OrbyGlasses - Utility Functions
Provides helper functions for audio, logging, configuration, and data management.
"""

import os
import yaml
import logging
import colorlog
import json
import numpy as np
import cv2
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import pyttsx3
import pyaudio
from pydub import AudioSegment
from pydub.playback import play
import threading
import queue


class ConfigManager:
    """Manages configuration loading and validation."""

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logging.error(f"Config file not found: {self.config_path}")
            return self._default_config()
        except yaml.YAMLError as e:
            logging.error(f"Error parsing config: {e}")
            return self._default_config()

    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'camera': {'source': 0, 'width': 640, 'height': 480, 'fps': 30},
            'models': {
                'yolo': {'path': 'models/yolo/yolo11n.pt', 'confidence': 0.5, 'device': 'mps'},
                'depth': {'path': 'models/depth/depth_pro.pt', 'device': 'mps'},
                'llm': {'primary': 'gemma2:2b', 'vision': 'moondream', 'temperature': 0.7}
            },
            'audio': {'tts_rate': 175, 'tts_volume': 1.0, 'echolocation_enabled': True},
            'safety': {'min_safe_distance': 1.5, 'emergency_stop_key': 'q'}
        }

    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'camera.width')."""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value


class Logger:
    """Custom logger with color output and file logging."""

    def __init__(self, name: str = "OrbyGlasses", log_file: str = "data/logs/orbyglass.log"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Create logs directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_formatter)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)

        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def info(self, message: str):
        self.logger.info(message)

    def warning(self, message: str):
        self.logger.warning(message)

    def error(self, message: str):
        self.logger.error(message)

    def debug(self, message: str):
        self.logger.debug(message)

    def critical(self, message: str):
        self.logger.critical(message)


class AudioManager:
    """Manages text-to-speech and audio playback."""

    def __init__(self, config: ConfigManager):
        self.config = config
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', config.get('audio.tts_rate', 175))
        self.tts_engine.setProperty('volume', config.get('audio.tts_volume', 1.0))

        # Thread-safe queue for audio messages
        self.tts_queue = queue.Queue()
        self.is_speaking = False
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

    def _tts_worker(self):
        """Worker thread for TTS to avoid blocking."""
        while True:
            try:
                text = self.tts_queue.get(timeout=1)
                if text:
                    self.is_speaking = True
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    self.is_speaking = False
                self.tts_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"TTS error: {e}")
                self.is_speaking = False

    def speak(self, text: str, priority: bool = False):
        """
        Convert text to speech.

        Args:
            text: Text to speak
            priority: If True, clear queue and speak immediately
        """
        if priority:
            # Clear queue for urgent messages
            while not self.tts_queue.empty():
                try:
                    self.tts_queue.get_nowait()
                except queue.Empty:
                    break

        self.tts_queue.put(text)

    def play_sound(self, audio_data: np.ndarray, sample_rate: int = 16000):
        """
        Play audio from numpy array.

        Args:
            audio_data: Audio samples as numpy array
            sample_rate: Sample rate in Hz
        """
        try:
            # Normalize audio
            audio_data = np.clip(audio_data, -1, 1)
            audio_data = (audio_data * 32767).astype(np.int16)

            # Play using PyAudio
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16,
                          channels=2 if len(audio_data.shape) > 1 else 1,
                          rate=sample_rate,
                          output=True)
            stream.write(audio_data.tobytes())
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            logging.error(f"Error playing sound: {e}")

    def stop(self):
        """Stop all audio playback."""
        self.tts_engine.stop()


class FrameProcessor:
    """Utilities for frame processing and visualization."""

    @staticmethod
    def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Resize frame to specified dimensions."""
        return cv2.resize(frame, (width, height))

    @staticmethod
    def annotate_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.

        Args:
            frame: Input frame
            detections: List of detection dicts with 'bbox', 'label', 'confidence', 'depth'

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for det in detections:
            bbox = det.get('bbox', [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            label = det.get('label', 'Unknown')
            confidence = det.get('confidence', 0.0)
            depth = det.get('depth', 0.0)

            # Color based on distance (red = close, green = far)
            if depth < 1.5:
                color = (0, 0, 255)  # Red - danger
            elif depth < 3.0:
                color = (0, 165, 255)  # Orange - caution
            else:
                color = (0, 255, 0)  # Green - safe

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label_text = f"{label} {confidence:.2f} | {depth:.1f}m"
            (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

            # Draw label text
            cv2.putText(annotated, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated

    @staticmethod
    def encode_frame_base64(frame: np.ndarray) -> str:
        """Encode frame to base64 for LLM input."""
        import base64
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')


class DataLogger:
    """Logs detection data and user habits for RL training."""

    def __init__(self, log_dir: str = "data/logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = os.path.join(log_dir, f"session_{self.session_id}.jsonl")

    def log_detection(self, frame_id: int, detections: List[Dict],
                     depth_map: Optional[np.ndarray] = None,
                     action: Optional[str] = None):
        """
        Log detection data for a frame.

        Args:
            frame_id: Frame number
            detections: List of detections
            depth_map: Optional depth map
            action: User action taken
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'frame_id': frame_id,
            'detections': detections,
            'action': action,
            'depth_stats': self._compute_depth_stats(depth_map) if depth_map is not None else None
        }

        with open(self.session_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def _compute_depth_stats(self, depth_map: np.ndarray) -> Dict:
        """Compute statistics from depth map."""
        return {
            'mean': float(np.mean(depth_map)),
            'min': float(np.min(depth_map)),
            'max': float(np.max(depth_map)),
            'std': float(np.std(depth_map))
        }

    def load_session_data(self) -> List[Dict]:
        """Load all data from current session."""
        data = []
        if os.path.exists(self.session_file):
            with open(self.session_file, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        return data


class PerformanceMonitor:
    """Monitor system performance and latency."""

    def __init__(self):
        self.timings = {}
        self.frame_times = []

    def start_timer(self, name: str):
        """Start a named timer."""
        import time
        self.timings[name] = time.perf_counter()

    def stop_timer(self, name: str) -> float:
        """Stop a named timer and return elapsed time in ms."""
        import time
        if name in self.timings:
            elapsed = (time.perf_counter() - self.timings[name]) * 1000
            return elapsed
        return 0.0

    def log_frame_time(self, frame_time: float):
        """Log frame processing time."""
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)

    def get_avg_fps(self) -> float:
        """Get average FPS over recent frames."""
        if not self.frame_times:
            return 0.0
        avg_time = np.mean(self.frame_times)
        return 1000.0 / avg_time if avg_time > 0 else 0.0

    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.frame_times:
            return {}
        return {
            'avg_frame_time_ms': np.mean(self.frame_times),
            'min_frame_time_ms': np.min(self.frame_times),
            'max_frame_time_ms': np.max(self.frame_times),
            'fps': self.get_avg_fps()
        }


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        'src',
        'models/yolo',
        'models/depth',
        'models/rl',
        'data/test_videos',
        'data/logs',
        'config',
        'tests'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def check_device() -> str:
    """Check and return the best available device for PyTorch."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"
