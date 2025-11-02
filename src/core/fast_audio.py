"""
Fast Audio for Quick Warnings

Gets audio warnings to you in less than half a second.
Uses your Mac's built-in speech.
"""

import subprocess
import threading
import queue
import time
from typing import Optional
from enum import Enum


class AudioPriority(Enum):
    """Audio message priority levels"""
    EMERGENCY = 0  # Immediate danger, interrupt everything
    DANGER = 1  # Close obstacle, high priority
    WARNING = 2  # Moderate distance obstacle
    INFO = 3  # General information
    LOW = 4  # Non-critical updates


class FastAudioManager:
    """
    Ultra-low latency audio manager for blind navigation.

    Target: <500ms from speak() call to audio output
    Method: Direct macOS 'say' command (bypasses pyttsx3 overhead)
    """

    def __init__(self, rate: int = 220, voice: str = "Samantha"):
        """
        Initialize fast audio manager.

        Args:
            rate: Words per minute (180-300, default 220 for clarity+speed)
            voice: macOS voice name (Samantha is clearest)
        """
        self.rate = rate
        self.voice = voice

        # Priority queue for messages
        self.message_queue = queue.PriorityQueue()

        # Track speaking state
        self.is_speaking = False
        self.current_process: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()

        # Start worker thread
        self.running = True
        self.worker_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self.worker_thread.start()

        # Pre-generate common phrases for instant playback
        self.phrase_cache = {}
        self._cache_common_phrases()

        print(f"✅ FastAudioManager initialized: {rate} WPM, voice={voice}")

    def _cache_common_phrases(self):
        """Pre-generate audio files for common emergency phrases"""
        common = [
            "Stop",
            "Car ahead",
            "Person ahead",
            "Obstacle left",
            "Obstacle right",
            "Clear"
        ]

        # Note: Actual caching would save to temp files
        # For now, we use direct 'say' which is already very fast
        pass

    def speak(self, text: str, priority: AudioPriority = AudioPriority.INFO, interrupt: bool = False):
        """
        Speak text with specified priority.

        Args:
            text: Text to speak
            priority: Message priority (EMERGENCY interrupts, others queue)
            interrupt: Force interrupt current speech
        """
        if not text:
            return

        # Emergency messages interrupt immediately
        if priority == AudioPriority.EMERGENCY or interrupt:
            self._interrupt_current()

        # Add to priority queue
        # Lower priority number = higher priority
        self.message_queue.put((priority.value, time.time(), text))

    def speak_immediate(self, text: str):
        """
        Speak text immediately, bypassing queue (for emergencies).
        Target latency: <200ms
        """
        self._interrupt_current()
        self._speak_now(text)

    def _speak_now(self, text: str):
        """Execute TTS immediately using macOS say command"""
        try:
            with self.lock:
                self.is_speaking = True

                # Use macOS 'say' command with optimized parameters
                # -r: rate (WPM)
                # -v: voice
                # No file output = direct to speakers (faster)
                self.current_process = subprocess.Popen(
                    ['say', '-r', str(self.rate), '-v', self.voice, text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                # Wait for completion
                self.current_process.wait()

                self.is_speaking = False
                self.current_process = None

        except Exception as e:
            print(f"❌ TTS error: {e}")
            self.is_speaking = False
            self.current_process = None

    def _interrupt_current(self):
        """Stop current speech immediately"""
        with self.lock:
            if self.current_process and self.current_process.poll() is None:
                self.current_process.terminate()
                self.current_process.wait(timeout=0.1)
                self.current_process = None

            self.is_speaking = False

        # Clear non-emergency messages from queue
        temp_queue = queue.PriorityQueue()
        while not self.message_queue.empty():
            try:
                priority, timestamp, text = self.message_queue.get_nowait()
                # Keep emergency and danger messages
                if priority <= AudioPriority.DANGER.value:
                    temp_queue.put((priority, timestamp, text))
            except queue.Empty:
                break

        self.message_queue = temp_queue

    def _audio_worker(self):
        """Background thread to process audio queue"""
        while self.running:
            try:
                # Get next message (blocks until available)
                priority, timestamp, text = self.message_queue.get(timeout=0.1)

                # Measure latency
                latency_ms = (time.time() - timestamp) * 1000

                # Speak the message
                self._speak_now(text)

                # Log latency for monitoring
                if latency_ms > 500:
                    print(f"⚠️  Audio latency: {latency_ms:.0f}ms (target <500ms)")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Audio worker error: {e}")

    def clear_queue(self):
        """Clear all queued messages"""
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except queue.Empty:
                break

    def shutdown(self):
        """Clean shutdown of audio manager"""
        self.running = False
        self._interrupt_current()
        self.clear_queue()

        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

        print("✅ FastAudioManager shutdown complete")

    def get_latency_stats(self) -> dict:
        """Get audio latency statistics"""
        return {
            'queue_size': self.message_queue.qsize(),
            'is_speaking': self.is_speaking,
            'target_latency_ms': 500,
            'rate_wpm': self.rate
        }


# Convenience function for emergency alerts
def emergency_alert(audio_manager: FastAudioManager, message: str):
    """
    Send emergency alert with <200ms latency.
    Use for imminent danger (obstacle <1m).
    """
    audio_manager.speak_immediate(f"Stop! {message}")


# Convenience function for danger warnings
def danger_warning(audio_manager: FastAudioManager, message: str):
    """
    Send danger warning with high priority.
    Use for close obstacles (1-2m).
    """
    audio_manager.speak(message, priority=AudioPriority.DANGER)


# Convenience function for regular guidance
def navigation_guidance(audio_manager: FastAudioManager, message: str):
    """
    Send navigation guidance with normal priority.
    Use for general directions and info.
    """
    audio_manager.speak(message, priority=AudioPriority.INFO)
