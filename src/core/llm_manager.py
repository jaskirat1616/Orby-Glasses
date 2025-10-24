"""
LLM Manager for OrbyGlasses
Handles Ollama calls with concurrency control to prevent multiple simultaneous requests
"""
import logging
import time
import ollama
import threading
from typing import Dict, Optional, List
from dataclasses import dataclass


@dataclass
class LLMRequest:
    """Represents an LLM request with all necessary parameters"""
    model: str
    prompt: str
    images: Optional[List[str]] = None
    options: Optional[Dict] = None
    request_id: Optional[str] = None


class LLMManager:
    """
    Manages Ollama calls with concurrency control to prevent multiple simultaneous requests.
    Uses a queue and single worker thread to handle all LLM requests sequentially.
    """
    
    def __init__(self, config):
        """
        Initialize the LLM Manager.
        
        Args:
            config: ConfigManager instance
        """
        self.config = config
        self.primary_model = config.get('models.llm.primary', 'gemma3:4b')
        
        # Concurrency control
        self._lock = threading.Lock()
        self._is_call_in_progress = False
        self._last_call_time = 0
        self._min_call_interval = 2.0  # Minimum time between calls
        
        # Performance tracking
        self._total_requests = 0
        self._total_wait_time = 0.0
        self._total_processing_time = 0.0
        
        logging.info(f"LLM Manager initialized with model: {self.primary_model}")

    def generate(self, model: str, prompt: str, images: Optional[List[str]] = None, 
                 options: Optional[Dict] = None) -> Dict:
        """
        Generate response from LLM with concurrency control.
        
        Args:
            model: Model name to use
            prompt: Input prompt
            images: Optional list of base64-encoded images
            options: Optional generation options
            
        Returns:
            Response dictionary from ollama
        """
        with self._lock:
            start_time = time.time()
            self._total_requests += 1
            
            # Wait for minimum interval if needed
            time_since_last = start_time - self._last_call_time
            if time_since_last < self._min_call_interval:
                wait_time = self._min_call_interval - time_since_last
                time.sleep(wait_time)
                actual_wait = time.time() - start_time
                self._total_wait_time += actual_wait
                if actual_wait > 0.1:  # Log significant waits
                    logging.info(f"Waited {actual_wait:.2f}s for minimum call interval")

            # Check if another call is in progress (shouldn't happen with lock, but safety check)
            while self._is_call_in_progress:
                time.sleep(0.05)  # Wait 50ms before checking again
                
            # Mark as in progress
            self._is_call_in_progress = True
            call_start = time.time()
        
        try:
            # Prepare request options
            if options is None:
                options = {
                    'temperature': self.config.get('models.llm.temperature', 0.7),
                    'num_predict': self.config.get('models.llm.max_tokens', 150)
                }
            
            # Make the actual Ollama call (outside of the critical lock section)
            logging.debug(f"Making Ollama call to model {model}, prompt length: {len(prompt)} chars")
            response = ollama.generate(
                model=model,
                prompt=prompt,
                images=images or [],
                options=options
            )
            
            # Calculate processing time
            processing_time = time.time() - call_start
            self._total_processing_time += processing_time
            
            if processing_time > 2.0:  # Log slow requests
                logging.info(f"Slow Ollama call completed in {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            logging.error(f"Ollama call failed: {e}")
            raise
        finally:
            # Update tracking and reset flag (with lock)
            with self._lock:
                self._is_call_in_progress = False
                self._last_call_time = time.time()
                
            total_call_time = time.time() - start_time
            logging.info(f"Ollama call completed: processing={processing_time:.2f}s, total={total_call_time:.2f}s")
    
    def get_stats(self) -> Dict:
        """Get statistics about LLM usage."""
        with self._lock:
            avg_wait = self._total_wait_time / max(1, self._total_requests)
            avg_processing = self._total_processing_time / max(1, self._total_requests)
            avg_total = (self._total_wait_time + self._total_processing_time) / max(1, self._total_requests)
            
        return {
            'total_requests': self._total_requests,
            'avg_wait_time': avg_wait,
            'avg_processing_time': avg_processing,
            'avg_total_time': avg_total,
            'current_call_in_progress': self._is_call_in_progress
        }
    
    def is_available(self) -> bool:
        """Check if the LLM manager is available for new requests."""
        with self._lock:
            return not self._is_call_in_progress