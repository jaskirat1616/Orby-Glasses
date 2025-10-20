"""
OrbyGlasses - Conversational Navigation System
Natural language conversation for goal-oriented navigation with voice input/output.
"""

import logging
import ollama
from typing import Optional, Dict, List
import time
import json
from datetime import datetime
import threading
import queue

# Try to import speech_recognition
try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    logging.warning("SpeechRecognition not available. Voice input will be disabled.")

# Import social navigation module
try:
    from social_navigation import SocialNavigationAI
    SOCIAL_NAVIGATION_AVAILABLE = True
except ImportError:
    SOCIAL_NAVIGATION_AVAILABLE = False
    logging.warning("SocialNavigation module not available. Social navigation features will be disabled.")


class ConversationManager:
    """
    Manages natural language conversations for goal-oriented navigation.
    Allows users to specify destinations, ask questions, and get contextual help.
    """

    def __init__(self, config, tts_system=None):
        """
        Initialize conversation manager.

        Args:
            config: ConfigManager instance
            tts_system: TTS system for voice output
        """
        self.config = config
        self.tts = tts_system

        # Conversation settings
        self.enabled = config.get('conversation.enabled', True)
        self.model = config.get('conversation.model', 'gemma3:4b')
        self.temperature = config.get('conversation.temperature', 0.7)
        self.max_tokens = config.get('conversation.max_tokens', 200)
        self.voice_input = config.get('conversation.voice_input', True)
        self.activation_phrase = config.get('conversation.activation_phrase', 'hey orby')

        # Voice recognition
        if self.voice_input and SPEECH_RECOGNITION_AVAILABLE:
            try:
                logging.info("Initializing voice input...")
                self.recognizer = sr.Recognizer()
                self.microphone = sr.Microphone()
                self._calibrate_microphone()
                logging.info(f"✓ Voice input initialized successfully (activation: '{self.activation_phrase}')")
            except Exception as e:
                logging.error(f"✗ Failed to initialize voice input: {e}")
                self.voice_input = False
        else:
            if not SPEECH_RECOGNITION_AVAILABLE:
                logging.warning("✗ SpeechRecognition not installed. Install with: pip install SpeechRecognition")
            else:
                logging.info("Voice input is disabled in config")
            self.voice_input = False

        # Conversation state
        self.active = False
        self.current_goal = None
        self.destination = None
        self.conversation_history = []
        self.max_history = 10
        self.context = {
            'current_location': 'unknown',
            'detected_objects': [],
            'obstacles': [],
            'path_clear': True,
            'last_instruction': None
        }
        
        # For non-blocking voice recognition
        self.listening_thread = None
        self.is_listening = False
        self.activation_queue = queue.Queue()
        self.input_queue = queue.Queue()
        self.listening_result = None
        self.listening_error = None

        # Start persistent background listener
        self.stop_listening = False
        if self.voice_input:
            self._start_background_listener()

        # Social Navigation AI
        self.social_navigation = None
        if SOCIAL_NAVIGATION_AVAILABLE:
            try:
                self.social_navigation = SocialNavigationAI()
                logging.info("Social Navigation AI initialized")
                # Set region from config if available
                region = config.get('social_navigation.region', 'us')
                self.social_navigation.set_region(region)
            except Exception as e:
                logging.error(f"Failed to initialize Social Navigation AI: {e}")
        else:
            logging.info("Social Navigation AI is not available")

        logging.info(f"Conversation manager initialized (model: {self.model})")

    def _calibrate_microphone(self):
        """Calibrate microphone for ambient noise."""
        if not self.voice_input:
            return

        try:
            with self.microphone as source:
                logging.info("Calibrating microphone for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logging.info("Microphone calibrated")
        except Exception as e:
            logging.error(f"Failed to calibrate microphone: {e}")
            self.voice_input = False

    def _start_background_listener(self):
        """
        Start a persistent background thread that continuously listens for activation.
        This prevents blocking the main thread and avoids creating new threads repeatedly.
        """
        def background_listen_worker():
            """Continuously listen in background without blocking."""
            logging.info("Starting persistent background voice listener...")

            while not self.stop_listening:
                try:
                    with self.microphone as source:
                        # Very short timeout to avoid blocking
                        audio = self.recognizer.listen(source, timeout=0.5, phrase_time_limit=2)

                    # Recognize speech in background
                    try:
                        text = self.recognizer.recognize_google(audio).lower()

                        # Check for activation phrase
                        if self.activation_phrase in text:
                            logging.info(f"✓ Activation detected: '{text}'")
                            # Clear old results and add new one
                            while not self.activation_queue.empty():
                                try:
                                    self.activation_queue.get_nowait()
                                except:
                                    break
                            self.activation_queue.put(True)
                    except sr.UnknownValueError:
                        # Couldn't understand - ignore
                        pass
                    except sr.RequestError as e:
                        logging.error(f"Speech recognition error: {e}")
                        time.sleep(1)  # Wait before retry

                except sr.WaitTimeoutError:
                    # No speech - normal, continue listening
                    pass
                except Exception as e:
                    logging.debug(f"Background listener error: {e}")
                    time.sleep(0.1)  # Brief pause before retry

            logging.info("Background voice listener stopped")

        # Start daemon thread that dies when main program exits
        self.listening_thread = threading.Thread(target=background_listen_worker, daemon=True)
        self.listening_thread.start()
        logging.info("✓ Background voice listener started")

    def listen_for_activation(self, timeout: float = 1.0) -> bool:
        """
        Start listening for activation phrase in non-blocking mode.

        Args:
            timeout: Listening timeout for the initial call

        Returns:
            True if activation phrase detected in a previous call
        """
        if not self.voice_input:
            return False

        # Check if there's already a result in the queue
        try:
            result = self.activation_queue.get_nowait()
            return result
        except queue.Empty:
            # No result ready, start listening if not already listening
            if not self.is_listening:
                self._start_listening_for_activation(timeout)
            return False

    def check_activation_result(self) -> Optional[bool]:
        """
        Check if there's an activation result available without starting a new listen.
        
        Returns:
            True if activation phrase was detected, False if not, None if no result yet
        """
        try:
            return self.activation_queue.get_nowait()
        except queue.Empty:
            return None

    def _start_listening_for_activation(self, timeout: float = 1.0):
        """
        Start listening for activation phrase in a separate thread.
        """
        if self.is_listening:
            return
            
        self.is_listening = True
        self.listening_error = None
        
        def listen_worker():
            try:
                with self.microphone as source:
                    # Adjust timeout to be very short to prevent blocking main thread
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=3)

                # Recognize speech
                text = self.recognizer.recognize_google(audio).lower()
                logging.info(f"🎤 Heard: '{text}'")

                # Check for activation phrase (also check common misrecognitions)
                activation_variations = [
                    self.activation_phrase,
                    self.activation_phrase.replace("hello", "hallo"),
                    self.activation_phrase.replace("hello", "helo"),
                ]

                activation_detected = False
                for variation in activation_variations:
                    if variation in text:
                        logging.info(f"✓ Activation phrase detected: '{text}'")
                        activation_detected = True
                        break

                # Add result to queue
                self.activation_queue.put(activation_detected)
                
            except sr.WaitTimeoutError:
                # No speech detected, normal behavior
                self.activation_queue.put(False)
            except sr.UnknownValueError:
                logging.debug("Could not understand speech")
                self.activation_queue.put(False)
            except sr.RequestError as e:
                logging.error(f"Speech recognition service error: {e}")
                self.activation_queue.put(False)
            except Exception as e:
                logging.debug(f"Listening error: {e}")
                self.activation_queue.put(False)
            finally:
                self.is_listening = False

        # Start the listening thread
        self.listening_thread = threading.Thread(target=listen_worker, daemon=True)
        self.listening_thread.start()

    def listen_for_input(self, timeout: float = 5.0, prompt: str = None) -> Optional[str]:
        """
        Start listening for user voice input in non-blocking mode.

        Args:
            timeout: Listening timeout
            prompt: Optional prompt to speak before listening

        Returns:
            Recognized text if available from a previous call, None otherwise
        """
        if not self.voice_input:
            return None

        # Check if there's already a result in the input queue
        try:
            result = self.input_queue.get_nowait()
            return result
        except queue.Empty:
            # No result ready, start listening if not already listening
            if not self.is_listening:
                self._start_listening_for_input(timeout, prompt)
            return None

    def check_input_result(self) -> Optional[str]:
        """
        Check if there's an input result available without starting a new listen.
        
        Returns:
            Recognized text if available, None if no result yet
        """
        try:
            return self.input_queue.get_nowait()
        except queue.Empty:
            return None

    def _start_listening_for_input(self, timeout: float = 5.0, prompt: str = None):
        """
        Start listening for user input in a separate thread.
        """
        if self.is_listening:
            return
            
        self.is_listening = True
        self.listening_error = None
        
        def listen_input_worker():
            try:
                if prompt and self.tts:
                    self.tts.speak_priority(prompt)

                with self.microphone as source:
                    logging.info("Listening for input...")
                    audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)

                # Recognize speech
                text = self.recognizer.recognize_google(audio)
                logging.info(f"User said: {text}")
                
                # Add result to queue
                self.input_queue.put(text)
                
            except sr.WaitTimeoutError:
                if self.tts:
                    self.tts.speak_priority("I didn't hear anything. Say 'hey glasses' to try again.")
                self.input_queue.put(None)
            except sr.UnknownValueError:
                if self.tts:
                    self.tts.speak_priority("Sorry, I didn't understand that.")
                self.input_queue.put(None)
            except Exception as e:
                logging.error(f"Voice input error: {e}")
                self.input_queue.put(None)
            finally:
                self.is_listening = False

        # Start the listening thread
        self.listening_thread = threading.Thread(target=listen_input_worker, daemon=True)
        self.listening_thread.start()

    def process_conversation(self, user_input: str, scene_context: Dict = None) -> str:
        """
        Process user conversation using LLM with scene context.

        Args:
            user_input: User's message
            scene_context: Current scene information (detections, obstacles, etc.)

        Returns:
            AI response
        """
        if scene_context:
            self.context.update(scene_context)

        # Check if this is a social navigation query
        social_nav_response = self._handle_social_navigation_query(user_input, scene_context)
        if social_nav_response:
            # Add to conversation history
            self.conversation_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            self.conversation_history.append({
                'role': 'assistant',
                'content': social_nav_response,
                'timestamp': datetime.now().isoformat()
            })
            
            return social_nav_response

        # Add to conversation history
        self.conversation_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })

        # Trim history
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

        # Build context-aware prompt
        system_prompt = self._build_system_prompt()
        messages = self._build_conversation_messages(system_prompt)

        try:
            # Generate response using Ollama
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens
                }
            )

            ai_response = response['message']['content'].strip()

            # Update conversation history
            self.conversation_history.append({
                'role': 'assistant',
                'content': ai_response,
                'timestamp': datetime.now().isoformat()
            })

            # Extract goal/destination if mentioned
            self._extract_goal(user_input, ai_response)

            return ai_response

        except Exception as e:
            logging.error(f"Conversation error: {e}")
            return "I'm having trouble processing that. Could you repeat?"

    def _handle_social_navigation_query(self, user_input: str, scene_context: Dict = None) -> Optional[str]:
        """
        Handle social navigation specific queries.
        
        Args:
            user_input: User's message
            scene_context: Current scene information
            
        Returns:
            Social navigation response if applicable, None otherwise
        """
        if not scene_context or not SOCIAL_NAVIGATION_AVAILABLE or not self.social_navigation:
            return None
            
        # Keywords that indicate social navigation queries
        social_keywords = [
            'crowd', 'crowded', 'people', 'hallway', 'walk', 'navigate', 'move',
            'gap', 'space', 'yield', 'right', 'left', 'sidewalk', 'hall', 
            'pass', 'around', 'through', 'between', 'avoid', 'social'
        ]
        
        user_lower = user_input.lower()
        has_social_keyword = any(keyword in user_lower for keyword in social_keywords)
        
        # Check for specific social navigation requests
        is_social_query = (
            has_social_keyword and (
                'how do I' in user_lower or 
                'how can I' in user_lower or 
                'where do I' in user_lower or 
                'which way' in user_lower or 
                'can I' in user_lower or 
                'should I' in user_lower or 
                'navigate' in user_lower or 
                'through' in user_lower or
                'between' in user_lower
            )
        )
        
        # Also handle direct queries about social conventions
        if not is_social_query:
            is_social_query = any(phrase in user_lower for phrase in [
                'stay to the right',
                'stay to the left',
                'what do i do in a crowd',
                'how to navigate crowds',
                'social norms',
                'people around',
                'space opening'
            ])
        
        if is_social_query and 'detected_objects' in scene_context:
            # Generate social navigation guidance
            detections = scene_context['detected_objects']
            social_guidance = self.social_navigation.get_social_navigation_guidance(detections, user_input)
            return social_guidance
        
        # Check for specific social navigation commands without explicit questions
        if has_social_keyword and ('right' in user_lower or 'left' in user_lower):
            # Even if not phrased as a question, if social navigation is relevant
            if 'detected_objects' in scene_context:
                detections = scene_context['detected_objects']
                social_guidance = self.social_navigation.get_social_navigation_guidance(detections, user_input)
                return social_guidance
        
        return None

    def _build_system_prompt(self) -> str:
        """Build system prompt with current context."""
        obstacles_str = ", ".join([f"{obj['label']} at {obj['depth']:.1f}m"
                                   for obj in self.context.get('obstacles', [])[:3]])

        detected_str = ", ".join([obj['label'] for obj in self.context.get('detected_objects', [])[:5]])

        prompt = f"""You are OrbyGlasses, an AI navigation assistant for blind users. You provide clear, concise, and accurate navigation guidance.

Current Context:
- Detected objects: {detected_str if detected_str else 'none'}
- Obstacles: {obstacles_str if obstacles_str else 'none'}
- Path clear: {'Yes' if self.context.get('path_clear', True) else 'No'}
"""

        if self.current_goal:
            prompt += f"- Current goal: {self.current_goal}\n"
        if self.destination:
            prompt += f"- Destination: {self.destination}\n"

        prompt += """
Instructions:
1. Be concise (1-2 sentences max)
2. Focus on safety and clear directions
3. Use spatial terms: "ahead", "left", "right", "behind"
4. Give exact distances when mentioning obstacles
5. If user asks about destination, help them navigate there
6. If user asks what you see, describe the scene accurately
7. Always prioritize immediate safety over destination

Respond naturally and helpfully."""

        return prompt

    def _build_conversation_messages(self, system_prompt: str) -> List[Dict]:
        """Build message list for LLM."""
        messages = [{'role': 'system', 'content': system_prompt}]

        # Add recent conversation history
        for msg in self.conversation_history[-6:]:  # Last 3 exchanges
            messages.append({
                'role': msg['role'],
                'content': msg['content']
            })

        return messages

    def _extract_goal(self, user_input: str, ai_response: str):
        """Extract navigation goal from conversation."""
        user_lower = user_input.lower()

        # Check for destination keywords
        destination_keywords = ['go to', 'find', 'take me to', 'where is', 'navigate to',
                               'looking for', 'need to get to', 'help me find']

        for keyword in destination_keywords:
            if keyword in user_lower:
                # Extract potential destination
                parts = user_lower.split(keyword)
                if len(parts) > 1:
                    potential_dest = parts[1].strip().split('.')[0].split(',')[0]
                    if len(potential_dest) > 2:
                        self.destination = potential_dest
                        self.current_goal = f"Navigate to {potential_dest}"
                        logging.info(f"Goal set: {self.current_goal}")
                        break

    def update_scene_context(self, detections: List[Dict], navigation_summary: Dict):
        """
        Update conversation context with current scene.

        Args:
            detections: List of detected objects
            navigation_summary: Navigation summary from detection pipeline
        """
        self.context['detected_objects'] = detections[:5]  # Top 5
        self.context['obstacles'] = navigation_summary.get('danger_objects', [])
        self.context['path_clear'] = navigation_summary.get('path_clear', True)

        # Update closest object
        closest = navigation_summary.get('closest_object')
        if closest:
            self.context['closest_object'] = f"{closest['label']} at {closest['depth']:.1f}m"
        
        # Update social navigation context if available
        if SOCIAL_NAVIGATION_AVAILABLE and self.social_navigation:
            social_context = self.social_navigation.update_social_context(detections)
            self.context['social_navigation'] = social_context

    def generate_contextual_guidance(self) -> Optional[str]:
        """
        Generate navigation guidance based on goal and current context.

        Returns:
            Guidance string or None
        """
        if not self.current_goal:
            return None

        # Build guidance request
        guidance_request = f"I'm trying to {self.current_goal}. What should I do based on what you see?"

        return self.process_conversation(guidance_request)

    def clear_goal(self):
        """Clear current navigation goal."""
        self.current_goal = None
        self.destination = None
        logging.info("Navigation goal cleared")

    def get_status(self) -> Dict:
        """Get conversation status."""
        return {
            'active': self.active,
            'goal': self.current_goal,
            'destination': self.destination,
            'voice_enabled': self.voice_input,
            'history_length': len(self.conversation_history)
        }

    def stop(self):
        """Stop the conversation manager and background listener."""
        self.stop_listening = True
        if self.listening_thread and self.listening_thread.is_alive():
            logging.info("Stopping background voice listener...")
            # Give thread time to exit gracefully
            self.listening_thread.join(timeout=1.0)

    def handle_conversation_interaction(self, scene_context: Dict = None) -> Optional[str]:
        """
        Handle a complete conversation interaction (listen + respond).

        Args:
            scene_context: Current scene information

        Returns:
            AI response or None
        """
        # This method is now used differently in the main loop
        # It will be called when input is detected via the non-blocking approach
        # For now, we'll keep this for compatibility but it won't be used in the main loop
        user_input = self.listen_for_input(
            prompt="I'm listening. How can I help?",
            timeout=7.0
        )

        if not user_input:
            return None

        # Process conversation
        response = self.process_conversation(user_input, scene_context)

        # Speak response
        if self.tts and response:
            self.tts.speak_priority(response)

        return response
