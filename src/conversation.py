"""
OrbyGlasses - Conversational Navigation System
Natural language conversation for goal-oriented navigation with voice input/output.
"""

import logging
import ollama
import speech_recognition as sr
from typing import Optional, Dict, List
import time
import json
from datetime import datetime


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
        self.activation_phrase = config.get('conversation.activation_phrase', 'hey glasses')

        # Voice recognition
        if self.voice_input:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            self._calibrate_microphone()

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

    def listen_for_activation(self, timeout: float = 1.0) -> bool:
        """
        Listen for activation phrase.

        Args:
            timeout: Listening timeout in seconds

        Returns:
            True if activation phrase detected
        """
        if not self.voice_input:
            return False

        try:
            with self.microphone as source:
                # Quick listen for activation
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=3)

            # Recognize speech
            text = self.recognizer.recognize_google(audio).lower()
            logging.debug(f"Heard: {text}")

            # Check for activation phrase
            if self.activation_phrase in text:
                logging.info(f"Activation phrase detected: '{text}'")
                return True

        except sr.WaitTimeoutError:
            # No speech detected, normal behavior
            pass
        except Exception as e:
            logging.debug(f"Listening error: {e}")

        return False

    def listen_for_input(self, timeout: float = 5.0, prompt: str = None) -> Optional[str]:
        """
        Listen for user voice input.

        Args:
            timeout: Listening timeout
            prompt: Optional prompt to speak before listening

        Returns:
            Recognized text or None
        """
        if not self.voice_input:
            return None

        if prompt and self.tts:
            self.tts.speak_priority(prompt)

        try:
            with self.microphone as source:
                logging.info("Listening for input...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)

            # Recognize speech
            text = self.recognizer.recognize_google(audio)
            logging.info(f"User said: {text}")
            return text

        except sr.WaitTimeoutError:
            if self.tts:
                self.tts.speak_priority("I didn't hear anything. Say 'hey glasses' to try again.")
            return None
        except sr.UnknownValueError:
            if self.tts:
                self.tts.speak_priority("Sorry, I didn't understand that.")
            return None
        except Exception as e:
            logging.error(f"Voice input error: {e}")
            return None

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

    def handle_conversation_interaction(self, scene_context: Dict = None) -> Optional[str]:
        """
        Handle a complete conversation interaction (listen + respond).

        Args:
            scene_context: Current scene information

        Returns:
            AI response or None
        """
        # Listen for user input
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
