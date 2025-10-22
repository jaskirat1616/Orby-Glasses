#!/usr/bin/env python3
"""
Test to diagnose the voice input issue in OrbyGlasses
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.conversation import ConversationManager, SPEECH_RECOGNITION_AVAILABLE
from src.utils import ConfigManager

def test_conversation_manager():
    print("Testing ConversationManager initialization...")
    
    # Load config
    config = ConfigManager("config/config.yaml")
    print(f"Voice input config setting: {config.get('conversation.voice_input', False)}")
    print(f"Speech recognition available (module level): {SPEECH_RECOGNITION_AVAILABLE}")
    
    # Create conversation manager
    try:
        conv_manager = ConversationManager(config)
        print(f"Conversation manager created successfully")
        print(f"Voice input available: {conv_manager.voice_input}")
        
        if not conv_manager.voice_input:
            print("Voice input is disabled, checking why...")
            if not SPEECH_RECOGNITION_AVAILABLE:
                print("- SpeechRecognition not available")
            else:
                print("- SpeechRecognition is available but voice input is still disabled")
                # This suggests an issue during microphone initialization
        else:
            print("Voice input is enabled - this is good!")
            
    except Exception as e:
        print(f"Error creating ConversationManager: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_conversation_manager()