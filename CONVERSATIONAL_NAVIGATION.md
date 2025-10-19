# 💬 Conversational Navigation Feature

OrbyGlasses now includes an advanced conversational navigation system that allows you to interact naturally with the AI assistant using voice commands.

## Features

### 🎤 Voice-Activated Conversations
- **Wake Phrase**: Say "**hey glasses**" to activate the conversation mode
- **Natural Language**: Talk to OrbyGlasses like you would to a friend
- **Context-Aware**: The AI understands your current surroundings and navigation state

### 🎯 Goal-Oriented Navigation
Tell OrbyGlasses where you want to go:
- "Take me to the kitchen"
- "Find the bathroom"
- "Help me get to the door"
- "Where is the nearest chair?"

### 🧠 Intelligent Understanding
The system can handle various types of requests:

**Navigation Requests:**
- "I want to go to the bedroom"
- "Navigate me to the front door"
- "Help me find a chair"

**Scene Understanding:**
- "What do you see?"
- "Is the path clear?"
- "What's in front of me?"
- "Are there any obstacles?"

**Contextual Questions:**
- "How far is the table?"
- "Which way should I go?"
- "Is it safe to walk forward?"

### 📍 Persistent Goal Tracking
- Set a destination and OrbyGlasses will guide you there
- Get continuous updates based on your environment
- Clear navigation instructions with spatial awareness

## How to Use

### 1. Enable Conversational Mode
The feature is enabled by default. Check your `config/config.yaml`:

```yaml
conversation:
  enabled: true
  voice_input: true
  activation_phrase: "hey glasses"
```

### 2. Install Required Dependencies
```bash
source venv/bin/activate
pip install -r requirements.txt
```

This installs `SpeechRecognition` for voice input.

### 3. Start OrbyGlasses
```bash
python src/main.py
```

You'll see: `💬 Conversational mode: Say 'hey glasses' to start`

### 4. Interact with OrbyGlasses

**Step 1:** Say the wake phrase
```
You: "Hey glasses"
OrbyGlasses: "I'm listening. How can I help?"
```

**Step 2:** Tell it what you need
```
You: "Take me to the kitchen"
OrbyGlasses: "I'll help you navigate to the kitchen. Based on what I see..."
```

**Step 3:** Get contextual guidance
- OrbyGlasses will provide navigation instructions
- It considers current obstacles and objects
- Gives safe, accurate directions

## Example Conversations

### Scenario 1: Finding a Destination
```
User: "Hey glasses"
OrbyGlasses: "I'm listening. How can I help?"

User: "I need to find a chair"
OrbyGlasses: "I can see a chair 3 meters ahead on your right. Walk forward slowly."

User: "Is the path clear?"
OrbyGlasses: "Yes, path is clear. Continue forward 2 meters then turn right."
```

### Scenario 2: Obstacle Avoidance
```
User: "Hey glasses"
OrbyGlasses: "I'm listening. How can I help?"

User: "What's in my way?"
OrbyGlasses: "There's a person directly ahead at arm's length. Step to your left."

User: "Can I go forward now?"
OrbyGlasses: "Yes, path ahead is clear for the next 5 meters."
```

### Scenario 3: Navigation to Room
```
User: "Hey glasses"
OrbyGlasses: "I'm listening. How can I help?"

User: "Take me to the bathroom"
OrbyGlasses: "I'll guide you to the bathroom. Turn slightly right and walk forward."

User: "How much further?"
OrbyGlasses: "I see a door 2 meters ahead on the right. That may be your destination."
```

## Configuration Options

Edit `config/config.yaml` to customize:

```yaml
conversation:
  enabled: true                      # Enable/disable feature
  model: "gemma3:4b"                 # LLM model (requires Ollama)
  temperature: 0.7                   # Response creativity (0-1)
  max_tokens: 200                    # Maximum response length
  voice_input: true                  # Enable microphone input
  activation_phrase: "hey glasses"   # Custom wake phrase
  check_interval: 2.0                # Seconds between activation checks
```

## Technical Details

### Architecture
- **Voice Recognition**: Google Speech Recognition API
- **Natural Language Processing**: Ollama (Gemma 3 4B)
- **Context Management**: Real-time scene understanding
- **Goal Tracking**: Persistent navigation state

### System Requirements
- **Microphone**: Required for voice input
- **Ollama**: Must be running with Gemma model installed
- **Internet**: Required for Google Speech Recognition

### Privacy
- Voice input uses Google Speech Recognition (requires internet)
- No conversation data is stored or transmitted beyond recognition
- All AI processing happens locally via Ollama

## Troubleshooting

### "No module named 'speech_recognition'"
```bash
pip install SpeechRecognition
```

### Microphone not working
1. Check microphone permissions in System Preferences
2. Test with: `python -m speech_recognition`
3. Adjust ambient noise calibration in config

### Wake phrase not detected
- Speak clearly and at normal volume
- Reduce background noise
- Try adjusting `check_interval` to 1.0 seconds
- Ensure microphone is working with other apps

### Slow responses
- Ensure Ollama is running: `ollama list`
- Use faster model: Change to `gemma3:2b` in config
- Reduce `max_tokens` to 150

## Best Practices

1. **Speak Naturally**: Use conversational language
2. **Be Specific**: "Take me to the kitchen door" vs "go there"
3. **Ask for Clarification**: "What do you mean?" if unclear
4. **Set Clear Goals**: Tell it your destination upfront
5. **Check Safety**: Always ask "Is it safe?" before moving

## Future Enhancements

- [ ] Offline voice recognition
- [ ] Multi-language support
- [ ] Custom wake phrases
- [ ] Conversation history export
- [ ] Smart home integration
- [ ] Location mapping and memory

## Examples of Super Accurate Use Cases

### Use Case 1: Morning Routine
```
User: "Hey glasses, help me get ready for the day"
OrbyGlasses: "I can guide you. What do you need to find first?"

User: "Take me to the bathroom"
OrbyGlasses: "Walk forward 3 meters, then turn left. I see a door ahead."

User: "Is this the bathroom door?"
OrbyGlasses: "Yes, I see a door directly in front of you at arm's length."
```

### Use Case 2: Kitchen Navigation
```
User: "Hey glasses, I want to make coffee"
OrbyGlasses: "I'll help. Let's navigate to the kitchen."

User: "Where's the coffee maker?"
OrbyGlasses: "I see what looks like a coffee maker on the counter 2 meters ahead on your right."

User: "Any obstacles?"
OrbyGlasses: "There's a chair on your left. Stay to the right as you walk forward."
```

### Use Case 3: Finding Lost Items
```
User: "Hey glasses, help me find my phone"
OrbyGlasses: "I'll look for it. Where did you last see it?"

User: "On the table"
OrbyGlasses: "I see a table 4 meters ahead. Walk forward slowly."

User: "Do you see a phone?"
OrbyGlasses: "I see a rectangular object on the table, could be a phone. It's on the far right side."
```

---

**Made with ❤️ by OrbyGlasses Team**
For support, open an issue on GitHub.
