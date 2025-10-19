# Social Navigation AI Feature

## Overview
The Social Navigation AI feature enables OrbyGlasses to navigate crowded areas using social norms and conventions. This feature helps users follow regional social conventions like "stay to the right" in hallways and safely navigate through crowds by identifying gaps and proper passing etiquette.

## Features

### Regional Social Norms
- **US Convention**: Stay to the right in hallways
- **UK/Japan Convention**: Stay to the left
- Configurable by region in settings

### Crowd Analysis
- Analyzes crowd density (sparse, moderate, dense)
- Identifies available gaps in crowds
- Tracks people positions and movements
- Suggests optimal paths based on social norms

### Smart Guidance
- "Stay to the right in hallway" (US convention)
- "Gap opening in crowd ahead on left"
- "People yielding space, safe to proceed"
- Context-aware navigation in crowded environments

## Usage Examples

### Voice Commands
- "How do I navigate through this crowd?"
- "Where should I walk in this hallway?"
- "Is there a gap in the crowd ahead?"

### System Responses
The system will provide guidance like:
- "Following US convention: stay to the right in hallways"
- "Gap opening in crowd ahead on your right. Safe to proceed to the right."
- "People yielding space, safe to proceed."

## Configuration

Edit `config/config.yaml` to customize social navigation:

```yaml
social_navigation:
  enabled: true              # Enable social navigation features
  region: "us"               # Regional norm ('us', 'uk', 'japan', etc.)
  voice_announce: true       # Announce social navigation advice through voice
```

## Technical Details

### Architecture
- **SocialNavigationAI**: Core class implementing social navigation logic
- **Integration**: Seamlessly integrated with existing conversation system
- **Detection**: Uses existing object detection and depth estimation
- **Regional Adaptation**: Configurable social norms by region

### System Requirements
- Same as base OrbyGlasses system
- No additional dependencies required

## Implementation
The Social Navigation AI works by:
1. Analyzing detected people in the scene using the existing detection pipeline
2. Determining crowd density and available gaps for passage
3. Applying regional social norms to suggest appropriate navigation paths
4. Integrating with the conversation system to respond to social navigation queries
5. Providing real-time guidance based on current scene analysis

## Privacy
- All social navigation processing happens locally
- No additional data collection beyond normal navigation operation
- No data transmission for social navigation features