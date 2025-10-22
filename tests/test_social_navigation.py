"""
Test script for Social Navigation AI feature
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.social_navigation import SocialNavigationAI
from src.utils import ConfigManager


def test_social_navigation_basic():
    """Test basic social navigation functionality"""
    print("Testing Social Navigation AI...")
    
    # Initialize the social navigation system
    social_nav = SocialNavigationAI()
    
    # Test setting different regions
    print("\n1. Testing regional norms:")
    social_nav.set_region('us')
    print(f"  - Set region to US: {social_nav.active_norm.description}")
    
    social_nav.set_region('uk')
    print(f"  - Set region to UK: {social_nav.active_norm.description}")
    
    social_nav.set_region('japan')
    print(f"  - Set region to Japan: {social_nav.active_norm.description}")
    
    # Test with no people detected
    print("\n2. Testing with no people:")
    no_people_detections = []
    result = social_nav.analyze_crowd(no_people_detections)
    print(f"  - Crowd density: {result.crowd_density}")
    print(f"  - Suggested path: {result.suggested_path}")
    print(f"  - Social advice: {result.social_norm_advice}")
    
    # Test with some people detections
    print("\n3. Testing with people detections:")
    people_detections = [
        {
            'label': 'person',
            'depth': 2.0,
            'center': [100, 150],
            'bbox': [90, 140, 110, 160]
        },
        {
            'label': 'person', 
            'depth': 1.5,
            'center': [200, 150],
            'bbox': [190, 140, 210, 160]
        }
    ]
    
    result = social_nav.analyze_crowd(people_detections)
    print(f"  - Crowd density: {result.crowd_density}")
    print(f"  - Suggested path: {result.suggested_path}")
    print(f"  - Social advice: {result.social_norm_advice}")
    
    # Test social navigation guidance
    print("\n4. Testing social navigation guidance:")
    guidance = social_nav.get_social_navigation_guidance(people_detections)
    print(f"  - Generated guidance: {guidance}")
    
    # Test with dense crowd
    print("\n5. Testing with dense crowd:")
    dense_people = [
        {
            'label': 'person',
            'depth': 1.0,
            'center': [80, 100],
            'bbox': [70, 90, 90, 110]
        },
        {
            'label': 'person',
            'depth': 1.2,
            'center': [150, 120], 
            'bbox': [140, 110, 160, 130]
        },
        {
            'label': 'person',
            'depth': 0.8,
            'center': [220, 140],
            'bbox': [210, 130, 230, 150]
        },
        {
            'label': 'person',
            'depth': 1.5,
            'center': [250, 160],
            'bbox': [240, 150, 260, 170]
        }
    ]
    
    dense_result = social_nav.analyze_crowd(dense_people)
    print(f"  - Crowd density: {dense_result.crowd_density}")
    print(f"  - Suggested path: {dense_result.suggested_path}")
    print(f"  - Social advice: {dense_result.social_norm_advice}")

    print("\n✓ Social Navigation AI tests completed successfully!")


def test_integration_with_config():
    """Test social navigation with configuration"""
    print("\nTesting Social Navigation AI with config integration...")
    
    # Create a minimal config
    config_content = """
social_navigation:
  region: "us"
"""
    
    # Write temporary config
    with open('temp_test_config.yaml', 'w') as f:
        f.write(config_content)
    
    config = ConfigManager('temp_test_config.yaml')
    
    # Initialize social navigation and set region from config
    social_nav = SocialNavigationAI()
    region = config.get('social_navigation.region', 'us')
    social_nav.set_region(region)
    
    print(f"  - Configured region: {region}")
    print(f"  - Active norm: {social_nav.active_norm.description}")
    
    # Clean up
    os.remove('temp_test_config.yaml')
    
    print("✓ Config integration test completed!")


if __name__ == "__main__":
    print("Starting Social Navigation AI tests...")
    
    test_social_navigation_basic()
    test_integration_with_config()
    
    print("\n🎉 All Social Navigation AI tests passed!")