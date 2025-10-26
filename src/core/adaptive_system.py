"""
OrbyGlasses - Adaptive System Enhancements
Provides context-aware features: time-based adaptation, emergency mode, battery optimization.
"""

import datetime
import logging
import time
from typing import Dict, Optional
import platform


class AdaptiveSystemManager:
    """
    Manages context-aware adaptations for better user experience.
    Includes time-of-day adaptation, emergency alerts, and battery optimization.
    """

    def __init__(self, config):
        """
        Initialize adaptive system manager.

        Args:
            config: ConfigManager instance
        """
        self.config = config
        self.emergency_mode = False
        self.emergency_start_time = None

        # Time-based settings
        self.time_adaptations_enabled = True
        self.current_time_mode = "day"  # "night", "day", "dawn", "dusk"

        # Battery monitoring
        self.battery_monitoring_enabled = self._check_battery_support()
        self.battery_saver_threshold = 20  # Activate saver below 20%
        self.battery_saver_active = False

        logging.info("Adaptive system manager initialized")

    def _check_battery_support(self) -> bool:
        """Check if battery monitoring is supported on this platform."""
        try:
            if platform.system() == "Darwin":  # macOS
                import subprocess
                subprocess.run(["pmset", "-g", "batt"], capture_output=True, check=True)
                return True
        except:
            pass
        return False

    def get_time_of_day_context(self) -> Dict:
        """
        Get time-based context adjustments.

        Returns:
            Dictionary with adjusted parameters based on time of day
        """
        if not self.time_adaptations_enabled:
            return {}

        now = datetime.datetime.now()
        hour = now.hour

        # Determine time mode
        if 22 <= hour or hour < 5:  # 10pm - 5am
            mode = "night"
            adjustments = {
                "audio_volume": 1.3,  # Louder warnings at night
                "danger_distance": 2.0,  # More cautious
                "audio_update_interval": 0.8,  # More frequent updates
                "description": "Night mode: Enhanced safety, louder alerts"
            }
        elif 5 <= hour < 7:  # Dawn
            mode = "dawn"
            adjustments = {
                "audio_volume": 1.1,
                "danger_distance": 1.7,
                "description": "Dawn mode: Moderate safety adjustments"
            }
        elif 19 <= hour < 22:  # Dusk
            mode = "dusk"
            adjustments = {
                "audio_volume": 1.1,
                "danger_distance": 1.7,
                "description": "Dusk mode: Moderate safety adjustments"
            }
        else:  # Day
            mode = "day"
            adjustments = {
                "audio_volume": 1.0,
                "danger_distance": 1.0,
                "description": "Day mode: Standard settings"
            }

        if mode != self.current_time_mode:
            self.current_time_mode = mode
            logging.info(f"Time mode changed to: {mode}")

        return adjustments

    def activate_emergency_mode(self):
        """
        Activate emergency alert mode.
        Triggers loud beeping, flashing visuals, and emergency logging.
        """
        if not self.emergency_mode:
            self.emergency_mode = True
            self.emergency_start_time = time.time()
            logging.critical("🚨 EMERGENCY MODE ACTIVATED 🚨")

    def deactivate_emergency_mode(self):
        """Deactivate emergency mode."""
        if self.emergency_mode:
            duration = time.time() - self.emergency_start_time
            self.emergency_mode = False
            self.emergency_start_time = None
            logging.info(f"Emergency mode deactivated (was active for {duration:.1f}s)")

    def get_emergency_status(self) -> Dict:
        """
        Get emergency mode status and configuration.

        Returns:
            Dictionary with emergency mode settings
        """
        if not self.emergency_mode:
            return {"active": False}

        return {
            "active": True,
            "duration": time.time() - self.emergency_start_time,
            "audio_settings": {
                "emergency_beep": True,
                "beep_frequency": 880,  # Hz
                "beep_interval": 0.3,  # seconds
                "voice_message": "HELP. Emergency alert activated."
            },
            "visual_settings": {
                "flash_screen": True,
                "flash_color": (0, 0, 255),  # Red
                "flash_interval": 0.5
            }
        }

    def get_battery_status(self) -> Optional[Dict]:
        """
        Get battery status and optimization recommendations.

        Returns:
            Dictionary with battery info or None if not supported
        """
        if not self.battery_monitoring_enabled:
            return None

        try:
            if platform.system() == "Darwin":  # macOS
                import subprocess
                result = subprocess.run(
                    ["pmset", "-g", "batt"],
                    capture_output=True,
                    text=True,
                    check=True
                )

                # Parse battery percentage
                output = result.stdout
                if "%" in output:
                    # Extract percentage (e.g., "50%" -> 50)
                    import re
                    match = re.search(r'(\d+)%', output)
                    if match:
                        battery_level = int(match.group(1))

                        # Check if battery saver should activate
                        should_activate = battery_level < self.battery_saver_threshold

                        if should_activate and not self.battery_saver_active:
                            self.battery_saver_active = True
                            logging.warning(f"Battery saver activated ({battery_level}%)")
                        elif not should_activate and self.battery_saver_active:
                            self.battery_saver_active = False
                            logging.info(f"Battery saver deactivated ({battery_level}%)")

                        return {
                            "level": battery_level,
                            "saver_active": self.battery_saver_active,
                            "optimizations": self._get_battery_optimizations() if self.battery_saver_active else {}
                        }
        except Exception as e:
            logging.debug(f"Battery status check failed: {e}")

        return None

    def _get_battery_optimizations(self) -> Dict:
        """
        Get recommended settings for battery saver mode.

        Returns:
            Dictionary with optimized settings
        """
        return {
            "depth_skip_frames": 3,  # Process every 4th frame
            "max_detections": 3,  # Track fewer objects
            "camera_fps": 15,  # Reduce camera FPS
            "disable_features": ["trajectory_prediction", "occupancy_grid_3d", "point_cloud_viewer"],
            "audio_update_interval": 2.0,  # Less frequent audio
            "slam_visualize": False,  # Disable visualization
            "description": "Battery saver: Reduced processing, essential features only"
        }

    def get_contextual_adjustments(self) -> Dict:
        """
        Get all contextual adjustments combined.

        Returns:
            Dictionary with all active adjustments
        """
        adjustments = {}

        # Time-based
        time_adj = self.get_time_of_day_context()
        if time_adj:
            adjustments["time_of_day"] = time_adj

        # Emergency
        emergency_status = self.get_emergency_status()
        if emergency_status["active"]:
            adjustments["emergency"] = emergency_status

        # Battery
        battery_status = self.get_battery_status()
        if battery_status and battery_status["saver_active"]:
            adjustments["battery_saver"] = battery_status

        return adjustments

    def should_reduce_performance(self) -> bool:
        """
        Check if performance should be reduced (battery saver active).

        Returns:
            True if battery saver is active
        """
        return self.battery_saver_active
