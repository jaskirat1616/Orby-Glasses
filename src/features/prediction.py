"""
OrbyGlasses - Reinforcement Learning Prediction
Uses RL to learn navigation patterns and predict optimal paths.
"""

import os
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import json


class NavigationEnvironment(gym.Env):
    """
    Custom Gym environment for navigation learning.
    """

    def __init__(self, max_obstacles: int = 10):
        """
        Initialize navigation environment.

        Args:
            max_obstacles: Maximum number of obstacles to consider
        """
        super(NavigationEnvironment, self).__init__()

        self.max_obstacles = max_obstacles

        # Observation space: [obstacle_distances (10), obstacle_angles (10), current_velocity (2)]
        obs_dim = max_obstacles * 2 + 2
        self.observation_space = spaces.Box(
            low=0, high=10.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Action space: [move_direction (0-8), speed (0-2)]
        # 0=forward, 1=forward-left, 2=left, 3=back-left, 4=back, 5=back-right, 6=right, 7=forward-right, 8=stop
        # Speed: 0=slow, 1=normal, 2=fast
        self.action_space = spaces.MultiDiscrete([9, 3])

        # State
        self.current_obstacles = []
        self.current_position = np.array([0.0, 0.0])
        self.current_velocity = np.array([0.0, 0.0])
        self.goal_position = np.array([10.0, 0.0])

        self.steps = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        # Random starting position
        self.current_position = np.array([0.0, 0.0])
        self.current_velocity = np.array([0.0, 0.0])

        # Random goal
        angle = np.random.uniform(-np.pi/4, np.pi/4)
        distance = np.random.uniform(5.0, 10.0)
        self.goal_position = np.array([
            distance * np.cos(angle),
            distance * np.sin(angle)
        ])

        # Random obstacles
        self.current_obstacles = self._generate_obstacles()

        self.steps = 0

        obs = self._get_observation()
        info = {}

        return obs, info

    def _generate_obstacles(self) -> List[Dict]:
        """Generate random obstacles."""
        num_obstacles = np.random.randint(0, self.max_obstacles)
        obstacles = []

        for _ in range(num_obstacles):
            # Random position in front of agent
            distance = np.random.uniform(0.5, 8.0)
            angle = np.random.uniform(-np.pi/2, np.pi/2)

            obstacle = {
                'distance': distance,
                'angle': angle,
                'type': np.random.choice(['person', 'vehicle', 'object'])
            }
            obstacles.append(obstacle)

        return obstacles

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Fill obstacle distances
        for i, obstacle in enumerate(self.current_obstacles[:self.max_obstacles]):
            obs[i] = obstacle['distance']

        # Fill obstacle angles
        for i, obstacle in enumerate(self.current_obstacles[:self.max_obstacles]):
            obs[self.max_obstacles + i] = obstacle['angle']

        # Current velocity
        obs[-2:] = self.current_velocity

        return obs

    def step(self, action):
        """
        Execute action and return next state.

        Args:
            action: [direction, speed]

        Returns:
            observation, reward, terminated, truncated, info
        """
        direction, speed = action

        # Convert action to velocity
        direction_map = {
            0: (1, 0),    # forward
            1: (1, -1),   # forward-left
            2: (0, -1),   # left
            3: (-1, -1),  # back-left
            4: (-1, 0),   # back
            5: (-1, 1),   # back-right
            6: (0, 1),    # right
            7: (1, 1),    # forward-right
            8: (0, 0)     # stop
        }

        speed_map = {0: 0.5, 1: 1.0, 2: 1.5}

        dx, dy = direction_map[direction]
        speed_val = speed_map[speed]

        # Update velocity
        self.current_velocity = np.array([dx, dy]) * speed_val

        # Update position
        self.current_position += self.current_velocity * 0.1  # dt = 0.1

        # Calculate reward
        reward = self._calculate_reward(action)

        # Check termination
        self.steps += 1
        terminated = self._check_collision() or self._reached_goal()
        truncated = self.steps >= self.max_steps

        # Update obstacles (simulate movement)
        self._update_obstacles()

        obs = self._get_observation()
        info = {
            'position': self.current_position.copy(),
            'goal': self.goal_position.copy(),
            'collision': self._check_collision()
        }

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self, action) -> float:
        """Calculate reward for current state and action."""
        reward = 0.0

        # Progress towards goal
        distance_to_goal = np.linalg.norm(self.goal_position - self.current_position)
        reward += -distance_to_goal * 0.1

        # Penalty for collision risk
        for obstacle in self.current_obstacles:
            if obstacle['distance'] < 1.5:  # Danger zone
                reward += -10.0
            elif obstacle['distance'] < 3.0:  # Caution zone
                reward += -2.0

        # Reward for reaching goal
        if distance_to_goal < 0.5:
            reward += 100.0

        # Penalty for stopping unnecessarily
        direction, speed = action
        if direction == 8 and distance_to_goal > 1.0:
            reward += -5.0

        # Small step penalty to encourage efficiency
        reward += -0.1

        return reward

    def _check_collision(self) -> bool:
        """Check if collision occurred."""
        for obstacle in self.current_obstacles:
            if obstacle['distance'] < 0.5:
                return True
        return False

    def _reached_goal(self) -> bool:
        """Check if goal reached."""
        distance = np.linalg.norm(self.goal_position - self.current_position)
        return distance < 0.5

    def _update_obstacles(self):
        """Update obstacle positions (simulate movement)."""
        for obstacle in self.current_obstacles:
            # Simple update: decrease distance (obstacles approach)
            obstacle['distance'] -= self.current_velocity[0] * 0.1

    def render(self):
        """Render environment (optional)."""
        pass


class NavigationPredictor:
    """
    RL-based navigation predictor using PPO.
    """

    def __init__(self, config):
        """
        Initialize navigation predictor.

        Args:
            config: ConfigManager instance
        """
        self.config = config
        self.enabled = config.get('prediction.enabled', True)

        if not self.enabled:
            logging.info("RL prediction disabled")
            return

        # Model path
        self.model_path = config.get('prediction.model_path', 'models/rl/ppo_navigation.zip')
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Training params
        self.training_steps = config.get('prediction.training_steps', 10000)
        self.save_interval = config.get('prediction.save_interval', 1000)

        # Create environment
        self.env = DummyVecEnv([lambda: NavigationEnvironment()])

        # Load or create model
        if os.path.exists(self.model_path):
            logging.info(f"Loading RL model from {self.model_path}")
            self.model = PPO.load(self.model_path, env=self.env)
        else:
            logging.info("Creating new RL model")
            self.model = PPO(
                "MlpPolicy",
                self.env,
                verbose=1,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                device='cpu'  # RL training on CPU is usually sufficient
            )

    def train(self, total_timesteps: Optional[int] = None):
        """
        Train the RL model.

        Args:
            total_timesteps: Number of timesteps to train (default from config)
        """
        if not self.enabled:
            logging.warning("RL prediction disabled, cannot train")
            return

        timesteps = total_timesteps or self.training_steps

        logging.info(f"Training RL model for {timesteps} timesteps...")

        # Callback for saving
        checkpoint_callback = CheckpointCallback(
            save_freq=self.save_interval,
            save_path=os.path.dirname(self.model_path),
            name_prefix='ppo_navigation'
        )

        try:
            self.model.learn(
                total_timesteps=timesteps,
                callback=checkpoint_callback
            )

            # Save final model
            self.model.save(self.model_path)
            logging.info(f"Model saved to {self.model_path}")

        except Exception as e:
            logging.error(f"Training error: {e}")

    def predict_action(self, detections: List[Dict]) -> Dict:
        """
        Predict optimal action based on current detections.

        Args:
            detections: List of detections with depth

        Returns:
            Prediction dict with 'action', 'speed', 'confidence'
        """
        if not self.enabled or not hasattr(self, 'model'):
            return {'action': 'forward', 'speed': 'normal', 'confidence': 0.0}

        try:
            # Convert detections to observation
            obs = self._detections_to_observation(detections)

            # Predict action
            action, _states = self.model.predict(obs, deterministic=True)

            # Convert action to human-readable format
            direction_map = {
                0: 'forward', 1: 'forward-left', 2: 'left', 3: 'back-left',
                4: 'back', 5: 'back-right', 6: 'right', 7: 'forward-right', 8: 'stop'
            }
            speed_map = {0: 'slow', 1: 'normal', 2: 'fast'}

            direction_idx = int(action[0][0])
            speed_idx = int(action[0][1])

            prediction = {
                'action': direction_map.get(direction_idx, 'forward'),
                'speed': speed_map.get(speed_idx, 'normal'),
                'confidence': 0.8  # Placeholder, could be computed from model uncertainty
            }

            return prediction

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return {'action': 'forward', 'speed': 'normal', 'confidence': 0.0}

    def _detections_to_observation(self, detections: List[Dict]) -> np.ndarray:
        """
        Convert detections to RL observation format.

        Args:
            detections: List of detections

        Returns:
            Observation array
        """
        max_obstacles = 10
        obs = np.zeros(max_obstacles * 2 + 2, dtype=np.float32)

        # Fill obstacle distances and angles
        for i, det in enumerate(detections[:max_obstacles]):
            depth = det.get('depth', 10.0)
            center = det.get('center', [320, 240])

            # Calculate angle from center (assume 640x480 frame)
            angle = (center[0] - 320) / 320 * (np.pi / 2)  # -pi/2 to pi/2

            obs[i] = depth
            obs[max_obstacles + i] = angle

        # Current velocity (assume stationary for now)
        obs[-2:] = [0.0, 0.0]

        return obs.reshape(1, -1)

    def log_experience(self, detections: List[Dict], action: str, outcome: str):
        """
        Log user experience for future training.

        Args:
            detections: Detections at decision point
            action: Action taken
            outcome: Outcome (success/collision/etc)
        """
        log_dir = "data/logs/rl_experiences"
        os.makedirs(log_dir, exist_ok=True)

        experience = {
            'detections': [{k: v for k, v in d.items() if k not in ['bbox']} for d in detections],
            'action': action,
            'outcome': outcome
        }

        log_file = os.path.join(log_dir, "experiences.jsonl")
        with open(log_file, 'a') as f:
            f.write(json.dumps(experience) + '\n')


class PathPlanner:
    """
    High-level path planner combining RL predictions with heuristics.
    """

    def __init__(self, config):
        """Initialize path planner."""
        self.config = config
        self.predictor = NavigationPredictor(config)

    def plan_path(self, detections: List[Dict], navigation_summary: Dict) -> Dict:
        """
        Plan optimal path based on detections and learned patterns.

        Args:
            detections: Current detections
            navigation_summary: Navigation summary

        Returns:
            Path plan with recommended action
        """
        # Get RL prediction
        rl_prediction = self.predictor.predict_action(detections)

        # Get heuristic recommendation
        heuristic = self._heuristic_planner(navigation_summary)

        # Combine predictions
        # If RL confidence is high, use it; otherwise use heuristic
        if rl_prediction['confidence'] > 0.6:
            plan = {
                'action': rl_prediction['action'],
                'speed': rl_prediction['speed'],
                'source': 'rl',
                'confidence': rl_prediction['confidence']
            }
        else:
            plan = heuristic
            plan['source'] = 'heuristic'

        return plan

    def _heuristic_planner(self, summary: Dict) -> Dict:
        """
        Simple heuristic path planner.

        Args:
            summary: Navigation summary

        Returns:
            Heuristic plan
        """
        danger = len(summary.get('danger_objects', []))
        caution = len(summary.get('caution_objects', []))
        path_clear = summary.get('path_clear', True)

        if danger > 0:
            return {'action': 'stop', 'speed': 'slow', 'confidence': 0.9}
        elif caution > 0:
            return {'action': 'forward', 'speed': 'slow', 'confidence': 0.7}
        elif path_clear:
            return {'action': 'forward', 'speed': 'normal', 'confidence': 0.8}
        else:
            return {'action': 'forward', 'speed': 'slow', 'confidence': 0.6}
