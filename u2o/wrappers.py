"""
Minimal environment wrappers for u2o.
The SFAgent handles skill conditioning internally, so we only need
simple wrappers for observation/action normalization if needed.
"""

import gymnasium as gym
import numpy as np


class EpisodeMonitor(gym.Wrapper):
    """Tracks episode statistics (reward, length, velocity)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.episode_reward = 0.0
        self.episode_length = 0
        self.velocities = []

    def reset(self, **kwargs):
        self.episode_reward = 0.0
        self.episode_length = 0
        self.velocities = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_length += 1
        if "x_velocity" in info:
            self.velocities.append(info["x_velocity"])

        if terminated or truncated:
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_length,
                "velocities": self.velocities,
            }

        return obs, reward, terminated, truncated, info
