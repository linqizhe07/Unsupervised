import numpy as np
from rl_agent.reward_utils import build_env_state_from_transition


class CustomEnvironment:
    def __init__(self):
        self._env_state = {"observation": None}

    def update_state(self, observation, joint_velocities=None, joint_forces=None):
        obs = np.asarray(observation, dtype=np.float32)
        self._env_state = build_env_state_from_transition(
            obs=obs,
            action=None,
            next_obs=obs,
            reward_on="next",
            joint_velocities=joint_velocities,
            joint_forces=joint_forces,
        )

    @property
    def env_state(self):
        return self._env_state
