"""
Episode-based replay buffer with future sampling for u2o.
Adapted from u2o_zsrl/url_benchmark/in_memory_replay_buffer.py
Simplified: no dm_env dependency, no physics state, no pixel support.
"""

import typing as tp
import dataclasses
import collections
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

T = tp.TypeVar("T", np.ndarray, torch.Tensor)


@dataclasses.dataclass
class EpisodeBatch(tp.Generic[T]):
    """Batch of transitions for training."""

    obs: T
    action: T
    reward: T
    next_obs: T
    discount: T
    future_obs: tp.Optional[T] = None

    def to(self, device: str) -> "EpisodeBatch[torch.Tensor]":
        out: tp.Dict[str, tp.Any] = {}
        for field in dataclasses.fields(self):
            data = getattr(self, field.name)
            if isinstance(data, (torch.Tensor, np.ndarray)):
                out[field.name] = torch.as_tensor(data, device=device, dtype=torch.float32)
            elif data is None:
                out[field.name] = data
            else:
                raise RuntimeError(f"Unknown type for {field.name}: {type(data)}")
        return EpisodeBatch(**out)


class ReplayBuffer:
    """Episode-based replay buffer with geometric future sampling."""

    def __init__(
        self,
        max_episodes: int,
        discount: float = 0.99,
        future: float = 0.99,
        p_currgoal: float = 0.0,
        p_randomgoal: float = 0.375,
        max_episode_length: tp.Optional[int] = None,
    ) -> None:
        self._max_episodes = max_episodes
        self._discount = discount
        assert 0 <= future <= 1
        self._future = future
        self._p_currgoal = p_currgoal
        self._p_randomgoal = p_randomgoal
        self._max_episode_length = max_episode_length

        self._current_episode: tp.Dict[
            str, tp.List[np.ndarray]
        ] = collections.defaultdict(list)
        self._idx = 0
        self._full = False
        self._num_transitions = 0
        self._storage: tp.Dict[str, np.ndarray] = {}
        self._episodes_length = np.zeros(max_episodes, dtype=np.int32)
        self._episodes_selection_probability = None
        self._is_fixed_episode_length = True

    def __len__(self) -> int:
        return self._max_episodes if self._full else self._idx

    @property
    def num_transitions(self) -> int:
        return int(self._episodes_length[: len(self)].sum())

    def add_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        discount: float = 1.0,
    ) -> None:
        """Add a single transition. Starts new episode on first call or after done."""
        self._current_episode["observation"].append(np.array(obs, dtype=np.float32))
        self._current_episode["action"].append(np.array(action, dtype=np.float32))
        self._current_episode["reward"].append(
            np.array([reward], dtype=np.float32)
        )
        self._current_episode["discount"].append(
            np.array([discount], dtype=np.float32)
        )
        self._current_episode["next_observation"].append(
            np.array(next_obs, dtype=np.float32)
        )

        if done:
            self._store_episode()

    def _store_episode(self) -> None:
        """Store current episode into the circular buffer."""
        if not self._current_episode:
            return

        ep_len = len(self._current_episode["observation"])
        if ep_len == 0:
            return

        for name, value_list in self._current_episode.items():
            values = np.array(value_list, dtype=np.float32)
            if name not in self._storage:
                if self._max_episode_length is not None:
                    shape = (self._max_episode_length,) + values.shape[1:]
                else:
                    shape = values.shape
                self._storage[name] = np.zeros(
                    (self._max_episodes,) + shape, dtype=np.float32
                )
            self._storage[name][self._idx][:ep_len] = values

        self._episodes_length[self._idx] = ep_len
        if (
            self._idx > 0
            and self._episodes_length[self._idx] != self._episodes_length[self._idx - 1]
            and self._episodes_length[self._idx - 1] != 0
        ):
            self._is_fixed_episode_length = False

        self._current_episode = collections.defaultdict(list)
        self._idx = (self._idx + 1) % self._max_episodes
        self._full = self._full or self._idx == 0
        self._episodes_selection_probability = None

    def sample(self, batch_size: int) -> EpisodeBatch:
        """Sample a batch of transitions with optional future state sampling."""
        n_eps = len(self)
        assert n_eps > 0, "Cannot sample from empty buffer"

        if self._is_fixed_episode_length:
            ep_idx = np.random.randint(0, n_eps, size=batch_size)
            random_ep_idx = np.random.randint(0, n_eps, size=batch_size)
        else:
            if self._episodes_selection_probability is None:
                lengths = self._episodes_length[:n_eps].astype(np.float64)
                self._episodes_selection_probability = lengths / lengths.sum()
            ep_idx = np.random.choice(n_eps, size=batch_size, p=self._episodes_selection_probability)
            random_ep_idx = np.random.choice(n_eps, size=batch_size, p=self._episodes_selection_probability)

        eps_lengths = self._episodes_length[ep_idx]
        random_eps_lengths = self._episodes_length[random_ep_idx]

        step_idx = np.array([np.random.randint(0, l) for l in eps_lengths])
        random_step_idx = np.array([np.random.randint(0, l) for l in random_eps_lengths])

        obs = self._storage["observation"][ep_idx, step_idx]
        action = self._storage["action"][ep_idx, step_idx]
        reward = self._storage["reward"][ep_idx, step_idx]
        next_obs = self._storage["next_observation"][ep_idx, step_idx]
        discount = self._discount * self._storage["discount"][ep_idx, step_idx]

        future_obs = None
        if self._future < 1:
            # Geometric future sampling
            future_offset = np.random.geometric(
                p=(1 - self._future), size=batch_size
            )
            future_step = np.clip(step_idx + future_offset, 0, eps_lengths - 1)
            future_obs = self._storage["observation"][ep_idx, future_step]

            # Mix in current goal with probability p_currgoal
            curr_obs = self._storage["observation"][ep_idx, step_idx]
            if self._p_randomgoal < 1.0:
                mask = (
                    np.random.rand(batch_size)
                    < self._p_currgoal / (1.0 - self._p_randomgoal)
                ).reshape(-1, *([1] * (len(future_obs.shape) - 1)))
                future_obs = np.where(mask, curr_obs, future_obs)

            # Mix in random goal with probability p_randomgoal
            random_obs = self._storage["observation"][random_ep_idx, random_step_idx]
            mask = (np.random.rand(batch_size) < self._p_randomgoal).reshape(
                -1, *([1] * (len(future_obs.shape) - 1))
            )
            future_obs = np.where(mask, random_obs, future_obs)

        return EpisodeBatch(
            obs=obs,
            action=action,
            reward=reward,
            next_obs=next_obs,
            discount=discount,
            future_obs=future_obs,
        )

    def save(self, path: str) -> None:
        """Save buffer to disk."""
        data = {
            "max_episodes": self._max_episodes,
            "discount": self._discount,
            "future": self._future,
            "idx": self._idx,
            "full": self._full,
            "episodes_length": self._episodes_length,
        }
        for name, arr in self._storage.items():
            data[f"storage_{name}"] = arr
        np.savez_compressed(path, **data)
        logger.info(f"Saved replay buffer to {path} ({len(self)} episodes)")

    def load(self, path: str) -> None:
        """Load buffer from disk."""
        data = np.load(path, allow_pickle=True)
        self._max_episodes = int(data["max_episodes"])
        self._discount = float(data["discount"])
        self._future = float(data["future"])
        self._idx = int(data["idx"])
        self._full = bool(data["full"])
        self._episodes_length = data["episodes_length"]

        self._storage = {}
        for key in data.files:
            if key.startswith("storage_"):
                name = key[len("storage_"):]
                self._storage[name] = data[key]

        self._episodes_selection_probability = None
        # Check if fixed length
        lengths = self._episodes_length[: len(self)]
        self._is_fixed_episode_length = (lengths.min() == lengths.max()) if len(lengths) > 0 else True
        logger.info(f"Loaded replay buffer from {path} ({len(self)} episodes, {self.num_transitions} transitions)")

    def relabel_rewards(self, reward_fn: tp.Callable) -> None:
        """Recompute rewards using a new reward function.

        reward_fn: callable(obs, action, next_obs) -> float
        """
        n_eps = len(self)
        for ep_idx in range(n_eps):
            ep_len = self._episodes_length[ep_idx]
            for step_idx in range(ep_len):
                obs = self._storage["observation"][ep_idx, step_idx]
                action = self._storage["action"][ep_idx, step_idx]
                next_obs = self._storage["next_observation"][ep_idx, step_idx]
                reward = reward_fn(obs, action, next_obs)
                self._storage["reward"][ep_idx, step_idx] = np.array(
                    [reward], dtype=np.float32
                )
