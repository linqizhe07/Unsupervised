"""
SFAgent: Successor Feature agent with pluggable feature learners.
Adapted from u2o_zsrl/url_benchmark/agent/sf.py for Revolve's HumanoidEnv.
Removed: pixel/encoder support, DMC-specific code, Hydra config store.
"""

import copy
import math
import logging
import dataclasses
from collections import OrderedDict
import typing as tp

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from u2o import utils
from u2o.fb_modules import Actor, DiagGaussianActor, ForwardMap, BackwardMap, OnlineCov
from u2o.replay_buffer import ReplayBuffer
from u2o.networks import FEATURE_LEARNERS, HILP

logger = logging.getLogger(__name__)

MetaDict = OrderedDict


@dataclasses.dataclass
class SFAgentConfig:
    """Configuration for SFAgent."""

    obs_dim: int = 376
    action_dim: int = 17
    device: str = "cpu"
    lr: float = 1e-4
    lr_coef: float = 5
    sf_target_tau: float = 0.01
    update_every_steps: int = 1
    hidden_dim: int = 1024
    phi_hidden_dim: int = 512
    feature_dim: int = 512
    z_dim: int = 50
    stddev_schedule: str = "0.2"
    stddev_clip: float = 0.3
    update_z_every_step: int = 300
    batch_size: int = 1024
    boltzmann: bool = False
    log_std_bounds: tp.Tuple[float, float] = (-5, 2)
    temp: float = 1
    preprocess: bool = True
    num_sf_updates: int = 1
    feature_learner: str = "hilp"
    mix_ratio: float = 0.5
    q_loss: bool = True
    add_trunk: bool = False
    feature_type: str = "state"  # 'state', 'diff', 'concat'
    hilp_discount: float = 0.98
    hilp_expectile: float = 0.5
    use_rew_norm: bool = True
    num_expl_steps: int = 2000


class SFAgent:
    """Successor Feature agent with HILP feature learning."""

    def __init__(self, cfg: SFAgentConfig):
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.solved_meta: tp.Any = None

        # State-based only (no encoder for Revolve)
        self.obs_dim = cfg.obs_dim
        if cfg.feature_learner == "identity":
            cfg.z_dim = self.obs_dim

        # Create actor
        if cfg.boltzmann:
            self.actor: nn.Module = DiagGaussianActor(
                self.obs_dim,
                cfg.z_dim,
                self.action_dim,
                cfg.hidden_dim,
                cfg.log_std_bounds,
            ).to(cfg.device)
        else:
            self.actor = Actor(
                self.obs_dim,
                cfg.z_dim,
                self.action_dim,
                cfg.feature_dim,
                cfg.hidden_dim,
                preprocess=cfg.preprocess,
                add_trunk=cfg.add_trunk,
            ).to(cfg.device)

        # Successor feature networks (dual Q)
        self.successor_net = ForwardMap(
            self.obs_dim,
            cfg.z_dim,
            self.action_dim,
            cfg.feature_dim,
            cfg.hidden_dim,
            preprocess=cfg.preprocess,
            add_trunk=cfg.add_trunk,
        ).to(cfg.device)

        self.successor_target_net = ForwardMap(
            self.obs_dim,
            cfg.z_dim,
            self.action_dim,
            cfg.feature_dim,
            cfg.hidden_dim,
            preprocess=cfg.preprocess,
            add_trunk=cfg.add_trunk,
        ).to(cfg.device)
        self.successor_target_net.load_state_dict(self.successor_net.state_dict())

        # Feature learner
        learner_cls = FEATURE_LEARNERS[cfg.feature_learner]
        extra_kwargs = {}
        if cfg.feature_learner == "hilp":
            extra_kwargs["cfg"] = cfg
        self.feature_learner = learner_cls(
            self.obs_dim, self.action_dim, cfg.z_dim, cfg.phi_hidden_dim, **extra_kwargs
        ).to(cfg.device)

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.sf_opt = torch.optim.Adam(self.successor_net.parameters(), lr=cfg.lr)
        self.phi_opt: tp.Optional[torch.optim.Adam] = None
        if cfg.feature_learner not in ["random", "identity"]:
            self.phi_opt = torch.optim.Adam(
                self.feature_learner.parameters(), lr=cfg.lr_coef * cfg.lr
            )

        self.train()
        self.successor_target_net.train()

        # Reward normalization state
        self.rew_running_mean = torch.zeros(1).to(cfg.device)
        self.rew_running_std = torch.ones(1).to(cfg.device)
        self.init_rew_running_mean = True

    def train(self, training: bool = True) -> None:
        self.training = training
        for net in [self.actor, self.successor_net]:
            net.train(training)
        if self.phi_opt is not None:
            self.feature_learner.train()

    def init_from(self, other: "SFAgent") -> None:
        """Copy parameters from another agent."""
        for name in ["actor", "successor_net", "feature_learner", "successor_target_net"]:
            utils.hard_update_params(getattr(other, name), getattr(self, name))
        for key, val in self.__dict__.items():
            if isinstance(val, torch.optim.Optimizer):
                val.load_state_dict(copy.deepcopy(getattr(other, key).state_dict()))

    def sample_z(self, size: int) -> torch.Tensor:
        gaussian_rdv = torch.randn((size, self.cfg.z_dim), dtype=torch.float32)
        z = math.sqrt(self.cfg.z_dim) * F.normalize(gaussian_rdv, dim=1)
        return z

    def init_meta(self) -> MetaDict:
        if self.solved_meta is not None:
            return self.solved_meta
        z = self.sample_z(1).squeeze().numpy()
        meta = OrderedDict()
        meta["z"] = z
        return meta

    def update_meta(self, meta: MetaDict, global_step: int) -> MetaDict:
        if global_step % self.cfg.update_z_every_step == 0:
            return self.init_meta()
        return meta

    def get_goal_meta(self, goal_obs: np.ndarray, obs: np.ndarray) -> MetaDict:
        """Compute skill vector z from current obs to goal obs using phi."""
        assert self.cfg.feature_learner == "hilp"
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.cfg.device)
        goal_t = torch.tensor(goal_obs, dtype=torch.float32).unsqueeze(0).to(self.cfg.device)

        with torch.no_grad():
            z_g = self.feature_learner.feature_net(goal_t)
            z_s = self.feature_learner.feature_net(obs_t)

        z = z_g - z_s
        z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=1)
        z = z.squeeze(0).cpu().numpy()
        meta = OrderedDict()
        meta["z"] = z
        return meta

    def infer_meta_from_obs_and_rewards(
        self, obs: torch.Tensor, reward: torch.Tensor, next_obs: torch.Tensor
    ) -> MetaDict:
        """Infer optimal skill z* from observations and task rewards via least-squares."""
        with torch.no_grad():
            if self.cfg.feature_type == "state":
                phi = self.feature_learner.feature_net(obs)
            elif self.cfg.feature_type == "diff":
                phi = self.feature_learner.feature_net(next_obs) - self.feature_learner.feature_net(obs)
            else:
                phi = torch.cat(
                    [self.feature_learner.feature_net(obs), self.feature_learner.feature_net(next_obs)],
                    dim=-1,
                )
        z = torch.linalg.lstsq(phi, reward).solution
        z = math.sqrt(self.cfg.z_dim) * F.normalize(z, dim=0)
        meta = OrderedDict()
        meta["z"] = z.squeeze().cpu().numpy()
        return meta

    def act(self, obs: np.ndarray, meta: MetaDict, step: int, eval_mode: bool = False) -> np.ndarray:
        obs = torch.as_tensor(obs, device=self.cfg.device, dtype=torch.float32).unsqueeze(0)
        z = torch.as_tensor(meta["z"], device=self.cfg.device, dtype=torch.float32).unsqueeze(0)

        if self.cfg.boltzmann:
            dist = self.actor(obs, z)
        else:
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(obs, z, stddev)

        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()
            if step < self.cfg.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.detach().cpu().numpy()[0]

    def rew_norm(self, reward: torch.Tensor) -> torch.Tensor:
        if self.init_rew_running_mean:
            self.rew_running_mean = reward.mean(dim=0)
            self.rew_running_std = reward.std(dim=0)
            self.init_rew_running_mean = False

        eps = 1e-6
        norm_reward = (reward - self.rew_running_mean) / (self.rew_running_std + eps)

        self.rew_running_mean = 0.995 * self.rew_running_mean + 0.005 * reward.mean(dim=0)
        self.rew_running_std = 0.995 * self.rew_running_std + 0.005 * reward.std(dim=0)
        return norm_reward

    def _get_phi(self, obs: torch.Tensor, next_obs: tp.Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get feature representation based on feature_type."""
        if self.cfg.feature_type == "state":
            return self.feature_learner.feature_net(obs)
        elif self.cfg.feature_type == "diff":
            assert next_obs is not None
            return self.feature_learner.feature_net(next_obs) - self.feature_learner.feature_net(obs)
        else:
            assert next_obs is not None
            return torch.cat(
                [self.feature_learner.feature_net(obs), self.feature_learner.feature_net(next_obs)],
                dim=-1,
            )

    def update_sf(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: tp.Optional[torch.Tensor],
        z: torch.Tensor,
        step: int,
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        with torch.no_grad():
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(next_obs, z, stddev)
            next_action = dist.sample(clip=self.cfg.stddev_clip)

            next_F1, next_F2 = self.successor_target_net(next_obs, z, next_action)
            target_phi = self._get_phi(next_obs, obs).detach()

            next_Q1 = torch.einsum("sd, sd -> s", next_F1, z)
            next_Q2 = torch.einsum("sd, sd -> s", next_F2, z)
            next_F = torch.where((next_Q1 < next_Q2).reshape(-1, 1), next_F1, next_F2)
            target_F = target_phi + discount * next_F
            next_F_disc = discount * next_F

        F1, F2 = self.successor_net(obs, z, action)

        if self.cfg.q_loss:
            Q1 = torch.einsum("sd, sd -> s", F1, z)
            Q2 = torch.einsum("sd, sd -> s", F2, z)
            with torch.no_grad():
                reward = torch.einsum("sd, sd -> s", target_phi, z)
                if self.cfg.use_rew_norm:
                    reward = self.rew_norm(reward)
            target_Q = reward + torch.einsum("sd, sd -> s", next_F_disc, z)
            sf_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        else:
            sf_loss = F.mse_loss(F1, target_F) + F.mse_loss(F2, target_F)

        # Feature learner loss
        if self.cfg.feature_learner == "hilp":
            phi_loss, info = self.feature_learner(
                obs=obs, action=action, next_obs=next_obs, future_obs=future_obs
            )
        else:
            phi_loss = self.feature_learner(
                obs=obs, action=action, next_obs=next_obs, future_obs=future_obs
            )
            info = None

        metrics["sf_loss"] = sf_loss.item()
        if self.cfg.q_loss:
            metrics["reward"] = reward.mean().item()
        metrics["rew_mean"] = self.rew_running_mean.item()
        metrics["rew_std"] = self.rew_running_std.item()
        if phi_loss is not None:
            metrics["phi_loss"] = phi_loss.item()
        if info is not None:
            for key, val in info.items():
                metrics[key] = val.item()

        # Optimize SF
        self.sf_opt.zero_grad(set_to_none=True)
        if self.phi_opt is not None:
            self.phi_opt.zero_grad(set_to_none=True)
            phi_loss.backward(retain_graph=True)
        sf_loss.backward()
        self.sf_opt.step()
        if self.phi_opt is not None:
            self.phi_opt.step()

        return metrics

    def update_sf_with_reward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        future_obs: tp.Optional[torch.Tensor],
        z: torch.Tensor,
        step: int,
        reward: torch.Tensor,
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        with torch.no_grad():
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(next_obs, z, stddev)
            next_action = dist.sample(clip=self.cfg.stddev_clip)

            next_F1, next_F2 = self.successor_target_net(next_obs, z, next_action)
            target_phi = self._get_phi(next_obs, obs).detach()

            next_Q1 = torch.einsum("sd, sd -> s", next_F1, z)
            next_Q2 = torch.einsum("sd, sd -> s", next_F2, z)
            next_F = torch.where((next_Q1 < next_Q2).reshape(-1, 1), next_F1, next_F2)
            target_F = discount * next_F

        F1, F2 = self.successor_net(obs, z, action)
        Q1 = torch.einsum("sd, sd -> s", F1, z)
        Q2 = torch.einsum("sd, sd -> s", F2, z)

        with torch.no_grad():
            if self.cfg.use_rew_norm:
                reward = self.rew_norm(reward)
            target_Q = reward.squeeze() + torch.einsum("sd, sd -> s", target_F, z)
        sf_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        metrics["sf_loss"] = sf_loss.item()
        metrics["reward"] = reward.mean().item()
        metrics["rew_mean"] = self.rew_running_mean.item()
        metrics["rew_std"] = self.rew_running_std.item()

        self.sf_opt.zero_grad(set_to_none=True)
        sf_loss.backward()
        self.sf_opt.step()

        return metrics

    def update_actor(self, obs: torch.Tensor, z: torch.Tensor, step: int) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        if self.cfg.boltzmann:
            dist = self.actor(obs, z)
            action = dist.rsample()
        else:
            stddev = utils.schedule(self.cfg.stddev_schedule, step)
            dist = self.actor(obs, z, stddev)
            action = dist.sample(clip=self.cfg.stddev_clip)

        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        F1, F2 = self.successor_net(obs, z, action)
        Q1 = torch.einsum("sd, sd -> s", F1, z)
        Q2 = torch.einsum("sd, sd -> s", F2, z)
        Q = torch.min(Q1, Q2)
        actor_loss = (
            (self.cfg.temp * log_prob - Q).mean() if self.cfg.boltzmann else -Q.mean()
        )

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        metrics["actor_loss"] = actor_loss.item()
        metrics["actor_logprob"] = log_prob.mean().item()
        return metrics

    def update(
        self,
        replay_loader: ReplayBuffer,
        step: int,
        with_reward: bool = False,
        meta: tp.Optional[MetaDict] = None,
    ) -> tp.Dict[str, float]:
        metrics: tp.Dict[str, float] = {}

        if step % self.cfg.update_every_steps != 0:
            return metrics

        for _ in range(self.cfg.num_sf_updates):
            batch = replay_loader.sample(self.cfg.batch_size)
            batch = batch.to(self.cfg.device)
            obs = batch.obs
            action = batch.action
            discount = batch.discount
            next_obs = batch.next_obs
            future_obs = batch.future_obs

            if meta is None:
                z = self.sample_z(self.cfg.batch_size).to(self.cfg.device)
            else:
                z = torch.from_numpy(meta["z"]).to(self.cfg.device).float()
                z = z.unsqueeze(0).repeat(obs.shape[0], 1)

            # Mix in phi-based skills
            if self.cfg.mix_ratio > 0:
                perm = torch.randperm(self.cfg.batch_size)
                with torch.no_grad():
                    phi = self._get_phi(next_obs[perm], obs[perm])
                cov = torch.matmul(phi.T, phi) / phi.shape[0]
                inv_cov = torch.linalg.pinv(cov)

                mix_idxs: tp.Any = np.where(
                    np.random.uniform(size=self.cfg.batch_size) < self.cfg.mix_ratio
                )[0]
                with torch.no_grad():
                    new_z = phi[mix_idxs]
                new_z = torch.matmul(new_z, inv_cov)
                new_z = math.sqrt(self.cfg.z_dim) * F.normalize(new_z, dim=1)
                z[mix_idxs] = new_z

            if with_reward:
                reward = batch.reward
                met = self.update_sf_with_reward(
                    obs=obs, action=action, discount=discount,
                    next_obs=next_obs, future_obs=future_obs,
                    z=z, step=step, reward=reward,
                )
            else:
                met = self.update_sf(
                    obs=obs, action=action, discount=discount,
                    next_obs=next_obs, future_obs=future_obs,
                    z=z, step=step,
                )
            metrics.update(met)

            # Update actor
            metrics.update(self.update_actor(obs.detach(), z, step))

            # Update target network
            utils.soft_update_params(
                self.successor_net, self.successor_target_net, self.cfg.sf_target_tau
            )

        return metrics

    def update_with_offline_data(
        self,
        replay_loader: ReplayBuffer,
        step: int,
        with_reward: bool = False,
        meta: tp.Optional[MetaDict] = None,
        replay_loader_offline: tp.Optional[ReplayBuffer] = None,
    ) -> tp.Dict[str, float]:
        """Update using mixed online and offline data."""
        metrics: tp.Dict[str, float] = {}

        if step % self.cfg.update_every_steps != 0:
            return metrics

        for _ in range(self.cfg.num_sf_updates):
            half_batch = self.cfg.batch_size // 2
            batch_online = replay_loader.sample(half_batch).to(self.cfg.device)
            batch_offline = replay_loader_offline.sample(half_batch).to(self.cfg.device)

            obs = torch.cat([batch_online.obs, batch_offline.obs], dim=0)
            action = torch.cat([batch_online.action, batch_offline.action], dim=0)
            discount = torch.cat([batch_online.discount, batch_offline.discount], dim=0)
            next_obs = torch.cat([batch_online.next_obs, batch_offline.next_obs], dim=0)

            # Handle future_obs (offline may not have it)
            if batch_online.future_obs is not None and batch_offline.future_obs is not None:
                future_obs = torch.cat([batch_online.future_obs, batch_offline.future_obs], dim=0)
            elif batch_online.future_obs is not None:
                future_obs = torch.cat([batch_online.future_obs, batch_offline.next_obs], dim=0)
            else:
                future_obs = None

            if meta is None:
                z = self.sample_z(self.cfg.batch_size).to(self.cfg.device)
            else:
                z = torch.from_numpy(meta["z"]).to(self.cfg.device).float()
                z = z.unsqueeze(0).repeat(obs.shape[0], 1)

            if self.cfg.mix_ratio > 0:
                perm = torch.randperm(self.cfg.batch_size)
                with torch.no_grad():
                    phi = self._get_phi(next_obs[perm], obs[perm])
                cov = torch.matmul(phi.T, phi) / phi.shape[0]
                inv_cov = torch.linalg.pinv(cov)
                mix_idxs = np.where(
                    np.random.uniform(size=self.cfg.batch_size) < self.cfg.mix_ratio
                )[0]
                with torch.no_grad():
                    new_z = phi[mix_idxs]
                new_z = torch.matmul(new_z, inv_cov)
                new_z = math.sqrt(self.cfg.z_dim) * F.normalize(new_z, dim=1)
                z[mix_idxs] = new_z

            if with_reward:
                reward = torch.cat([batch_online.reward, batch_offline.reward], dim=0)
                met = self.update_sf_with_reward(
                    obs=obs, action=action, discount=discount,
                    next_obs=next_obs, future_obs=future_obs,
                    z=z, step=step, reward=reward,
                )
            else:
                met = self.update_sf(
                    obs=obs, action=action, discount=discount,
                    next_obs=next_obs, future_obs=future_obs,
                    z=z, step=step,
                )
            metrics.update(met)
            metrics.update(self.update_actor(obs.detach(), z, step))
            utils.soft_update_params(
                self.successor_net, self.successor_target_net, self.cfg.sf_target_tau
            )

        return metrics

    def save(self, path: str) -> None:
        """Save agent state to disk."""
        state = {
            "actor": self.actor.state_dict(),
            "successor_net": self.successor_net.state_dict(),
            "successor_target_net": self.successor_target_net.state_dict(),
            "feature_learner": self.feature_learner.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "sf_opt": self.sf_opt.state_dict(),
            "rew_running_mean": self.rew_running_mean,
            "rew_running_std": self.rew_running_std,
            "init_rew_running_mean": self.init_rew_running_mean,
            "cfg": dataclasses.asdict(self.cfg),
        }
        if self.phi_opt is not None:
            state["phi_opt"] = self.phi_opt.state_dict()
        torch.save(state, path)
        logger.info(f"Saved agent to {path}")

    def load(self, path: str) -> None:
        """Load agent state from disk."""
        state = torch.load(path, map_location=self.cfg.device)
        self.actor.load_state_dict(state["actor"])
        self.successor_net.load_state_dict(state["successor_net"])
        self.successor_target_net.load_state_dict(state["successor_target_net"])
        self.feature_learner.load_state_dict(state["feature_learner"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.sf_opt.load_state_dict(state["sf_opt"])
        if self.phi_opt is not None and "phi_opt" in state:
            self.phi_opt.load_state_dict(state["phi_opt"])
        self.rew_running_mean = state.get("rew_running_mean", self.rew_running_mean)
        self.rew_running_std = state.get("rew_running_std", self.rew_running_std)
        self.init_rew_running_mean = state.get("init_rew_running_mean", True)
        logger.info(f"Loaded agent from {path}")
