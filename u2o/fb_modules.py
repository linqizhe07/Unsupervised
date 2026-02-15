"""
Network building blocks for u2o: Actor, ForwardMap, BackwardMap, etc.
Ported from u2o_zsrl/url_benchmark/agent/fb_modules.py
"""

import math
import typing as tp

import torch
from torch import nn
import torch.nn.functional as F

from u2o import utils


class OnlineCov(nn.Module):
    """Online covariance tracker with momentum."""

    def __init__(self, mom: float, dim: int) -> None:
        super().__init__()
        self.mom = mom
        self.count = torch.nn.Parameter(torch.LongTensor([0]), requires_grad=False)
        self.cov: tp.Any = torch.nn.Parameter(
            torch.zeros((dim, dim), dtype=torch.float32), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self.count += 1
            self.cov.data *= self.mom
            self.cov.data += (1 - self.mom) * torch.matmul(x.T, x) / x.shape[0]
        count = self.count.item()
        cov = self.cov / (1 - self.mom**count)
        return cov


class _L2(nn.Module):
    """L2 normalization layer scaled by sqrt(dim)."""

    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return math.sqrt(self.dim) * F.normalize(x, dim=1)


def _nl(name: str, dim: int) -> tp.List[nn.Module]:
    """Returns a non-linearity given name and dimension."""
    if name == "irelu":
        return [nn.ReLU(inplace=True)]
    if name == "relu":
        return [nn.ReLU()]
    if name == "ntanh":
        return [nn.LayerNorm(dim), nn.Tanh()]
    if name == "layernorm":
        return [nn.LayerNorm(dim)]
    if name == "tanh":
        return [nn.Tanh()]
    if name == "L2":
        return [_L2(dim)]
    raise ValueError(f"Unknown non-linearity {name}")


def mlp(*layers: tp.Sequence[tp.Union[int, str]]) -> nn.Sequential:
    """Flexible MLP builder.

    Eg: mlp(10, 12, "relu", 15) returns:
    Sequential(Linear(10, 12), ReLU(), Linear(12, 15))
    """
    assert len(layers) >= 2
    sequence: tp.List[nn.Module] = []
    assert isinstance(layers[0], int), "First input must provide the dimension"
    prev_dim: int = layers[0]
    for layer in layers[1:]:
        if isinstance(layer, str):
            sequence.extend(_nl(layer, prev_dim))
        else:
            assert isinstance(layer, int)
            sequence.append(nn.Linear(prev_dim, layer))
            prev_dim = layer
    return nn.Sequential(*sequence)


class Actor(nn.Module):
    """Policy network outputting TruncatedNormal actions, conditioned on (obs, z)."""

    def __init__(
        self,
        obs_dim,
        z_dim,
        action_dim,
        feature_dim,
        hidden_dim,
        preprocess=False,
        add_trunk=True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess

        if self.preprocess:
            self.obs_net = mlp(self.obs_dim, hidden_dim, "ntanh", feature_dim, "irelu")
            self.obs_z_net = mlp(
                self.obs_dim + self.z_dim, hidden_dim, "ntanh", feature_dim, "irelu"
            )
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                feature_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(
                self.obs_dim + self.z_dim,
                hidden_dim,
                "ntanh",
                hidden_dim,
                "irelu",
                hidden_dim,
                "irelu",
            )
            feature_dim = hidden_dim

        self.policy = mlp(feature_dim, hidden_dim, "irelu", self.action_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, z, std):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            obs = self.obs_net(obs)
            h = torch.cat([obs, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z], dim=-1)
        if hasattr(self, "trunk"):
            h = self.trunk(h)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist


class DiagGaussianActor(nn.Module):
    """Alternative Gaussian actor with learned std."""

    def __init__(
        self, obs_dim, z_dim, action_dim, hidden_dim, log_std_bounds, preprocess=False
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.log_std_bounds = log_std_bounds
        self.preprocess = preprocess
        feature_dim = obs_dim + z_dim

        self.policy = mlp(
            feature_dim, hidden_dim, "ntanh", hidden_dim, "relu", 2 * action_dim
        )
        self.apply(utils.weight_init)

    def forward(self, obs, z):
        assert z.shape[-1] == self.z_dim
        h = torch.cat([obs, z], dim=-1)
        mu, log_std = self.policy(h).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()
        dist = utils.SquashedNormal(mu, std)
        return dist


class ForwardMap(nn.Module):
    """Successor feature network: F(s, z, a) -> dual predictions of phi(s')."""

    def __init__(
        self,
        obs_dim,
        z_dim,
        action_dim,
        feature_dim,
        hidden_dim,
        preprocess=False,
        add_trunk=True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.action_dim = action_dim
        self.preprocess = preprocess

        if self.preprocess:
            self.obs_action_net = mlp(
                self.obs_dim + self.action_dim,
                hidden_dim,
                "ntanh",
                feature_dim,
                "irelu",
            )
            self.obs_z_net = mlp(
                self.obs_dim + self.z_dim, hidden_dim, "ntanh", feature_dim, "irelu"
            )
            if not add_trunk:
                self.trunk: nn.Module = nn.Identity()
                feature_dim = 2 * feature_dim
            else:
                self.trunk = mlp(2 * feature_dim, hidden_dim, "irelu")
                feature_dim = hidden_dim
        else:
            self.trunk = mlp(
                self.obs_dim + self.z_dim + self.action_dim,
                hidden_dim,
                "ntanh",
                hidden_dim,
                "irelu",
                hidden_dim,
                "irelu",
            )
            feature_dim = hidden_dim

        seq = [feature_dim, hidden_dim, "irelu", self.z_dim]
        self.F1 = mlp(*seq)
        self.F2 = mlp(*seq)
        self.apply(utils.weight_init)

    def forward(self, obs, z, action):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            obs_action = self.obs_action_net(torch.cat([obs, action], dim=-1))
            obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            h = torch.cat([obs_action, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z, action], dim=-1)
        if hasattr(self, "trunk"):
            h = self.trunk(h)
        F1 = self.F1(h)
        F2 = self.F2(h)
        return F1, F2

    def extract_penultimate_feature(self, obs, z, action):
        assert z.shape[-1] == self.z_dim

        if self.preprocess:
            obs_action = self.obs_action_net(torch.cat([obs, action], dim=-1))
            obs_z = self.obs_z_net(torch.cat([obs, z], dim=-1))
            h = torch.cat([obs_action, obs_z], dim=-1)
        else:
            h = torch.cat([obs, z, action], dim=-1)
        if hasattr(self, "trunk"):
            h = self.trunk(h)

        penultimate_layer = self.F1[:-1]
        penultimate_feature = penultimate_layer(h)
        return penultimate_feature


class BackwardMap(nn.Module):
    """Inverse mapping: obs -> z."""

    def __init__(self, obs_dim, z_dim, hidden_dim, norm_z: bool = True) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.B = mlp(self.obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", self.z_dim)
        self.apply(utils.weight_init)

    def forward(self, obs):
        B = self.B(obs)
        if self.norm_z:
            B = math.sqrt(self.z_dim) * F.normalize(B, dim=1)
        return B
