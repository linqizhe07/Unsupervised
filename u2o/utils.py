"""
Utility functions for u2o training.
Ported from u2o_zsrl/url_benchmark/utils.py
"""

import math
import random
import re
import time
import typing as tp

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


class eval_mode:
    """Context manager to temporarily set models to eval mode."""

    def __init__(self, *models) -> None:
        self.models = models
        self.prev_states: tp.List[bool] = []

    def __enter__(self) -> None:
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args: tp.Any) -> None:
        for model, state in zip(self.models, self.prev_states):
            model.train(state)


def set_seed_everywhere(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau) -> None:
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def hard_update_params(net, target_net) -> None:
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(param.data)


def to_torch(xs, device) -> tuple:
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m) -> None:
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def grad_norm(params, norm_type: float = 2.0):
    params = [p for p in params if p.grad is not None]
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type) for p in params]),
        norm_type,
    )
    return total_norm.item()


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()) -> torch.Tensor:
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1) -> None:
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x) -> torch.Tensor:
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x) -> torch.Tensor:
        return x.tanh()

    def _inverse(self, y) -> torch.Tensor:
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y) -> torch.Tensor:
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale) -> None:
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


def schedule(schdl, step) -> float:
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class RMS:
    """Running mean and std."""

    def __init__(self, device, epsilon=1e-4, shape=(1,)) -> None:
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (
            self.S * self.n
            + torch.var(x, dim=0) * bs
            + torch.square(delta) * self.n * bs / (self.n + bs)
        ) / (self.n + bs)
        self.M = new_M
        self.S = new_S
        self.n += bs
        return self.M, self.S


class Timer:
    def __init__(self) -> None:
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self) -> tp.Tuple[float, float]:
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self) -> float:
        return time.time() - self._start_time


class Until:
    def __init__(self, until: tp.Optional[int], action_repeat: int = 1) -> None:
        self.until = until
        self.action_repeat = action_repeat

    def __call__(self, step: int) -> bool:
        if self.until is None:
            return True
        until = self.until // self.action_repeat
        return step < until


class Every:
    def __init__(self, every: tp.Optional[int], action_repeat: int = 1) -> None:
        self.every = every
        self.action_repeat = action_repeat

    def __call__(self, step: int) -> bool:
        if self.every is None:
            return False
        every = self.every // self.action_repeat
        if step % every == 0:
            return True
        return False
