"""
Feature learners for u2o: HILP (primary), Laplacian, Contrastive, ICM, etc.
Ported from u2o_zsrl/url_benchmark/agent/sf.py feature learner classes.
"""

import typing as tp

import torch
from torch import nn
import torch.nn.functional as F

from u2o import utils
from u2o.fb_modules import mlp


class FeatureLearner(nn.Module):
    """Base class for all feature learners."""

    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__()
        self.feature_net: nn.Module = mlp(
            obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2"
        )
        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs, future_obs):
        return None


class Identity(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.feature_net = nn.Identity()


class HILP(FeatureLearner):
    """HILP: dual phi networks with expectile regression for value learning."""

    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim, cfg) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)

        self.z_dim = z_dim
        self.cfg = cfg

        if self.cfg.feature_type != "concat":
            feature_dim = z_dim
        else:
            assert z_dim % 2 == 0
            feature_dim = z_dim // 2

        layers = [obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", feature_dim]

        self.phi1 = mlp(*layers)
        self.phi2 = mlp(*layers)
        self.target_phi1 = mlp(*layers)
        self.target_phi2 = mlp(*layers)
        self.target_phi1.load_state_dict(self.phi1.state_dict())
        self.target_phi2.load_state_dict(self.phi2.state_dict())

        self.apply(utils.weight_init)

        # Running mean and std for feature normalization
        self.register_buffer("running_mean", torch.zeros(feature_dim))
        self.register_buffer("running_std", torch.ones(feature_dim))

    def feature_net(self, obs):
        phi = self.phi1(obs)
        phi = phi - self.running_mean
        return phi

    def value(self, obs, goals, is_target=False):
        if is_target:
            phi1 = self.target_phi1
            phi2 = self.target_phi2
        else:
            phi1 = self.phi1
            phi2 = self.phi2

        phi1_s = phi1(obs)
        phi1_g = phi1(goals)
        phi2_s = phi2(obs)
        phi2_g = phi2(goals)

        squared_dist1 = ((phi1_s - phi1_g) ** 2).sum(dim=-1)
        v1 = -torch.sqrt(torch.clamp(squared_dist1, min=1e-6))
        squared_dist2 = ((phi2_s - phi2_g) ** 2).sum(dim=-1)
        v2 = -torch.sqrt(torch.clamp(squared_dist2, min=1e-6))

        if is_target:
            v1 = v1.detach()
            v2 = v2.detach()

        return v1, v2

    def expectile_loss(self, adv, diff, expectile=0.7):
        weight = torch.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def forward(self, obs, action, next_obs, future_obs):
        goals = future_obs
        rewards = (torch.linalg.norm(obs - goals, dim=-1) < 1e-6).float()
        masks = 1.0 - rewards
        rewards = rewards - 1.0

        next_v1, next_v2 = self.value(next_obs, goals, is_target=True)
        next_v = torch.minimum(next_v1, next_v2)
        q = rewards + self.cfg.hilp_discount * masks * next_v

        v1_t, v2_t = self.value(obs, goals, is_target=True)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = rewards + self.cfg.hilp_discount * masks * next_v1
        q2 = rewards + self.cfg.hilp_discount * masks * next_v2
        v1, v2 = self.value(obs, goals, is_target=False)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.cfg.hilp_expectile).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.cfg.hilp_expectile).mean()
        value_loss = value_loss1 + value_loss2

        utils.soft_update_params(self.phi1, self.target_phi1, 0.005)
        utils.soft_update_params(self.phi2, self.target_phi2, 0.005)

        with torch.no_grad():
            phi1 = self.phi1(obs)
            self.running_mean = 0.995 * self.running_mean + 0.005 * phi1.mean(dim=0)
            self.running_std = 0.995 * self.running_std + 0.005 * phi1.std(dim=0)

        return value_loss, {
            "hilp/value_loss": value_loss,
            "hilp/v_mean": v.mean(),
            "hilp/v_max": v.max(),
            "hilp/v_min": v.min(),
            "hilp/abs_adv_mean": torch.abs(adv).mean(),
            "hilp/adv_mean": adv.mean(),
            "hilp/adv_max": adv.max(),
            "hilp/adv_min": adv.min(),
            "hilp/accept_prob": (adv >= 0).float().mean(),
        }


class Laplacian(FeatureLearner):
    def forward(self, obs, action, next_obs, future_obs):
        del action, future_obs
        phi = self.feature_net(obs)
        next_phi = self.feature_net(next_obs)
        loss = (phi - next_phi).pow(2).mean()
        Cov = torch.matmul(phi, phi.T)
        I = torch.eye(*Cov.size(), device=Cov.device)
        off_diag = ~I.bool()
        orth_loss_diag = -2 * Cov.diag().mean()
        orth_loss_offdiag = Cov[off_diag].pow(2).mean()
        orth_loss = orth_loss_offdiag + orth_loss_diag
        loss += orth_loss
        return loss


class ContrastiveFeature(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.mu_net = mlp(obs_dim, hidden_dim, "ntanh", hidden_dim, "relu", z_dim, "L2")
        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs, future_obs):
        del action, next_obs
        assert future_obs is not None
        phi = self.feature_net(obs)
        future_mu = self.mu_net(future_obs)
        phi = F.normalize(phi, dim=1)
        future_mu = F.normalize(future_mu, dim=1)
        logits = torch.einsum("sd, td-> st", phi, future_mu)
        I = torch.eye(*logits.size(), device=logits.device)
        off_diag = ~I.bool()
        logits_off_diag = logits[off_diag].reshape(logits.shape[0], logits.shape[0] - 1)
        loss = -logits.diag() + torch.logsumexp(logits_off_diag, dim=1)
        loss = loss.mean()
        return loss


class ICM(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.inverse_dynamic_net = mlp(
            2 * z_dim, hidden_dim, "irelu", hidden_dim, "irelu", action_dim, "tanh"
        )
        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs, future_obs):
        del future_obs
        phi = self.feature_net(obs)
        next_phi = self.feature_net(next_obs)
        predicted_action = self.inverse_dynamic_net(torch.cat([phi, next_phi], dim=-1))
        backward_error = (action - predicted_action).pow(2).mean()
        return backward_error


class TransitionModel(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.forward_dynamic_net = mlp(
            z_dim + action_dim, hidden_dim, "irelu", hidden_dim, "irelu", obs_dim
        )
        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs, future_obs):
        del future_obs
        phi = self.feature_net(obs)
        predicted_next_obs = self.forward_dynamic_net(torch.cat([phi, action], dim=-1))
        forward_error = (predicted_next_obs - next_obs).pow(2).mean()
        return forward_error


class AutoEncoder(FeatureLearner):
    def __init__(self, obs_dim, action_dim, z_dim, hidden_dim) -> None:
        super().__init__(obs_dim, action_dim, z_dim, hidden_dim)
        self.decoder = mlp(z_dim, hidden_dim, "irelu", hidden_dim, "irelu", obs_dim)
        self.apply(utils.weight_init)

    def forward(self, obs, action, next_obs, future_obs):
        del future_obs, next_obs, action
        phi = self.feature_net(obs)
        predicted_obs = self.decoder(phi)
        reconstruction_error = (predicted_obs - obs).pow(2).mean()
        return reconstruction_error


# Registry of all feature learners
FEATURE_LEARNERS = {
    "hilp": HILP,
    "identity": Identity,
    "icm": ICM,
    "transition": TransitionModel,
    "contrastive": ContrastiveFeature,
    "autoencoder": AutoEncoder,
    "lap": Laplacian,
    "random": FeatureLearner,
}
