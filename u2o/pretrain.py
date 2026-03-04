"""
U2O Pretraining Script for Revolve.

Custom training loop using SFAgent (no SB3).
Two-phase: random data collection -> offline pretraining.

Produces:
  1. agent_checkpoint.pt  - pretrained SFAgent state
  2. replay_buffer.npz    - collected transitions
  3. pretrain_config.json  - config for reproducibility

Usage:
  export ROOT_PATH=/path/to/Revolve
  python -m u2o.pretrain --output_dir ./u2o_pretrained --pretrain_steps 1000000
"""

import argparse
import json
import logging
import os
import sys
import tempfile

import numpy as np
import torch

# Ensure ROOT_PATH is on sys.path
root_path = os.environ.get("ROOT_PATH", os.path.dirname(os.path.dirname(__file__)))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

from rl_agent.HumanoidEnv import HumanoidEnv
from u2o.agent import SFAgent, SFAgentConfig, RunningMeanStd
from u2o.networks import FEATURE_LEARNERS
from u2o.replay_buffer import ReplayBuffer
from u2o import utils

logger = logging.getLogger(__name__)

# Dummy reward function for pretraining (returns 0, we use intrinsic reward)
DUMMY_REWARD_FUNC_STR = """
def compute_reward(observation):
    reward = 0.0
    reward_components = {"dummy": 0.0}
    return reward, reward_components
"""


def create_dummy_humanoid_env() -> HumanoidEnv:
    """Create a HumanoidEnv with a dummy reward function for pretraining."""
    tmp_dir = tempfile.mkdtemp()
    return HumanoidEnv(
        reward_func_str=DUMMY_REWARD_FUNC_STR,
        counter=0,
        generation_id=0,
        island_id="pretrain",
        reward_history_file=os.path.join(tmp_dir, "reward_history.json"),
        velocity_file=os.path.join(tmp_dir, "velocity.txt"),
        model_checkpoint_file=os.path.join(tmp_dir, "checkpoint"),
    )


def create_dummy_adroit_env():
    """Create an AdroitHandDoorEnv with a dummy reward function for pretraining."""
    from rl_agent.AdroitEnv import AdroitHandDoorEnv
    tmp_dir = tempfile.mkdtemp()
    reward_fn_path = os.path.join(tmp_dir, "dummy_reward.py")
    with open(reward_fn_path, "w") as f:
        f.write(DUMMY_REWARD_FUNC_STR)
    return AdroitHandDoorEnv(
        reward_fn_path=reward_fn_path,
        counter=0,
        iteration=0,
        group_id="pretrain",
        llm_model="pretrain",
        baseline="pretrain",
        reward_history_file=os.path.join(tmp_dir, "reward_history.json"),
        max_episode_steps=400,
        mode="train",
    )


# -----------------------------------------------------------------------
# Environment registry
# Add new environments here: name -> (obs_dim, action_dim, factory_fn)
# -----------------------------------------------------------------------
ENV_REGISTRY = {
    "humanoid": (376, 17, create_dummy_humanoid_env),
    "adroit_door": (39, 28, create_dummy_adroit_env),
    # "carla": (128, 2, create_dummy_carla_env),  # future: autonomous driving
}


def create_env(env_name: str):
    """Return (env, obs_dim, action_dim) for the given environment name."""
    if env_name not in ENV_REGISTRY:
        choices = ", ".join(ENV_REGISTRY.keys())
        raise ValueError(f"Unknown env '{env_name}'. Supported: {choices}")
    obs_dim, action_dim, factory_fn = ENV_REGISTRY[env_name]
    return factory_fn(), obs_dim, action_dim


# D4RL door dataset registry — no mujoco_py or d4rl package required.
# URLs sourced from the d4rl repository (hand_manipulation_suite/__init__.py).
_D4RL_DATASET_INFO = {
    "door-human-v1":  {
        "url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg_v1/door-human-v1.hdf5",
        "max_episode_steps": 200,
    },
    "door-cloned-v1": {
        "url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg_v1/door-cloned-v1.hdf5",
        "max_episode_steps": 200,
    },
    "door-expert-v1": {
        "url": "http://rail.eecs.berkeley.edu/datasets/offline_rl/hand_dapg_v1/door-expert-v1.hdf5",
        "max_episode_steps": 200,
    },
}


def _get_d4rl_hdf5_path(dataset_name: str) -> str:
    """Return local HDF5 path, downloading from RAIL if not cached."""
    import urllib.request

    cache_dir = os.path.expanduser("~/.d4rl/datasets")
    os.makedirs(cache_dir, exist_ok=True)
    h5_path = os.path.join(cache_dir, f"{dataset_name}.hdf5")

    if not os.path.exists(h5_path):
        if dataset_name not in _D4RL_DATASET_INFO:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Known datasets: {list(_D4RL_DATASET_INFO.keys())}. "
                f"To use a custom dataset, place the HDF5 file at {h5_path} manually."
            )
        url = _D4RL_DATASET_INFO[dataset_name]["url"]
        print(f"[D4RL] Downloading {dataset_name} from {url} ...")
        urllib.request.urlretrieve(url, h5_path)
        print(f"[D4RL] Saved to {h5_path}")

    return h5_path


def load_d4rl_data(
    d4rl_dataset_name: str,
    replay_buffer: ReplayBuffer,
    max_episodes: int = None,
    expected_obs_dim: int = None,
    expected_action_dim: int = None,
) -> int:
    """Load a D4RL offline dataset into the replay buffer via h5py.

    Does NOT require the d4rl or mujoco_py packages.
    Downloads the HDF5 file on first use and caches it in ~/.d4rl/datasets/.

    Args:
        d4rl_dataset_name: D4RL dataset id, e.g. 'door-human-v1',
                           'door-cloned-v1', 'door-expert-v1'.
        replay_buffer:     ReplayBuffer to populate.
        max_episodes:      Optional cap on number of episodes to load.
        expected_obs_dim:  Optional observation dimension check.
        expected_action_dim: Optional action dimension check.

    Returns:
        Total number of transitions loaded.
    """
    import h5py

    h5_path = _get_d4rl_hdf5_path(d4rl_dataset_name)
    max_episode_steps = _D4RL_DATASET_INFO.get(d4rl_dataset_name, {}).get("max_episode_steps")

    with h5py.File(h5_path, "r") as f:
        observations = f["observations"][:].astype(np.float32)
        actions = f["actions"][:].astype(np.float32)
        rewards = f["rewards"][:].astype(np.float32)
        terminals = f["terminals"][:].astype(bool)
        next_obs_data = f["next_observations"][:].astype(np.float32) if "next_observations" in f else None
        timeouts = f["timeouts"][:].astype(bool) if "timeouts" in f else None

    actions = np.clip(actions, -(1 - 1e-5), 1 - 1e-5)

    if expected_obs_dim is not None and observations.shape[-1] != expected_obs_dim:
        raise ValueError(
            f"D4RL dataset '{d4rl_dataset_name}' has obs_dim={observations.shape[-1]}, "
            f"but env expects obs_dim={expected_obs_dim}."
        )
    if expected_action_dim is not None and actions.shape[-1] != expected_action_dim:
        raise ValueError(
            f"D4RL dataset '{d4rl_dataset_name}' has action_dim={actions.shape[-1]}, "
            f"but env expects action_dim={expected_action_dim}."
        )

    n = len(observations)
    total_steps = 0
    episode_count = 0
    episode_step = 0

    # Process all n transitions. When next_observations is absent, the last transition
    # (i = n-1) has no valid next_obs so we stop at n-1 in that case.
    n_iter = n if next_obs_data is not None else n - 1

    # For Adroit (39-dim obs), approximate joint_velocities via finite difference.
    # obs layout: qpos[1:-2](27), latch(1), door_hinge(1), palm(3), handle(3), delta(3), door_open(1)
    # qvel layout: 30 dims matching qpos (slide + 27 hand + latch + door_hinge)
    is_adroit = expected_obs_dim == 39
    # Adroit frame_skip=5, MuJoCo default timestep=0.002 → dt=0.01
    adroit_dt = 0.01

    for i in range(n_iter):
        done = bool(terminals[i])
        timeout = bool(timeouts[i]) if timeouts is not None else (
            max_episode_steps is not None and episode_step == max_episode_steps - 1
        )
        next_obs = next_obs_data[i] if next_obs_data is not None else observations[i + 1]
        episode_done = done or timeout

        extras = {}
        if is_adroit:
            # Approximate joint velocities from consecutive observations.
            # Recovers 29/30 DOF; qvel[0] (ARTz slide) is not in obs.
            approx_vel = np.zeros(30, dtype=np.float32)
            approx_vel[1:28] = (next_obs[:27] - observations[i][:27]) / adroit_dt
            approx_vel[28] = (next_obs[28] - observations[i][28]) / adroit_dt  # door hinge
            approx_vel[29] = (next_obs[27] - observations[i][27]) / adroit_dt  # latch
            extras["joint_velocities"] = approx_vel
            extras["joint_forces"] = np.zeros(28, dtype=np.float32)

        # Bug fix: timeout is NOT a true terminal — use discount=1.0 for timeouts so
        # the bootstrapped TD target is not incorrectly zeroed out.
        replay_buffer.add_transition(
            obs=observations[i],
            action=actions[i],
            reward=float(rewards[i]),
            next_obs=next_obs,
            done=episode_done,
            discount=0.0 if done else 1.0,
            **extras,
        )
        total_steps += 1

        if episode_done:
            episode_count += 1
            episode_step = 0
            if max_episodes is not None and episode_count >= max_episodes:
                break
        else:
            episode_step += 1

    # Store last partial episode (dataset may not end with terminal=True)
    if replay_buffer._current_episode:
        replay_buffer._store_episode()
        episode_count += 1

    print(
        f"[D4RL] Loaded {d4rl_dataset_name}: "
        f"{episode_count} episodes, {total_steps} transitions"
    )
    return total_steps


def collect_random_data(
    env: HumanoidEnv,
    replay_buffer: ReplayBuffer,
    num_episodes: int,
    max_episode_steps: int = 1000,
) -> int:
    """Collect random exploration data into replay buffer."""
    # Detect Adroit env to capture joint_velocities/joint_forces from MuJoCo state.
    is_adroit = hasattr(env, "data") and env.observation_space.shape[0] == 39
    total_steps = 0
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        while not done and step < max_episode_steps:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            extras = {}
            if is_adroit:
                extras["joint_velocities"] = env.data.qvel.ravel().copy().astype(np.float32)
                extras["joint_forces"] = env.data.actuator_force.ravel().copy().astype(np.float32)
            replay_buffer.add_transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                discount=0.0 if terminated else 1.0,
                **extras,
            )
            obs = next_obs
            step += 1
            total_steps += 1

        # Force store if episode didn't end naturally
        if not done:
            replay_buffer._store_episode()

        if (ep + 1) % 100 == 0:
            logger.info(f"Collected {ep + 1}/{num_episodes} episodes, {total_steps} total steps")

    return total_steps


def collect_rnd_data(
    env: HumanoidEnv,
    replay_buffer: ReplayBuffer,
    num_episodes: int,
    max_episode_steps: int = 500,
    device: str = "cpu",
    rnd_lr: float = 1e-3,
    actor_lr: float = 3e-4,
    random_warmup_episodes: int = 50,
    actor_update_every: int = 10,
) -> int:
    """Collect exploration data guided by RND intrinsic rewards.

    Uses a lightweight REINFORCE actor trained to maximize RND novelty.
    The RND predictor is updated after each episode so visited states
    become less novel, pushing the actor toward unexplored regions.
    """
    from u2o.networks import RNDNetwork, ExplorationActor

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    is_adroit = hasattr(env, "data") and obs_dim == 39

    rnd = RNDNetwork(obs_dim, embedding_dim=128, hidden_dim=256).to(device)
    rnd_opt = torch.optim.Adam(rnd.predictor.parameters(), lr=rnd_lr)

    actor = ExplorationActor(obs_dim, action_dim, hidden_dim=256).to(device)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=actor_lr)

    obs_rms = RunningMeanStd(obs_dim, device=device)
    rew_rms = RunningMeanStd(1, device=device)

    total_steps = 0
    recent_intrinsic = []

    def safe_normalize(obs_t: torch.Tensor) -> torch.Tensor:
        """Normalize obs with var clamped to prevent explosion on near-constant dims."""
        safe_std = torch.sqrt(torch.clamp(obs_rms.var, min=1e-2)) + 1e-8
        return (obs_t - obs_rms.mean) / safe_std

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        ep_raw_obs = []        # raw obs tensors for batch obs_rms update
        ep_obs_list = []       # detached normalized obs for RND update
        ep_transitions = []    # (norm_obs, action_pre_clip) for REINFORCE
        ep_intrinsic = []

        while not done and step < max_episode_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            ep_raw_obs.append(obs_t.squeeze(0))
            norm_obs = safe_normalize(obs_t)

            if ep < random_warmup_episodes:
                action = env.action_space.sample()
                is_actor = False
            else:
                with torch.no_grad():
                    dist = actor(norm_obs)
                    action_t = dist.sample()
                    action_pre_clip = action_t.squeeze(0).cpu().numpy()
                    action = np.clip(action_pre_clip, -1.0, 1.0)
                is_actor = True

            # NaN safety: fall back to random action
            if np.any(np.isnan(action)):
                action = env.action_space.sample()
                is_actor = False

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            with torch.no_grad():
                intrinsic_r = rnd.intrinsic_reward(norm_obs).item()
                if np.isnan(intrinsic_r):
                    intrinsic_r = 0.0

            # Store env reward (0.0) in buffer, not intrinsic reward
            extras = {}
            if is_adroit:
                extras["joint_velocities"] = env.data.qvel.ravel().copy().astype(np.float32)
                extras["joint_forces"] = env.data.actuator_force.ravel().copy().astype(np.float32)
            replay_buffer.add_transition(
                obs=obs,
                action=action,
                reward=0.0,
                next_obs=next_obs,
                done=done,
                discount=0.0 if terminated else 1.0,
                **extras,
            )

            norm_obs_detached = norm_obs.squeeze(0).detach()
            ep_obs_list.append(norm_obs_detached)
            if is_actor:
                # Store pre-clip action for correct log_prob recompute
                ep_transitions.append((
                    norm_obs_detached,
                    torch.tensor(action_pre_clip, dtype=torch.float32, device=device),
                ))
            ep_intrinsic.append(intrinsic_r)

            obs = next_obs
            step += 1
            total_steps += 1

        if not done:
            replay_buffer._store_episode()

        # Batch update obs_rms once per episode (stable normalization)
        if ep_raw_obs:
            obs_rms.update(torch.stack(ep_raw_obs))

        # Update RND predictor on episode observations
        if ep_obs_list:
            obs_batch = torch.stack(ep_obs_list)
            rnd_loss = rnd.loss(obs_batch)
            rnd_opt.zero_grad()
            rnd_loss.backward()
            rnd_opt.step()

        # REINFORCE update on actor (after warmup, every K episodes)
        # Recompute log_probs with fresh graph instead of storing stale ones
        if (
            ep >= random_warmup_episodes
            and (ep - random_warmup_episodes) % actor_update_every == 0
            and len(ep_transitions) >= 2
        ):
            n = len(ep_transitions)
            rewards_t = torch.tensor(ep_intrinsic[-n:], device=device)
            rew_rms.update(rewards_t.unsqueeze(-1))
            rewards_t = (rewards_t - rew_rms.mean.squeeze()) / (
                torch.sqrt(torch.clamp(rew_rms.var.squeeze(), min=1e-6)) + 1e-8
            )
            # Discounted returns
            returns = torch.zeros_like(rewards_t)
            G = 0.0
            for i in range(n - 1, -1, -1):
                G = rewards_t[i].item() + 0.99 * G
                returns[i] = G
            ret_std = returns.std()
            if ret_std > 1e-8:
                returns = (returns - returns.mean()) / ret_std
            else:
                returns = returns - returns.mean()

            # Recompute log_probs with current actor (fresh computation graph)
            # Use pre-clip actions so log_prob matches the actual sampled values
            obs_batch = torch.stack([t[0] for t in ep_transitions])
            act_batch = torch.stack([t[1] for t in ep_transitions])
            dist = actor(obs_batch)
            log_probs = dist.log_prob(act_batch).sum(-1)

            policy_loss = -(log_probs * returns.detach()).mean()
            actor_opt.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
            actor_opt.step()

        recent_intrinsic.append(np.mean(ep_intrinsic) if ep_intrinsic else 0)

        if (ep + 1) % 100 == 0:
            avg_r = np.mean(recent_intrinsic[-100:])
            print(
                f"[RND] Episode {ep + 1}/{num_episodes} | "
                f"Steps: {total_steps} | Avg intrinsic reward: {avg_r:.4f}"
            )

    return total_steps


def resolve_device(device: str = None) -> str:
    """Resolve requested device to an available torch device string."""
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if device is None or device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"

    try:
        parsed = torch.device(device)
    except Exception as exc:
        raise ValueError(f"Invalid device string: {device}") from exc

    if parsed.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return "cpu"
    if parsed.type == "mps" and not mps_available:
        logger.warning("MPS requested but not available. Falling back to CPU.")
        return "cpu"

    return str(parsed)


def validate_pretrain_args(
    feature_learner: str,
    collection_episodes: int,
    pretrain_steps: int,
    max_episode_steps: int,
    max_buffer_episodes: int,
    batch_size: int,
    eval_every: int,
    log_every: int,
    future: float,
    p_randomgoal: float,
    exploration: str = "random",
) -> None:
    """Validate pretraining arguments with actionable error messages."""
    if feature_learner not in FEATURE_LEARNERS:
        choices = ", ".join(sorted(FEATURE_LEARNERS.keys()))
        raise ValueError(
            f"Unsupported feature_learner='{feature_learner}'. Supported: {choices}"
        )
    if exploration != "d4rl" and collection_episodes <= 0:
        raise ValueError("collection_episodes must be > 0.")
    if pretrain_steps < 0:
        raise ValueError("pretrain_steps must be >= 0.")
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be > 0.")
    if max_buffer_episodes <= 0:
        raise ValueError("max_buffer_episodes must be > 0.")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if eval_every <= 0:
        raise ValueError("eval_every must be > 0.")
    if log_every <= 0:
        raise ValueError("log_every must be > 0.")
    if not (0.0 <= future <= 1.0):
        raise ValueError("future must be within [0.0, 1.0].")
    if feature_learner == "hilp" and future >= 1.0:
        raise ValueError("HILP requires future < 1.0 so future goals can be sampled.")
    if not (0.0 <= p_randomgoal <= 1.0):
        raise ValueError("p_randomgoal must be within [0.0, 1.0].")


def pretrain(
    output_dir: str,
    env_name: str = "humanoid",
    z_dim: int = 50,
    hidden_dim: int = 1024,
    phi_hidden_dim: int = 512,
    feature_dim: int = 512,
    feature_learner: str = "hilp",
    hilp_discount: float = 0.98,
    hilp_expectile: float = 0.5,
    lr: float = 1e-4,
    lr_coef: float = 5.0,
    batch_size: int = 1024,
    discount: float = 0.98,
    future: float = 0.99,
    p_randomgoal: float = 0.375,
    sf_target_tau: float = 0.01,
    mix_ratio: float = 0.5,
    q_loss: bool = True,
    feature_type: str = "state",
    stddev_schedule: str = "0.2",
    stddev_clip: float = 0.3,
    use_rew_norm: bool = True,
    boltzmann: bool = False,
    temp: float = 1.0,
    num_sf_updates: int = 1,
    update_every_steps: int = 1,
    update_z_every_step: int = 300,
    update_cov_every_step: int = 1000,
    num_expl_steps: int = 2000,
    preprocess: bool = True,
    add_trunk: bool = False,
    collection_episodes: int = 10000,
    pretrain_steps: int = 1000000,
    max_episode_steps: int = 500,
    max_buffer_episodes: int = 10000,
    eval_every: int = 500000,
    log_every: int = 1000,
    seed: int = 0,
    device: str = None,
    exploration: str = "random",
    d4rl_dataset: str = None,
    wandb_project: str = None,
    wandb_entity: str = None,
    wandb_run_name: str = None,
):
    """
    Complete U2O pretraining pipeline with custom training loop.

    Phase 1: Data collection (random or RND-guided)
    Phase 2: Offline pretraining (train SFAgent on collected data)
    """
    validate_pretrain_args(
        feature_learner=feature_learner,
        collection_episodes=collection_episodes,
        pretrain_steps=pretrain_steps,
        max_episode_steps=max_episode_steps,
        max_buffer_episodes=max_buffer_episodes,
        batch_size=batch_size,
        eval_every=eval_every,
        log_every=log_every,
        future=future,
        p_randomgoal=p_randomgoal,
        exploration=exploration,
    )
    if exploration == "d4rl" and not d4rl_dataset:
        raise ValueError("--d4rl_dataset is required when --exploration d4rl is set.")
    # Parse comma-separated dataset names
    d4rl_datasets = [s.strip() for s in d4rl_dataset.split(",")] if d4rl_dataset else []
    device = resolve_device(device)

    os.makedirs(output_dir, exist_ok=True)
    utils.set_seed_everywhere(seed)

    if env_name not in ENV_REGISTRY:
        choices = ", ".join(ENV_REGISTRY.keys())
        raise ValueError(f"Unknown env '{env_name}'. Supported: {choices}")
    obs_dim, action_dim, _ = ENV_REGISTRY[env_name]

    # Build config dict for logging
    config = {
        "env_name": env_name,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        # architecture
        "z_dim": z_dim,
        "hidden_dim": hidden_dim,
        "phi_hidden_dim": phi_hidden_dim,
        "feature_dim": feature_dim,
        "feature_learner": feature_learner,
        "preprocess": preprocess,
        "add_trunk": add_trunk,
        # HILP
        "hilp_discount": hilp_discount,
        "hilp_expectile": hilp_expectile,
        "feature_type": feature_type,
        # optimizer
        "lr": lr,
        "lr_coef": lr_coef,
        # training loop
        "batch_size": batch_size,
        "num_sf_updates": num_sf_updates,
        "update_every_steps": update_every_steps,
        "update_z_every_step": update_z_every_step,
        "update_cov_every_step": update_cov_every_step,
        "num_expl_steps": num_expl_steps,
        # SF-specific
        "sf_target_tau": sf_target_tau,
        "mix_ratio": mix_ratio,
        "q_loss": q_loss,
        "use_rew_norm": use_rew_norm,
        # actor
        "stddev_schedule": stddev_schedule,
        "stddev_clip": stddev_clip,
        "boltzmann": boltzmann,
        "temp": temp,
        # replay buffer
        "discount": discount,
        "future": future,
        "p_randomgoal": p_randomgoal,
        # collection
        "collection_episodes": collection_episodes,
        "pretrain_steps": pretrain_steps,
        "max_episode_steps": max_episode_steps,
        "max_buffer_episodes": max_buffer_episodes,
        "eval_every": eval_every,
        "log_every": log_every,
        "exploration": exploration,
        "d4rl_dataset": d4rl_dataset,  # original comma-separated string kept for logging
        "device": device,
        "seed": seed,
    }

    # Initialize wandb
    wandb = None
    use_wandb = wandb_project is not None
    if use_wandb:
        try:
            import wandb  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "wandb_project was provided, but wandb is not installed. "
                "Install wandb or omit --wandb_project."
            ) from exc
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name or f"u2o_pretrain_z{z_dim}_{feature_learner}_s{seed}",
            config=config,
        )

    print(f"=== U2O Pretraining ===")
    print(f"  Env: {env_name} (obs_dim={obs_dim}, action_dim={action_dim})")
    print(f"  Device: {device}")
    print(f"  z_dim: {z_dim}, feature_learner: {feature_learner}")
    if exploration == "d4rl":
        print(f"  Data source: D4RL datasets {d4rl_datasets} (GCRL pipeline)")
    else:
        print(f"  Collection episodes: {collection_episodes}")
    print(f"  Pretrain steps: {pretrain_steps}")
    print(f"  Output: {output_dir}")
    print(f"  Wandb: {'enabled' if use_wandb else 'disabled'}")

    # Create replay buffer
    replay_episode_length = max_episode_steps + 1
    if exploration == "d4rl" and d4rl_datasets:
        for _ds in d4rl_datasets:
            dataset_horizon = _D4RL_DATASET_INFO.get(_ds, {}).get("max_episode_steps")
            if dataset_horizon is not None:
                replay_episode_length = max(replay_episode_length, int(dataset_horizon) + 1)

    replay_buffer = ReplayBuffer(
        max_episodes=max_buffer_episodes,
        discount=discount,
        future=future,
        p_randomgoal=p_randomgoal,
        max_episode_length=replay_episode_length,
    )

    # ============================================================
    # Phase 1: Data Collection
    # ============================================================
    if exploration == "d4rl":
        # GCRL pipeline: load D4RL offline datasets directly (no env needed).
        #
        # Datasets are loaded in the order specified. The circular buffer naturally
        # handles overflow: when total episodes across all datasets exceeds
        # max_buffer_episodes, the oldest episodes (from the first datasets) are
        # overwritten by the newest ones.
        #
        # Recommended order for door tasks:
        #   door-cloned-v1,door-expert-v1,door-human-v1
        # This way the small but high-quality human/expert trajectories are loaded
        # last and are guaranteed to remain in the buffer, while cloned provides the
        # bulk of state-space coverage. Equal-per-dataset caps are NOT used because
        # door-human-v1 has only ~25 episodes; capping cloned at 3333 wastes buffer
        # capacity without improving the sampling distribution.
        num_datasets = len(d4rl_datasets)
        print(f"\n--- Phase 1: Loading {num_datasets} D4RL dataset(s) (GCRL pipeline) ---")
        total_collected = 0
        for i, ds_name in enumerate(d4rl_datasets):
            print(f"  [{i+1}/{num_datasets}] {ds_name}  (up to {max_buffer_episodes} episodes)")
            total_collected += load_d4rl_data(
                d4rl_dataset_name=ds_name,
                replay_buffer=replay_buffer,
                max_episodes=max_buffer_episodes,
                expected_obs_dim=obs_dim,
                expected_action_dim=action_dim,
            )
    else:
        env, _, _ = create_env(env_name)
        try:
            if exploration == "rnd":
                print(
                    f"\n--- Phase 1: Collecting {collection_episodes} episodes with RND exploration ---"
                )
                total_collected = collect_rnd_data(
                    env, replay_buffer, collection_episodes, max_episode_steps, device=device
                )
            else:
                print(f"\n--- Phase 1: Collecting {collection_episodes} episodes of random data ---")
                total_collected = collect_random_data(
                    env, replay_buffer, collection_episodes, max_episode_steps
                )
        finally:
            env.close()

    print(f"Loaded {total_collected} transitions in {len(replay_buffer)} episodes")
    if len(replay_buffer) == 0:
        raise RuntimeError(
            "Replay buffer is empty after data collection. "
            "Increase collection_episodes or check environment rollout."
        )

    # ============================================================
    # Phase 2: Offline Pretraining
    # ============================================================
    print(f"\n--- Phase 2: Offline pretraining for {pretrain_steps} steps ---")

    # Create SFAgent with full config
    cfg = SFAgentConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        z_dim=z_dim,
        hidden_dim=hidden_dim,
        phi_hidden_dim=phi_hidden_dim,
        feature_dim=feature_dim,
        feature_learner=feature_learner,
        preprocess=preprocess,
        add_trunk=add_trunk,
        hilp_discount=hilp_discount,
        hilp_expectile=hilp_expectile,
        feature_type=feature_type,
        lr=lr,
        lr_coef=lr_coef,
        batch_size=batch_size,
        num_sf_updates=num_sf_updates,
        update_every_steps=update_every_steps,
        update_z_every_step=update_z_every_step,
        update_cov_every_step=update_cov_every_step,
        num_expl_steps=num_expl_steps,
        sf_target_tau=sf_target_tau,
        mix_ratio=mix_ratio,
        q_loss=q_loss,
        use_rew_norm=use_rew_norm,
        stddev_schedule=stddev_schedule,
        stddev_clip=stddev_clip,
        boltzmann=boltzmann,
        temp=temp,
    )
    agent = SFAgent(cfg)

    timer = utils.Timer()

    process = None
    try:
        import psutil

        process = psutil.Process()
    except ImportError:
        logger.warning("psutil not installed; memory usage logging will be skipped.")

    for step in range(1, pretrain_steps + 1):
        # Update agent (unsupervised, no task reward)
        metrics = agent.update(replay_buffer, step, with_reward=False)

        # Logging
        if step % log_every == 0 and metrics:
            elapsed, total = timer.reset()
            log_str = f"Step {step}/{pretrain_steps}"
            log_str += f" | Time: {elapsed:.1f}s (total: {total:.0f}s)"
            for key in ["sf_loss", "phi_loss", "actor_loss", "reward"]:
                if key in metrics:
                    log_str += f" | {key}: {metrics[key]:.4f}"
            # Memory monitoring
            if process is not None:
                rss_mb = process.memory_info().rss / 1024 / 1024
                log_str += f" | RAM: {rss_mb:.0f}MB"
            if torch.cuda.is_available():
                gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
                log_str += f" | GPU: {gpu_mb:.0f}MB"
            print(log_str)

            if use_wandb:
                wandb.log(metrics, step=step)

        # Periodic checkpoint
        if step % eval_every == 0:
            ckpt_path = os.path.join(output_dir, f"agent_step_{step}.pt")
            agent.save(ckpt_path)

    # ============================================================
    # Save final artifacts
    # ============================================================
    print(f"\n--- Saving final artifacts ---")

    # Save agent
    agent_path = os.path.join(output_dir, "agent_checkpoint.pt")
    agent.save(agent_path)
    print(f"Saved agent to {agent_path}")

    # Save replay buffer
    buffer_path = os.path.join(output_dir, "replay_buffer.npz")
    replay_buffer.save(buffer_path)
    print(f"Saved replay buffer to {buffer_path}")

    # Save config
    config_path = os.path.join(output_dir, "pretrain_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    if use_wandb:
        wandb.finish()

    print(f"\n=== Pretraining complete! Total time: {timer.total_time():.0f}s ===")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="U2O Pretraining for Revolve")
    parser.add_argument("--output_dir", type=str, default="./u2o_pretrained")
    parser.add_argument("--env", type=str, default="humanoid", choices=list(ENV_REGISTRY.keys()),
                        help="Environment to pretrain on. Supported: " + ", ".join(ENV_REGISTRY.keys()))
    # architecture
    parser.add_argument("--z_dim", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--phi_hidden_dim", type=int, default=512)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--feature_learner", type=str, default="hilp")
    parser.add_argument("--preprocess", type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--add_trunk", type=lambda x: x.lower() != "false", default=False)
    # HILP
    parser.add_argument("--hilp_discount", type=float, default=0.98)
    parser.add_argument("--hilp_expectile", type=float, default=0.5)
    parser.add_argument("--feature_type", type=str, default="state", choices=["state", "diff", "concat"])
    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_coef", type=float, default=5.0)
    # training loop
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_sf_updates", type=int, default=1)
    parser.add_argument("--update_every_steps", type=int, default=1)
    parser.add_argument("--update_z_every_step", type=int, default=300)
    parser.add_argument("--update_cov_every_step", type=int, default=1000)
    parser.add_argument("--num_expl_steps", type=int, default=2000)
    # SF-specific
    parser.add_argument("--sf_target_tau", type=float, default=0.01)
    parser.add_argument("--mix_ratio", type=float, default=0.5)
    parser.add_argument("--q_loss", type=lambda x: x.lower() != "false", default=True)
    parser.add_argument("--use_rew_norm", type=lambda x: x.lower() != "false", default=True)
    # actor
    parser.add_argument("--stddev_schedule", type=str, default="0.2")
    parser.add_argument("--stddev_clip", type=float, default=0.3)
    parser.add_argument("--boltzmann", type=lambda x: x.lower() != "false", default=False)
    parser.add_argument("--temp", type=float, default=1.0)
    # replay buffer
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--future", type=float, default=0.99)
    parser.add_argument("--p_randomgoal", type=float, default=0.375)
    # collection
    parser.add_argument("--collection_episodes", type=int, default=10000)
    parser.add_argument("--max_buffer_episodes", type=int, default=10000)
    parser.add_argument("--pretrain_steps", type=int, default=1000000)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=500000)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exploration", type=str, default="random", choices=["random", "rnd", "d4rl"])
    parser.add_argument("--d4rl_dataset", type=str, default=None,
                        help="Comma-separated D4RL dataset names for GCRL pipeline. "
                             "E.g. 'door-human-v1,door-cloned-v1,door-expert-v1'. "
                             "Buffer capacity is split equally across datasets.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    pretrain(
        output_dir=args.output_dir,
        env_name=args.env,
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        phi_hidden_dim=args.phi_hidden_dim,
        feature_dim=args.feature_dim,
        feature_learner=args.feature_learner,
        preprocess=args.preprocess,
        add_trunk=args.add_trunk,
        hilp_discount=args.hilp_discount,
        hilp_expectile=args.hilp_expectile,
        feature_type=args.feature_type,
        lr=args.lr,
        lr_coef=args.lr_coef,
        batch_size=args.batch_size,
        num_sf_updates=args.num_sf_updates,
        update_every_steps=args.update_every_steps,
        update_z_every_step=args.update_z_every_step,
        update_cov_every_step=args.update_cov_every_step,
        num_expl_steps=args.num_expl_steps,
        sf_target_tau=args.sf_target_tau,
        mix_ratio=args.mix_ratio,
        q_loss=args.q_loss,
        use_rew_norm=args.use_rew_norm,
        stddev_schedule=args.stddev_schedule,
        stddev_clip=args.stddev_clip,
        boltzmann=args.boltzmann,
        temp=args.temp,
        discount=args.discount,
        future=args.future,
        p_randomgoal=args.p_randomgoal,
        collection_episodes=args.collection_episodes,
        max_buffer_episodes=args.max_buffer_episodes,
        pretrain_steps=args.pretrain_steps,
        max_episode_steps=args.max_episode_steps,
        eval_every=args.eval_every,
        log_every=args.log_every,
        exploration=args.exploration,
        seed=args.seed,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        d4rl_dataset=args.d4rl_dataset,
    )
