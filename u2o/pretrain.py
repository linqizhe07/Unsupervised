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


def collect_random_data(
    env: HumanoidEnv,
    replay_buffer: ReplayBuffer,
    num_episodes: int,
    max_episode_steps: int = 1000,
) -> int:
    """Collect random exploration data into replay buffer."""
    total_steps = 0
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        while not done and step < max_episode_steps:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_buffer.add_transition(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=next_obs,
                done=done,
                discount=0.0 if terminated else 1.0,
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
            replay_buffer.add_transition(
                obs=obs,
                action=action,
                reward=0.0,
                next_obs=next_obs,
                done=done,
                discount=0.0 if terminated else 1.0,
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
) -> None:
    """Validate pretraining arguments with actionable error messages."""
    if feature_learner not in FEATURE_LEARNERS:
        choices = ", ".join(sorted(FEATURE_LEARNERS.keys()))
        raise ValueError(
            f"Unsupported feature_learner='{feature_learner}'. Supported: {choices}"
        )
    if collection_episodes <= 0:
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
    z_dim: int = 50,
    hidden_dim: int = 1024,
    phi_hidden_dim: int = 512,
    feature_dim: int = 512,
    feature_learner: str = "hilp",
    hilp_discount: float = 0.98,
    hilp_expectile: float = 0.5,
    lr: float = 1e-4,
    batch_size: int = 1024,
    discount: float = 0.98,
    future: float = 0.99,
    p_randomgoal: float = 0.375,
    collection_episodes: int = 5000,
    pretrain_steps: int = 500000,
    max_episode_steps: int = 500,
    max_buffer_episodes: int = 5000,
    eval_every: int = 500000,
    log_every: int = 1000,
    seed: int = 0,
    device: str = None,
    exploration: str = "random",
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
    )
    device = resolve_device(device)

    os.makedirs(output_dir, exist_ok=True)
    utils.set_seed_everywhere(seed)

    obs_dim = 376
    action_dim = 17

    # Build config dict for logging
    config = {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "z_dim": z_dim,
        "hidden_dim": hidden_dim,
        "phi_hidden_dim": phi_hidden_dim,
        "feature_dim": feature_dim,
        "feature_learner": feature_learner,
        "hilp_discount": hilp_discount,
        "hilp_expectile": hilp_expectile,
        "lr": lr,
        "batch_size": batch_size,
        "discount": discount,
        "future": future,
        "p_randomgoal": p_randomgoal,
        "collection_episodes": collection_episodes,
        "pretrain_steps": pretrain_steps,
        "max_episode_steps": max_episode_steps,
        "max_buffer_episodes": max_buffer_episodes,
        "eval_every": eval_every,
        "log_every": log_every,
        "exploration": exploration,
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
    print(f"  Device: {device}")
    print(f"  z_dim: {z_dim}, feature_learner: {feature_learner}")
    print(f"  Collection episodes: {collection_episodes}")
    print(f"  Pretrain steps: {pretrain_steps}")
    print(f"  Output: {output_dir}")
    print(f"  Wandb: {'enabled' if use_wandb else 'disabled'}")

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        max_episodes=max_buffer_episodes,
        discount=discount,
        future=future,
        p_randomgoal=p_randomgoal,
        max_episode_length=max_episode_steps + 1,
    )

    # ============================================================
    # Phase 1: Data Collection
    # ============================================================
    env = create_dummy_humanoid_env()
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

    print(f"Collected {total_collected} transitions in {len(replay_buffer)} episodes")
    if len(replay_buffer) == 0:
        raise RuntimeError(
            "Replay buffer is empty after data collection. "
            "Increase collection_episodes or check environment rollout."
        )

    # ============================================================
    # Phase 2: Offline Pretraining
    # ============================================================
    print(f"\n--- Phase 2: Offline pretraining for {pretrain_steps} steps ---")

    # Create SFAgent
    cfg = SFAgentConfig(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        lr=lr,
        hidden_dim=hidden_dim,
        phi_hidden_dim=phi_hidden_dim,
        feature_dim=feature_dim,
        z_dim=z_dim,
        batch_size=batch_size,
        feature_learner=feature_learner,
        hilp_discount=hilp_discount,
        hilp_expectile=hilp_expectile,
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
    parser.add_argument("--z_dim", type=int, default=50)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--phi_hidden_dim", type=int, default=512)
    parser.add_argument("--feature_dim", type=int, default=512)
    parser.add_argument("--feature_learner", type=str, default="hilp")
    parser.add_argument("--hilp_discount", type=float, default=0.98)
    parser.add_argument("--hilp_expectile", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--future", type=float, default=0.99)
    parser.add_argument("--p_randomgoal", type=float, default=0.375)
    parser.add_argument("--collection_episodes", type=int, default=5000)
    parser.add_argument("--max_buffer_episodes", type=int, default=5000)
    parser.add_argument("--pretrain_steps", type=int, default=500000)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=500000)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exploration", type=str, default="random", choices=["random", "rnd"])
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()

    pretrain(
        output_dir=args.output_dir,
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        phi_hidden_dim=args.phi_hidden_dim,
        feature_dim=args.feature_dim,
        feature_learner=args.feature_learner,
        hilp_discount=args.hilp_discount,
        hilp_expectile=args.hilp_expectile,
        lr=args.lr,
        batch_size=args.batch_size,
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
    )
