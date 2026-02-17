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

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        ep_obs_list = []
        ep_logprobs = []
        ep_intrinsic = []

        while not done and step < max_episode_steps:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            obs_rms.update(obs_t)
            norm_obs = obs_rms.normalize(obs_t)

            if ep < random_warmup_episodes:
                action = env.action_space.sample()
                log_prob = None
            else:
                with torch.no_grad():
                    dist = actor(norm_obs)
                    action_t = dist.sample()
                    log_prob = dist.log_prob(action_t).sum(-1)
                    action = action_t.squeeze(0).cpu().numpy()
                    action = np.clip(action, -1.0, 1.0)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            with torch.no_grad():
                intrinsic_r = rnd.intrinsic_reward(norm_obs).item()

            replay_buffer.add_transition(
                obs=obs,
                action=action,
                reward=intrinsic_r,
                next_obs=next_obs,
                done=done,
                discount=0.0 if terminated else 1.0,
            )

            ep_obs_list.append(norm_obs.squeeze(0))
            if log_prob is not None:
                ep_logprobs.append(log_prob)
            ep_intrinsic.append(intrinsic_r)

            obs = next_obs
            step += 1
            total_steps += 1

        if not done:
            replay_buffer._store_episode()

        # Update RND predictor on episode observations
        if ep_obs_list:
            obs_batch = torch.stack(ep_obs_list)
            rnd_loss = rnd.loss(obs_batch)
            rnd_opt.zero_grad()
            rnd_loss.backward()
            rnd_opt.step()

        # REINFORCE update on actor (after warmup, every K episodes)
        if (
            ep >= random_warmup_episodes
            and (ep - random_warmup_episodes) % actor_update_every == 0
            and ep_logprobs
        ):
            rewards_t = torch.tensor(ep_intrinsic[-len(ep_logprobs):], device=device)
            rew_rms.update(rewards_t.unsqueeze(-1))
            rewards_t = (rewards_t - rew_rms.mean.squeeze()) / (
                torch.sqrt(rew_rms.var.squeeze()) + 1e-8
            )
            # Discounted returns
            returns = torch.zeros_like(rewards_t)
            G = 0.0
            for i in range(len(rewards_t) - 1, -1, -1):
                G = rewards_t[i].item() + 0.99 * G
                returns[i] = G
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            log_probs = torch.stack(ep_logprobs).squeeze()
            policy_loss = -(log_probs * returns).mean()
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
    collection_episodes: int = 2000,
    pretrain_steps: int = 500000,
    max_episode_steps: int = 500,
    max_buffer_episodes: int = 2000,
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
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
        "exploration": exploration,
        "seed": seed,
    }

    # Initialize wandb
    use_wandb = wandb_project is not None
    if use_wandb:
        import wandb
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
    if exploration == "rnd":
        print(f"\n--- Phase 1: Collecting {collection_episodes} episodes with RND exploration ---")
        total_collected = collect_rnd_data(
            env, replay_buffer, collection_episodes, max_episode_steps, device=device
        )
    else:
        print(f"\n--- Phase 1: Collecting {collection_episodes} episodes of random data ---")
        total_collected = collect_random_data(
            env, replay_buffer, collection_episodes, max_episode_steps
        )
    print(f"Collected {total_collected} transitions in {len(replay_buffer)} episodes")

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

    import psutil
    import gc
    process = psutil.Process()

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
            rss_mb = process.memory_info().rss / 1024 / 1024
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            log_str += f" | RAM: {rss_mb:.0f}MB | GPU: {gpu_mb:.0f}MB"
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
    parser.add_argument("--collection_episodes", type=int, default=2000)
    parser.add_argument("--pretrain_steps", type=int, default=500000)
    parser.add_argument("--max_episode_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exploration", type=str, default="random", choices=["random", "rnd"])
    parser.add_argument("--device", type=str, default=None)
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
        collection_episodes=args.collection_episodes,
        pretrain_steps=args.pretrain_steps,
        max_episode_steps=args.max_episode_steps,
        eval_every=args.eval_every,
        exploration=args.exploration,
        seed=args.seed,
        device=args.device,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )
