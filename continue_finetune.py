"""
U2O fine-tune a single individual (SFAgent with z*).

Supports two scenarios:
  A) Continue a previous U2O fine-tune: loads u2o_final_{gen}_{counter}.pt
  B) Start fresh U2O fine-tune for a SAC individual: falls back to
     pretrained_dir/agent_checkpoint.pt (only needs the reward function)

Usage:
    # Scenario A: continue from existing U2O checkpoint
    python continue_finetune.py \
        --db_root /path/to/database/revolve_auto/139 \
        --island_id 6 --generation_id 0 --counter_id 13 \
        --pretrained_dir /path/to/u2o_pretrained

    # Scenario B: explicit paths (e.g. files moved or individual discarded)
    python continue_finetune.py \
        --reward_fn /path/to/reward_function.txt \
        --pretrained_dir /path/to/u2o_pretrained

    # Scenario C: continue from a specific U2O checkpoint
    python continue_finetune.py \
        --reward_fn /path/to/reward_function.txt \
        --checkpoint /path/to/u2o_final_0_13.pt \
        --pretrained_dir /path/to/u2o_pretrained
"""

import argparse
import json
import os
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="U2O fine-tune a single individual (SFAgent with z*)")

    # --- Source: either db_root + ids, or explicit paths ---
    parser.add_argument("--db_root", type=str, default=None,
                        help="Database root, e.g. database/revolve_auto/139")
    parser.add_argument("--island_id", type=int, default=None)
    parser.add_argument("--generation_id", type=int, default=None)
    parser.add_argument("--counter_id", type=int, default=None)

    # --- Explicit path overrides (take priority over db_root) ---
    parser.add_argument("--reward_fn", type=str, default=None,
                        help="Path to reward function .txt file (overrides db_root lookup)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to U2O .pt checkpoint (overrides db_root lookup). "
                             "If omitted, falls back to pretrained_dir/agent_checkpoint.pt")

    parser.add_argument("--pretrained_dir", type=str, required=True,
                        help="U2O pretrained directory (pretrain_config.json, agent_checkpoint.pt, replay_buffer.npz)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. Defaults to db_root/finetune_continued/ or ./finetune_output/")
    parser.add_argument("--finetune_steps", type=int, default=300000)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--env_name", type=str, default="HumanoidEnv",
                        choices=["HumanoidEnv", "AdroitHandDoorEnv"])
    # U2O override options
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--mix_ratio", type=float, default=None)
    parser.add_argument("--stddev_schedule", type=str, default=None)

    # --- wandb ---
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="wandb project name (enables wandb if set)")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="wandb run name (auto-generated if omitted)")
    args = parser.parse_args()

    # ---- Resolve reward function path ----
    reward_fn_path = args.reward_fn
    if reward_fn_path is None:
        if args.db_root is None or args.island_id is None or args.generation_id is None or args.counter_id is None:
            parser.error("Must provide either --reward_fn or all of --db_root/--island_id/--generation_id/--counter_id")
        island_dir = os.path.join(args.db_root, f"island_{args.island_id}")
        reward_fn_path = os.path.join(island_dir, "generated_fns",
                                      f"{args.generation_id}_{args.counter_id}.txt")

    if not os.path.exists(reward_fn_path):
        print(f"[Error] Reward function not found: {reward_fn_path}")
        sys.exit(1)

    with open(reward_fn_path) as f:
        reward_func_str = f.read()
    print(f"[U2O] Reward function: {reward_fn_path}")

    # ---- Resolve checkpoint path ----
    # Priority: --checkpoint > db_root u2o_final_*.pt > pretrained agent
    checkpoint_path = args.checkpoint
    if checkpoint_path is None and args.db_root is not None and args.island_id is not None:
        candidate = os.path.join(
            args.db_root, f"island_{args.island_id}", "model_checkpoints",
            f"u2o_final_{args.generation_id}_{args.counter_id}.pt",
        )
        if os.path.exists(candidate):
            checkpoint_path = candidate

    if checkpoint_path is None:
        # Fallback to pretrained agent
        checkpoint_path = os.path.join(args.pretrained_dir, "agent_checkpoint.pt")

    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    print(f"[U2O] Checkpoint: {checkpoint_path}")

    # ---- Identifiers for output file naming ----
    gen = args.generation_id if args.generation_id is not None else 0
    cnt = args.counter_id if args.counter_id is not None else 0
    iid = args.island_id if args.island_id is not None else 0
    tag = f"{gen}_{cnt}"

    # ---- Output paths ----
    if args.output_dir:
        output_dir = args.output_dir
    elif args.db_root:
        output_dir = os.path.join(args.db_root, "finetune_continued")
    else:
        output_dir = "./finetune_output"
    out_island = os.path.join(output_dir, f"island_{iid}")

    reward_history_file = os.path.join(out_island, "reward_history", f"{tag}.json")
    model_checkpoint_dir = os.path.join(out_island, "model_checkpoints")
    fitness_file = os.path.join(out_island, "fitness_scores", f"{tag}.txt")
    velocity_file = os.path.join(out_island, "velocity_logs", f"velocity_{tag}.txt")
    log_dir = os.path.join(out_island, "log_dir", tag)

    for d in [os.path.dirname(reward_history_file), model_checkpoint_dir,
              os.path.dirname(fitness_file), os.path.dirname(velocity_file), log_dir]:
        os.makedirs(d, exist_ok=True)

    # ---- Load pretrain config ----
    config_path = os.path.join(args.pretrained_dir, "pretrain_config.json")
    with open(config_path) as f:
        pretrain_config = json.load(f)

    # ---- Build SFAgent ----
    from u2o.agent import SFAgent, SFAgentConfig
    from u2o.replay_buffer import ReplayBuffer
    from u2o.wrappers import EpisodeMonitor
    from utils import define_function_from_string
    from rl_agent.reward_utils import build_env_state_from_transition, call_reward_func_dynamically

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    def _pc(key, default):
        return pretrain_config.get(key, default)

    cfg = SFAgentConfig(
        obs_dim=pretrain_config["obs_dim"],
        action_dim=pretrain_config["action_dim"],
        device=str(device),
        z_dim=pretrain_config["z_dim"],
        hidden_dim=pretrain_config["hidden_dim"],
        phi_hidden_dim=pretrain_config["phi_hidden_dim"],
        feature_dim=pretrain_config["feature_dim"],
        feature_learner=pretrain_config["feature_learner"],
        preprocess=pretrain_config.get("preprocess", True),
        add_trunk=pretrain_config.get("add_trunk", False),
        hilp_discount=pretrain_config.get("hilp_discount", 0.98),
        hilp_expectile=pretrain_config.get("hilp_expectile", 0.5),
        feature_type=pretrain_config.get("feature_type", "state"),
        lr=args.lr if args.lr is not None else _pc("lr", 1e-4),
        lr_coef=pretrain_config.get("lr_coef", 5.0),
        batch_size=args.batch_size if args.batch_size is not None else _pc("batch_size", 1024),
        num_sf_updates=_pc("num_sf_updates", 1),
        update_every_steps=_pc("update_every_steps", 1),
        update_z_every_step=_pc("update_z_every_step", 300),
        update_cov_every_step=_pc("update_cov_every_step", 1000),
        num_expl_steps=0,
        sf_target_tau=_pc("sf_target_tau", 0.01),
        mix_ratio=args.mix_ratio if args.mix_ratio is not None else _pc("mix_ratio", 0.5),
        q_loss=_pc("q_loss", True),
        use_rew_norm=_pc("use_rew_norm", True),
        stddev_schedule=args.stddev_schedule if args.stddev_schedule is not None else _pc("stddev_schedule", "0.2"),
        stddev_clip=_pc("stddev_clip", 0.3),
        boltzmann=_pc("boltzmann", False),
        temp=_pc("temp", 1.0),
    )
    agent = SFAgent(cfg)
    agent.load(checkpoint_path)
    agent.cfg.num_expl_steps = 0
    print(f"[U2O] Agent loaded from {checkpoint_path}")

    # ---- Load offline replay buffer ----
    offline_buffer = ReplayBuffer(
        max_episodes=pretrain_config.get(
            "max_buffer_episodes",
            pretrain_config.get("collection_episodes", 5000),
        ),
        discount=pretrain_config.get("discount", 0.98),
        future=pretrain_config.get("future", 0.99),
        p_randomgoal=pretrain_config.get("p_randomgoal", 0.375),
    )
    buffer_path = os.path.join(args.pretrained_dir, "replay_buffer.npz")
    offline_buffer.load(buffer_path)
    print(f"[U2O] Loaded offline buffer ({len(offline_buffer)} episodes)")

    # ---- Infer z* from reward function ----
    reward_func_obj, _ = define_function_from_string(reward_func_str)
    sample_size = min(cfg.batch_size * 4, offline_buffer.num_transitions)

    n_eps = len(offline_buffer)
    ep_idx = np.random.randint(0, n_eps, size=sample_size)
    eps_lengths = offline_buffer._episodes_length[ep_idx]
    step_idx = (np.random.rand(sample_size) * eps_lengths).astype(np.int32)

    obs_np = offline_buffer._storage["observation"][ep_idx, step_idx]
    action_np = offline_buffer._storage["action"][ep_idx, step_idx]
    next_obs_np = offline_buffer._storage["next_observation"][ep_idx, step_idx]

    extra_names = offline_buffer.extra_fields
    extras_np = {
        name: offline_buffer._storage[name][ep_idx, step_idx]
        for name in extra_names
    }

    task_rewards = []
    reward_failures = 0
    for i in range(sample_size):
        try:
            per_transition_extras = {name: extras_np[name][i] for name in extra_names}
            env_state = build_env_state_from_transition(
                obs=obs_np[i], action=action_np[i], next_obs=next_obs_np[i],
                reward_on="next", **per_transition_extras,
            )
            r, _ = call_reward_func_dynamically(reward_func_obj, env_state)
            task_rewards.append(float(r))
        except Exception:
            reward_failures += 1
            task_rewards.append(0.0)
    if reward_failures > 0:
        print(f"[U2O] Warning: task reward eval failed on {reward_failures}/{sample_size} transitions")

    task_reward_tensor = torch.tensor(task_rewards, device=device, dtype=torch.float32).unsqueeze(-1)
    obs_t = torch.as_tensor(obs_np, device=device, dtype=torch.float32)
    next_obs_t = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)

    if agent.cfg.feature_type == "state":
        meta_obs = next_obs_t
        meta_next_obs = next_obs_t
    else:
        meta_obs = obs_t
        meta_next_obs = next_obs_t

    meta = agent.infer_meta_from_obs_and_rewards(meta_obs, task_reward_tensor, meta_next_obs)
    z_star = meta["z"]
    print(f"[U2O] Inferred z* (norm={np.linalg.norm(z_star):.4f})")

    # ---- Relabel offline buffer with task reward ----
    print(f"[U2O] Relabeling offline buffer ({offline_buffer.num_transitions} transitions)...")
    relabel_failures = {"count": 0}

    def _task_reward_fn(obs, action, next_obs, **extras):
        try:
            env_state = build_env_state_from_transition(
                obs=obs, action=action, next_obs=next_obs,
                reward_on="next", **extras,
            )
            r, _ = call_reward_func_dynamically(reward_func_obj, env_state)
            return float(r)
        except Exception:
            relabel_failures["count"] += 1
            return 0.0

    offline_buffer.relabel_rewards(_task_reward_fn)
    if relabel_failures["count"] > 0:
        print(f"[U2O] Warning: relabel failed on {relabel_failures['count']} transitions")
    print("[U2O] Offline buffer relabeled.")

    agent.solved_meta = meta

    # ---- Create environment ----
    if args.env_name == "AdroitHandDoorEnv":
        from rl_agent.AdroitEnv import AdroitHandDoorEnv
        gymenv = AdroitHandDoorEnv(
            reward_func_str=reward_func_str,
            counter=cnt, iteration=gen, group_id=str(iid),
            reward_history_file=reward_history_file, mode="train",
        )
    else:
        from rl_agent.HumanoidEnv import HumanoidEnv
        gymenv = HumanoidEnv(
            reward_func_str=reward_func_str,
            counter=cnt, generation_id=gen, island_id=iid,
            reward_history_file=reward_history_file,
            model_checkpoint_file=model_checkpoint_dir,
            velocity_file=velocity_file,
        )
    env = EpisodeMonitor(gymenv)

    # ---- Online replay buffer ----
    online_buffer = ReplayBuffer(
        max_episodes=1000,
        discount=pretrain_config.get("discount", 0.98),
        future=1.0,
        max_episode_length=getattr(gymenv, "max_episode_steps", 1000) + 1,
    )

    # ---- wandb init ----
    wb_run = None
    if args.wandb_project:
        try:
            import wandb
            wb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                name=args.wandb_name or f"finetune_g{gen}_c{cnt}_i{iid}",
                job_type="finetune",
                reinit="finish_previous",
            )
        except Exception as e:
            print(f"[wandb] Failed to init: {e}")
            wb_run = None

    # ---- Fine-tuning loop ----
    obs, info = env.reset()
    episode_reward = 0.0
    episode_count = 0
    episode_step = 0
    velocity_log = []
    log_every = 400

    print(f"[U2O] Starting fine-tuning for {args.finetune_steps} steps...")

    try:
        for step in range(1, args.finetune_steps + 1):
            action = agent.act(obs, meta, step, eval_mode=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            online_buffer.add_transition(
                obs=obs, action=action, reward=reward,
                next_obs=next_obs, done=done,
                discount=0.0 if terminated else 1.0,
            )

            episode_reward += reward
            episode_step += 1
            if args.env_name == "AdroitHandDoorEnv":
                signal = info.get("fitness_signal")
            else:
                signal = info.get("x_velocity")
            if signal is not None:
                velocity_log.append(signal)

            if done:
                episode_count += 1
                if wb_run is not None:
                    wb_dict = {
                        "finetune/episode_reward": episode_reward,
                        "finetune/episode_length": episode_step,
                        "finetune/episode_count": episode_count,
                    }
                    if args.env_name == "AdroitHandDoorEnv":
                        wb_dict["finetune/door_hinge"] = info.get("fitness_signal", 0)
                        wb_dict["finetune/success"] = float(info.get("success", False))
                    else:
                        wb_dict["finetune/x_velocity"] = info.get("x_velocity", 0)
                    wb_run.log(wb_dict, step=step)
                if episode_count % 10 == 0:
                    print(f"  Episode {episode_count} | step {step}/{args.finetune_steps} | reward={episode_reward:.2f}")
                obs, info = env.reset()
                episode_reward = 0.0
                episode_step = 0
            else:
                obs = next_obs

            # Update agent with mixed online + offline data
            if len(online_buffer) >= 2 and step % cfg.update_every_steps == 0:
                metrics = agent.update_with_offline_data(
                    replay_loader=online_buffer,
                    step=step,
                    with_reward=True,
                    meta=meta,
                    replay_loader_offline=offline_buffer,
                )
                if wb_run is not None and step % log_every == 0 and metrics:
                    wb_run.log(
                        {f"finetune/{k}": v for k, v in metrics.items()},
                        step=step,
                    )

        # ---- Save results ----
        with open(velocity_file, "w") as f:
            for v in velocity_log:
                f.write(f"{v}\n")

        final_path = os.path.join(model_checkpoint_dir, f"u2o_final_{tag}.pt")
        agent.save(final_path)
    finally:
        if wb_run is not None:
            wb_run.finish()

    # Evaluate fitness
    from rl_agent.evaluate import return_score
    from rl_agent.fitness_score import calculate_fitness_score
    if args.env_name == "AdroitHandDoorEnv":
        fitness = calculate_fitness_score(
            os.path.join(out_island, "reward_history", f"{tag}.txt")
        )
    else:
        fitness = return_score(velocity_file)

    with open(fitness_file, "w") as f:
        f.write(f"{fitness}\n")

    print(f"\n[U2O] Fine-tuning complete!")
    print(f"  Episodes: {episode_count}")
    print(f"  Steps: {args.finetune_steps}")
    print(f"  Fitness: {fitness:.4f}")
    print(f"  Checkpoint: {final_path}")
    print(f"  Velocity log: {velocity_file}")


if __name__ == "__main__":
    main()
