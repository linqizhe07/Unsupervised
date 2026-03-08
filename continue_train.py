"""
Continue training a SAC individual from an existing checkpoint.

Usage:
    # From database paths
    python continue_train.py \
        --db_root /path/to/database/revolve_auto/139 \
        --island_id 6 --generation_id 0 --counter_id 13 \
        --total_timesteps 500000

    # Explicit paths (e.g. files moved or individual discarded)
    python continue_train.py \
        --reward_fn /path/to/reward_function.txt \
        --checkpoint /path/to/0_13.zip \
        --total_timesteps 500000

    # With wandb logging
    python continue_train.py \
        --db_root /path/to/database/revolve_auto/139 \
        --island_id 6 --generation_id 0 --counter_id 13 \
        --wandb_project humanoid-baseline-1 \
        --total_timesteps 500000

The script will:
  1. Load the reward function (.txt)
  2. Load the SAC checkpoint (.zip)
  3. Continue training with the same reward function
  4. Save the new checkpoint and evaluate fitness
"""

import argparse
import os
import sys

import torch


def main():
    parser = argparse.ArgumentParser(description="Continue SAC training for a single individual")

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
                        help="Path to SAC .zip checkpoint (overrides db_root lookup)")

    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory. Defaults to db_root/train_continued/ or ./train_output/")
    parser.add_argument("--total_timesteps", type=int, default=500000,
                        help="Total NEW timesteps to train (default: 500000)")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--env_name", type=str, default="HumanoidEnv",
                        choices=["HumanoidEnv", "AdroitHandDoorEnv"])
    parser.add_argument("--algo", type=str, default="SAC",
                        choices=["SAC", "TD3", "A2C", "PPO", "DQN"])

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
    print(f"[Train] Reward function: {reward_fn_path}")

    # ---- Resolve checkpoint path ----
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        if args.db_root is None or args.island_id is None:
            parser.error("Must provide --checkpoint or --db_root with --island_id/--generation_id/--counter_id")
        checkpoint_path = os.path.join(
            args.db_root, f"island_{args.island_id}", "model_checkpoints",
            f"{args.generation_id}_{args.counter_id}.zip",
        )

    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    print(f"[Train] Checkpoint: {checkpoint_path}")

    # ---- Identifiers for output file naming ----
    gen = args.generation_id if args.generation_id is not None else 0
    cnt = args.counter_id if args.counter_id is not None else 0
    iid = args.island_id if args.island_id is not None else 0
    tag = f"{gen}_{cnt}"

    # ---- Output paths ----
    if args.output_dir:
        output_dir = args.output_dir
    elif args.db_root:
        output_dir = os.path.join(args.db_root, "train_continued")
    else:
        output_dir = "./train_output"
    out_island = os.path.join(output_dir, f"island_{iid}")

    reward_history_file = os.path.join(out_island, "reward_history", f"{tag}.json")
    model_checkpoint_dir = os.path.join(out_island, "model_checkpoints")
    fitness_file = os.path.join(out_island, "fitness_scores", f"{tag}.txt")
    velocity_file = os.path.join(out_island, "velocity_logs", f"velocity_{tag}.txt")
    log_dir = os.path.join(out_island, "log_dir", tag)

    for d in [os.path.dirname(reward_history_file), model_checkpoint_dir,
              os.path.dirname(fitness_file), os.path.dirname(velocity_file), log_dir]:
        os.makedirs(d, exist_ok=True)

    # ---- Create environment ----
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3 import SAC, TD3, A2C, PPO, DQN

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

    monitored_env = Monitor(gymenv)
    vec_env = DummyVecEnv([lambda _env=monitored_env: _env])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=False, clip_obs=100.0, clip_reward=100)

    # ---- Load model ----
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    algo_cls = {"SAC": SAC, "TD3": TD3, "A2C": A2C, "PPO": PPO, "DQN": DQN}
    model = algo_cls[args.algo].load(
        checkpoint_path, env=vec_env, device=device, tensorboard_log=log_dir,
    )
    print(f"[Train] Loaded {args.algo} checkpoint, continuing for {args.total_timesteps} timesteps...")

    # ---- wandb config ----
    wandb_cfg = None
    if args.wandb_project:
        wandb_cfg = {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "group": args.wandb_group,
            "name": args.wandb_name or f"train_g{gen}_c{cnt}_i{iid}",
        }

    # ---- Callbacks ----
    from rl_agent.main import VelocityLoggerCallback, RewardLoggerCallback

    velocity_callback = VelocityLoggerCallback(
        velocity_filepath=velocity_file,
        wandb_cfg=wandb_cfg,
        env_name=args.env_name,
        verbose=1,
    )
    reward_callback = RewardLoggerCallback(log_dir, verbose=1)

    # ---- Training loop ----
    TIMESTEPS = 5000
    current_timesteps = 0
    final_checkpoint_path = os.path.join(model_checkpoint_dir, f"{tag}.zip")

    try:
        while current_timesteps < args.total_timesteps:
            model.learn(
                total_timesteps=TIMESTEPS,
                reset_num_timesteps=False,
                callback=[velocity_callback, reward_callback],
            )
            current_timesteps += TIMESTEPS
            model.save(final_checkpoint_path)

            if current_timesteps % 50000 == 0:
                print(f"  Progress: {current_timesteps}/{args.total_timesteps}")
    finally:
        if velocity_callback._wandb_run is not None:
            velocity_callback._wandb_run.finish()

    # ---- Evaluate fitness ----
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

    print(f"\n[Train] Training complete!")
    print(f"  Timesteps: {args.total_timesteps}")
    print(f"  Fitness: {fitness:.4f}")
    print(f"  Checkpoint: {final_checkpoint_path}")
    print(f"  Velocity log: {velocity_file}")


if __name__ == "__main__":
    main()
