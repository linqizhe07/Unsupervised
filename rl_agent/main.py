import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C, DQN, PPO
import os
import argparse
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from gymnasium.envs.registration import register
import glob
import numpy as np
import torch
from pathlib import Path
import json
from rl_agent.HumanoidEnv import HumanoidEnv  # Import your custom environment class


class RewardLoggerCallback(BaseCallback):
    def __init__(
        self, log_dir="reward_logs", log_file_name="reward_log.json", verbose=0
    ):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.log_file_path = os.path.join(log_dir, log_file_name)
        self.all_episode_logs = []  # This will store all episodes' data

        os.makedirs(log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_episode_end(self) -> None:
        # Get the info dictionary from the environment
        info = self.locals.get("infos", [])[0]
        episode_info = info.get("episode", {})

        if episode_info or 1 > 0:  # Always true, to avoid conditional logging issues
            # Extract the reward components
            reward_components = {
                key: episode_info[key]
                for key in episode_info.keys()
                if key not in ["r", "l", "t"]
            }

            # Prepare data to save
            log_data = {
                "total_reward": episode_info.get("r", None),
                "reward_components": reward_components,
                "episode_length": episode_info.get("l", None),
                "episode_time": episode_info.get("t", None),
                "full_info": info,  # Save the entire info dictionary for debugging purposes
            }

            # Append the log data to the list
            self.all_episode_logs.append(log_data)

            # Save to a single JSON file
            with open(self.log_file_path, "w") as f:
                json.dump(self.all_episode_logs, f, indent=4)


class VelocityLoggerCallback(BaseCallback):
    def __init__(self, velocity_filepath, wandb_cfg=None, env_name="HumanoidEnv", verbose=0):
        super(VelocityLoggerCallback, self).__init__(verbose)
        self.velocity_filepath = velocity_filepath
        self.wandb_cfg = wandb_cfg
        self.env_name = env_name
        self._wandb_run = None
        self._step_offset = None  # offset for wandb step (handles inherited checkpoints)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.velocity_filepath), exist_ok=True)

    def _init_wandb(self):
        if self.wandb_cfg is None:
            return
        try:
            import wandb
            self._wandb_run = wandb.init(
                project=self.wandb_cfg["project"],
                entity=self.wandb_cfg.get("entity"),
                group=self.wandb_cfg.get("group"),
                name=self.wandb_cfg.get("name"),
                job_type="train",
                reinit="finish_previous",
            )
        except Exception as e:
            print(f"[wandb] Failed to init in subprocess: {e}")
            self._wandb_run = None

    def _on_step(self) -> bool:
        if self._wandb_run is None and self.wandb_cfg is not None:
            self._init_wandb()
        # Normalize wandb step to start from 0 (handles inherited checkpoints
        # where model.num_timesteps carries over from the parent).
        if self._step_offset is None:
            self._step_offset = self.num_timesteps - 1
        wb_step = self.num_timesteps - self._step_offset
        info = self.locals.get("infos", [])
        if len(info) > 0:
            info0 = info[0]
            if self.env_name == "AdroitHandDoorEnv":
                signal = info0.get("fitness_signal")
                if signal is not None:
                    with open(self.velocity_filepath, "a") as f:
                        f.write(f"{signal}\n")
                    if self._wandb_run is not None:
                        wb_dict = {"train/door_hinge": signal}
                        if "success" in info0:
                            wb_dict["train/success"] = float(info0["success"])
                        self._wandb_run.log(wb_dict, step=wb_step)
            else:
                signal = info0.get("x_velocity")
                if signal is not None:
                    with open(self.velocity_filepath, "a") as f:
                        f.write(f"{signal}\n")
                    if self._wandb_run is not None:
                        self._wandb_run.log(
                            {"train/x_velocity": signal},
                            step=wb_step,
                        )
        return True

    def _on_training_end(self) -> None:
        # Do NOT call finish() here: train() calls model.learn() in a while loop,
        # so _on_training_end fires after every 5k-step chunk. Finishing the run
        # here would cause the next chunk's _on_step to log to an already-finished
        # run and raise UsageError. The run is finished in train()'s finally block.
        pass


#    train(env, sb3_algo, reward_func, island_id, generation_id, counter)


def train(
    env,
    sb3_algo,
    reward_func,
    island_id,
    generation_id,
    counter,
    velocity_path,
    model_checkpoint_path,
    output_path,
    log_dir,
    wandb_cfg=None,
    env_name="HumanoidEnv",
    parent_checkpoint_path=None,
    gpu_id=0,
):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    current_timesteps = 0
    final_checkpoint_path = os.path.join(
        model_checkpoint_path,
        f"{generation_id}_{counter}.zip",
    )
    velocity_callback = VelocityLoggerCallback(
        velocity_filepath=velocity_path,
        wandb_cfg=wandb_cfg,
        env_name=env_name,
        verbose=1,
    )
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_checkpoint_path, exist_ok=True)

    reward_callback = RewardLoggerCallback(log_dir, verbose=1)

    # Load parent checkpoint for fine-tuning, or create from scratch
    if parent_checkpoint_path and os.path.exists(parent_checkpoint_path):
        print(f"[Inherit] Loading parent checkpoint: {parent_checkpoint_path}")
        algo_cls = {"SAC": SAC, "TD3": TD3, "A2C": A2C, "DQN": DQN, "PPO": PPO}
        model = algo_cls[sb3_algo].load(
            parent_checkpoint_path, env=env, device=device, tensorboard_log=log_dir,
        )
    elif sb3_algo == "SAC":
        model = SAC("MlpPolicy", env, verbose=1, device=device, tensorboard_log=log_dir)
    elif sb3_algo == "TD3":
        model = TD3("MlpPolicy", env, verbose=1, device=device, tensorboard_log=log_dir)
    elif sb3_algo == "A2C":
        model = A2C("MlpPolicy", env, verbose=1, device=device, tensorboard_log=log_dir)
    elif sb3_algo == "DQN":
        model = DQN("MlpPolicy", env, verbose=1, device=device, tensorboard_log=log_dir)
    elif sb3_algo == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, device=device, tensorboard_log=log_dir)
    elif sb3_algo == "custom_net":
        model = SAC(
            CustomSACPolicy, env, verbose=1, device=device, tensorboard_log=log_dir
        )

    TIMESTEPS = 5000
    total_timesteps = 500000

    try:
        while current_timesteps < total_timesteps:
            model.learn(
                total_timesteps=TIMESTEPS,
                reset_num_timesteps=False,
                callback=[velocity_callback, reward_callback],
            )
            current_timesteps += TIMESTEPS

            model.save(final_checkpoint_path)
    finally:
        # Explicitly finish subprocess wandb run so data is flushed to the server.
        # Cannot rely on atexit because ProcessPoolExecutor reuses worker processes.
        if velocity_callback._wandb_run is not None:
            velocity_callback._wandb_run.finish()


# env=HumanoidEnv(
#         reward_fn_path=reward_fn_path,
#         counter=counter,
#         iteration=iteration,
#         group_id=group_id,
#         llm_model=llm_model,
#         baseline=baseline,
#         render_mode=None
#     )


def run_training(
    reward_func,
    island_id,
    generation_id,
    counter,
    reward_history_file,
    model_checkpoint_file,
    fitness_file,
    velocity_file,
    output_path,
    log_dir,
    env_name="HumanoidEnv",
    wandb_cfg=None,
    parent_checkpoint_path=None,
    gpu_id=0,
):
    if env_name == "AdroitHandDoorEnv":
        from rl_agent.AdroitEnv import AdroitHandDoorEnv
        gymenv = AdroitHandDoorEnv(
            reward_func_str=reward_func,
            counter=counter,
            iteration=generation_id,
            group_id=str(island_id),
            reward_history_file=reward_history_file,
            mode="train",
        )
    else:
        gymenv = HumanoidEnv(
            reward_func_str=reward_func,
            counter=counter,
            generation_id=generation_id,
            island_id=island_id,
            reward_history_file=reward_history_file,
            model_checkpoint_file=model_checkpoint_file,
            velocity_file=velocity_file,
        )
    sb3_algo = "SAC"

    env = Monitor(gymenv)  # Ensure monitoring
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(
        env, norm_obs=False, norm_reward=False, clip_obs=100.0, clip_reward=100
    )  # if norm_obs or norm_reward is true you need to save the state of the vecnormalize when loading the model weights.
    train(
        env,
        sb3_algo,
        reward_func,
        island_id,
        generation_id,
        counter,
        velocity_file,
        model_checkpoint_file,
        output_path,
        log_dir,
        wandb_cfg=wandb_cfg,
        env_name=env_name,
        parent_checkpoint_path=parent_checkpoint_path,
        gpu_id=gpu_id,
    )

    # return velocity_filepath


def run_training_u2o(
    reward_func,
    island_id,
    generation_id,
    counter,
    reward_history_file,
    model_checkpoint_file,
    fitness_file,
    velocity_file,
    output_path,
    log_dir,
    pretrained_dir,
    u2o_cfg,
    parent_checkpoint_path=None,
    wandb_cfg=None,
    env_name="HumanoidEnv",
    gpu_id=0,
):
    """
    U2O version of run_training using SFAgent with successor features.

    1. Load pretrained (or parent) SFAgent and replay buffer
    2. Create HumanoidEnv with LLM reward
    3. Infer skill z* from task reward via least-squares on phi
    4. Fine-tune with online data collection + offline data mixing
    """
    import json
    from collections import OrderedDict
    from u2o.agent import SFAgent, SFAgentConfig
    from u2o.replay_buffer import ReplayBuffer
    from u2o.wrappers import EpisodeMonitor
    from utils import define_function_from_string
    from rl_agent.reward_utils import (
        build_env_state_from_transition,
        call_reward_func_dynamically,
    )

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_checkpoint_file, exist_ok=True)

    # Initialize wandb for this fine-tune run (subprocess)
    wb_run = None
    if wandb_cfg is not None:
        try:
            import wandb
            wb_run = wandb.init(
                project=wandb_cfg["project"],
                entity=wandb_cfg.get("entity"),
                group=wandb_cfg.get("group"),
                name=f"finetune_g{generation_id}_c{counter}_i{island_id}",
                job_type="finetune",
                reinit="finish_previous",
            )
        except Exception as e:
            print(f"[wandb] Failed to init in subprocess: {e}")
            wb_run = None

    # Load pretrained config
    config_path = os.path.join(pretrained_dir, "pretrain_config.json")
    with open(config_path, "r") as f:
        pretrain_config = json.load(f)

    # Recreate SFAgent: pretrain_config is the base, u2o_cfg overrides specific keys
    def _pc(key, default):
        """Read from u2o_cfg first (runtime override), then pretrain_config, then default."""
        return u2o_cfg.get(key, pretrain_config.get(key, default))

    cfg = SFAgentConfig(
        obs_dim=pretrain_config["obs_dim"],
        action_dim=pretrain_config["action_dim"],
        device=str(device),
        # architecture — must match pretrain exactly
        z_dim=pretrain_config["z_dim"],
        hidden_dim=pretrain_config["hidden_dim"],
        phi_hidden_dim=pretrain_config["phi_hidden_dim"],
        feature_dim=pretrain_config["feature_dim"],
        feature_learner=pretrain_config["feature_learner"],
        preprocess=pretrain_config.get("preprocess", True),
        add_trunk=pretrain_config.get("add_trunk", False),
        # HILP
        hilp_discount=pretrain_config.get("hilp_discount", 0.98),
        hilp_expectile=pretrain_config.get("hilp_expectile", 0.5),
        feature_type=pretrain_config.get("feature_type", "state"),
        # optimizer
        lr=_pc("lr", 1e-4),
        lr_coef=pretrain_config.get("lr_coef", 5.0),
        # training loop — can be overridden at finetune time
        batch_size=_pc("batch_size", 1024),
        num_sf_updates=_pc("num_sf_updates", 1),
        update_every_steps=_pc("update_every_steps", 1),
        update_z_every_step=_pc("update_z_every_step", 300),
        update_cov_every_step=_pc("update_cov_every_step", 1000),
        num_expl_steps=_pc("num_expl_steps", 2000),
        # SF-specific
        sf_target_tau=_pc("sf_target_tau", 0.01),
        mix_ratio=_pc("mix_ratio", 0.5),
        q_loss=_pc("q_loss", True),
        use_rew_norm=_pc("use_rew_norm", True),
        # actor
        stddev_schedule=_pc("stddev_schedule", "0.2"),
        stddev_clip=_pc("stddev_clip", 0.3),
        boltzmann=_pc("boltzmann", False),
        temp=_pc("temp", 1.0),
    )
    agent = SFAgent(cfg)

    # Load weights: parent checkpoint if available, otherwise pretrained
    if parent_checkpoint_path and os.path.exists(parent_checkpoint_path):
        agent.load(parent_checkpoint_path)
        agent.cfg.num_expl_steps = 0
        print(f"[U2O] Loaded PARENT checkpoint from {parent_checkpoint_path}")
    else:
        agent_path = os.path.join(pretrained_dir, "agent_checkpoint.pt")
        agent.load(agent_path)
        agent.cfg.num_expl_steps = 0
        print(f"[U2O] Loaded pretrained agent from {agent_path}")

    # Load offline replay buffer
    offline_buffer = ReplayBuffer(
        max_episodes=pretrain_config.get(
            "max_buffer_episodes",
            pretrain_config.get("collection_episodes", 5000),
        ),
        discount=pretrain_config.get("discount", 0.98),
        future=pretrain_config.get("future", 0.99),
        p_randomgoal=pretrain_config.get("p_randomgoal", 0.375),
    )
    buffer_path = os.path.join(pretrained_dir, "replay_buffer.npz")
    offline_buffer.load(buffer_path)
    print(f"[U2O] Loaded offline buffer ({len(offline_buffer)} episodes)")

    # Infer z* from task reward using least-squares on phi.
    # Sample directly from buffer storage so we can access per-transition
    # extras (joint_velocities, joint_forces) for accurate reward evaluation.
    reward_func_obj, _ = define_function_from_string(reward_func)
    sample_size = min(cfg.batch_size * 4, offline_buffer.num_transitions)

    n_eps = len(offline_buffer)
    ep_idx = np.random.randint(0, n_eps, size=sample_size)
    eps_lengths = offline_buffer._episodes_length[ep_idx]
    step_idx = (np.random.rand(sample_size) * eps_lengths).astype(np.int32)

    obs_np = offline_buffer._storage["observation"][ep_idx, step_idx]
    action_np = offline_buffer._storage["action"][ep_idx, step_idx]
    next_obs_np = offline_buffer._storage["next_observation"][ep_idx, step_idx]

    # Retrieve stored extras (e.g. joint_velocities, joint_forces for Adroit)
    extra_names = offline_buffer.extra_fields
    extras_np = {
        name: offline_buffer._storage[name][ep_idx, step_idx]
        for name in extra_names
    }

    # Compute task rewards on sampled transitions using next_obs timing
    # to match env.step reward semantics and original U2O behavior.
    task_rewards = []
    reward_failures = 0
    for i in range(sample_size):
        try:
            per_transition_extras = {name: extras_np[name][i] for name in extra_names}
            env_state = build_env_state_from_transition(
                obs=obs_np[i],
                action=action_np[i],
                next_obs=next_obs_np[i],
                reward_on="next",
                **per_transition_extras,
            )
            r, _ = call_reward_func_dynamically(reward_func_obj, env_state)
            task_rewards.append(float(r))
        except Exception:
            reward_failures += 1
            task_rewards.append(0.0)
    if reward_failures > 0:
        print(
            f"[U2O] Warning: task reward eval failed on "
            f"{reward_failures}/{len(task_rewards)} sampled transitions; filled with 0."
        )
    task_reward_tensor = torch.tensor(
        task_rewards, device=device, dtype=torch.float32
    ).unsqueeze(-1)

    # Build torch tensors for phi computation
    obs_t = torch.as_tensor(obs_np, device=device, dtype=torch.float32)
    next_obs_t = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)

    # Match U2O init-meta logic for state features.
    if agent.cfg.feature_type == "state":
        meta_obs = next_obs_t
        meta_next_obs = next_obs_t
    else:
        meta_obs = obs_t
        meta_next_obs = next_obs_t
    meta = agent.infer_meta_from_obs_and_rewards(
        meta_obs, task_reward_tensor, meta_next_obs
    )
    z_star = meta["z"]
    print(f"[U2O] Inferred z* (norm={np.linalg.norm(z_star):.4f})")

    # Relabel offline buffer rewards with task reward function
    print(f"[U2O] Relabeling offline buffer ({offline_buffer.num_transitions} transitions) with task reward...")
    relabel_failures = {"count": 0}

    def _task_reward_fn(obs, action, next_obs, **extras):
        try:
            env_state = build_env_state_from_transition(
                obs=obs,
                action=action,
                next_obs=next_obs,
                reward_on="next",
                **extras,
            )
            r, _ = call_reward_func_dynamically(reward_func_obj, env_state)
            return float(r)
        except Exception:
            relabel_failures["count"] += 1
            return 0.0

    offline_buffer.relabel_rewards(_task_reward_fn)
    if relabel_failures["count"] > 0:
        print(
            f"[U2O] Warning: relabel failed on {relabel_failures['count']} "
            "transitions; corresponding rewards were set to 0."
        )
    print(f"[U2O] Offline buffer relabeled.")

    # Set solved meta for agent
    agent.solved_meta = meta

    # Create environment for online fine-tuning
    if env_name == "AdroitHandDoorEnv":
        from rl_agent.AdroitEnv import AdroitHandDoorEnv
        gymenv = AdroitHandDoorEnv(
            reward_func_str=reward_func,
            counter=counter,
            iteration=generation_id,
            group_id=str(island_id),
            reward_history_file=reward_history_file,
            mode="train",
        )
    else:
        gymenv = HumanoidEnv(
            reward_func_str=reward_func,
            counter=counter,
            generation_id=generation_id,
            island_id=island_id,
            reward_history_file=reward_history_file,
            model_checkpoint_file=model_checkpoint_file,
            velocity_file=velocity_file,
        )
    env = EpisodeMonitor(gymenv)

    # Online replay buffer for fine-tuning
    finetune_steps = u2o_cfg.get("finetune_steps", 300000)
    online_buffer = ReplayBuffer(
        max_episodes=1000,
        discount=pretrain_config.get("discount", 0.98),
        future=1.0,
        max_episode_length=getattr(gymenv, "max_episode_steps", 1000) + 1,
    )

    # Fine-tuning loop
    obs, info = env.reset()
    episode_reward = 0.0
    episode_count = 0
    episode_step = 0
    velocity_log = []
    log_every = 400

    os.makedirs(os.path.dirname(velocity_file), exist_ok=True)

    try:
        for step in range(1, finetune_steps + 1):
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
            if env_name == "AdroitHandDoorEnv":
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
                    if env_name == "AdroitHandDoorEnv":
                        wb_dict["finetune/door_hinge"] = info.get("fitness_signal", 0)
                        wb_dict["finetune/success"] = float(info.get("success", False))
                    else:
                        wb_dict["finetune/x_velocity"] = info.get("x_velocity", 0)
                    wb_run.log(wb_dict, step=step)
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

        # Save velocity log
        with open(velocity_file, "w") as f:
            for v in velocity_log:
                f.write(f"{v}\n")

        # Save final checkpoint
        final_path = os.path.join(
            model_checkpoint_file,
            f"u2o_final_{generation_id}_{counter}.pt",
        )
        agent.save(final_path)
        print(
            f"[U2O] Fine-tuning complete: {episode_count} episodes, {finetune_steps} steps"
        )
    finally:
        if wb_run is not None:
            wb_run.finish()
