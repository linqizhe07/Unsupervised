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
    def __init__(self, velocity_filepath, verbose=0):
        super(VelocityLoggerCallback, self).__init__(verbose)
        self.velocity_filepath = velocity_filepath
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.velocity_filepath), exist_ok=True)

    def _on_step(self) -> bool:
        info = self.locals.get("infos", [])
        if len(info) > 0 and "x_velocity" in info[0]:
            x_velocity = info[0]["x_velocity"]
            # Save the velocity to a file with the specified name in the directory
            with open(self.velocity_filepath, "a") as f:
                f.write(f"{x_velocity}\n")
        return True


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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_timesteps = 0
    velocity_callback = VelocityLoggerCallback(
        velocity_filepath=velocity_path,  # Directly use the full file path
        verbose=1,
    )
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_checkpoint_path, exist_ok=True)

    reward_callback = RewardLoggerCallback(log_dir, verbose=1)

    if sb3_algo == "SAC":
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

    TIMESTEPS = 500
    # total_timesteps = 3000000
    total_timesteps = 1000

    while current_timesteps < total_timesteps:
        model.learn(
            total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            callback=[velocity_callback, reward_callback],
        )
        current_timesteps += TIMESTEPS
        model_save_path = os.path.join(
            model_checkpoint_path, f"{sb3_algo}_{current_timesteps}.zip"
        )
        model_save_path = os.path.join(
            model_checkpoint_path,
            f"{sb3_algo}_{generation_id}_{counter}_{current_timesteps}.zip",
        )
        model.save(model_save_path)
        env.render()


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
):
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
):
    """
    U2O version of run_training using SFAgent with successor features.

    1. Load pretrained SFAgent and replay buffer
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_checkpoint_file, exist_ok=True)

    # Load pretrained config
    config_path = os.path.join(pretrained_dir, "pretrain_config.json")
    with open(config_path, "r") as f:
        pretrain_config = json.load(f)

    # Recreate SFAgent with same architecture
    cfg = SFAgentConfig(
        obs_dim=pretrain_config["obs_dim"],
        action_dim=pretrain_config["action_dim"],
        device=str(device),
        lr=u2o_cfg.get("lr", pretrain_config.get("lr", 1e-4)),
        hidden_dim=pretrain_config["hidden_dim"],
        phi_hidden_dim=pretrain_config["phi_hidden_dim"],
        feature_dim=pretrain_config["feature_dim"],
        z_dim=pretrain_config["z_dim"],
        batch_size=u2o_cfg.get("batch_size", pretrain_config.get("batch_size", 1024)),
        feature_learner=pretrain_config["feature_learner"],
        hilp_discount=pretrain_config.get("hilp_discount", 0.98),
        hilp_expectile=pretrain_config.get("hilp_expectile", 0.5),
    )
    agent = SFAgent(cfg)

    # Load pretrained weights
    agent_path = os.path.join(pretrained_dir, "agent_checkpoint.pt")
    agent.load(agent_path)
    agent.cfg.num_expl_steps = 0  # no random exploration during finetuning
    print(f"[U2O] Loaded pretrained agent from {agent_path}")

    # Load offline replay buffer
    offline_buffer = ReplayBuffer(
        max_episodes=pretrain_config.get("collection_episodes", 5000),
        discount=pretrain_config.get("discount", 0.98),
        future=pretrain_config.get("future", 0.99),
        p_randomgoal=pretrain_config.get("p_randomgoal", 0.375),
    )
    buffer_path = os.path.join(pretrained_dir, "replay_buffer.npz")
    offline_buffer.load(buffer_path)
    print(f"[U2O] Loaded offline buffer ({len(offline_buffer)} episodes)")

    # Infer z* from task reward using least-squares on phi
    reward_func_obj, _ = define_function_from_string(reward_func)
    sample_size = min(cfg.batch_size * 4, offline_buffer.num_transitions)
    batch = offline_buffer.sample(sample_size)
    batch = batch.to(str(device))

    # Compute task rewards on buffer transitions
    from rl_agent.HumanoidEnv import call_reward_func_dynamically, build_env_state_from_obs
    task_rewards = []
    obs_np = batch.obs.cpu().numpy()
    for i in range(obs_np.shape[0]):
        try:
            env_state = build_env_state_from_obs(obs_np[i])
            r, _ = call_reward_func_dynamically(reward_func_obj, env_state)
            task_rewards.append(float(r))
        except Exception:
            task_rewards.append(0.0)
    task_reward_tensor = torch.tensor(
        task_rewards, device=device, dtype=torch.float32
    ).unsqueeze(-1)

    meta = agent.infer_meta_from_obs_and_rewards(
        batch.obs, task_reward_tensor, batch.next_obs
    )
    z_star = meta["z"]
    print(f"[U2O] Inferred z* (norm={np.linalg.norm(z_star):.4f})")

    # Set solved meta for agent
    agent.solved_meta = meta

    # Create environment for online fine-tuning
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
    finetune_steps = u2o_cfg.get("finetune_steps", 1000)
    online_buffer = ReplayBuffer(
        max_episodes=1000,
        discount=pretrain_config.get("discount", 0.98),
        future=1.0,
    )

    # Fine-tuning loop
    obs, info = env.reset()
    episode_reward = 0.0
    episode_count = 0
    velocity_log = []

    os.makedirs(os.path.dirname(velocity_file), exist_ok=True)

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
        if "x_velocity" in info:
            velocity_log.append(info["x_velocity"])

        if done:
            episode_count += 1
            obs, info = env.reset()
            episode_reward = 0.0
        else:
            obs = next_obs

        # Update agent with mixed online + offline data
        if len(online_buffer) >= 2 and step % cfg.update_every_steps == 0:
            agent.update_with_offline_data(
                replay_loader=online_buffer,
                step=step,
                with_reward=True,
                meta=meta,
                replay_loader_offline=offline_buffer,
            )

        # Periodic checkpoint
        if step % 500 == 0:
            ckpt_path = os.path.join(
                model_checkpoint_file,
                f"u2o_{generation_id}_{counter}_{step}.pt",
            )
            agent.save(ckpt_path)

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
