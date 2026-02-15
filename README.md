# REvolve + U2O: Reward Evolution with Unsupervised-to-Online RL
******************************************************
**Based on the official ICLR 2025 REvolve paper, extended with U2O (Unsupervised-to-Online RL) pretraining.**

<p align="center">
    <a href="https://rishihazra.github.io/REvolve/" target="_blank">
        <img alt="Documentation" src="https://img.shields.io/website/https/rishihazra.github.io/EgoTV?down_color=red&down_message=offline&up_message=link">
    </a>
    <a href="https://arxiv.org/abs/2406.01309" target="_blank">
        <img src="https://img.shields.io/badge/arXiv-2406.01309-red">
    </a>
    <a href="https://arxiv.org/pdf/2406.01309">
        <img src="https://img.shields.io/badge/Downloads-PDF-blue">
    </a>
</p>

<p align="center">
  <img src="revolve.gif" alt="egoTV">
</p>

## Overview

REvolve uses LLM-guided evolutionary algorithms to automatically design reward functions for RL. This fork integrates **U2O (Unsupervised-to-Online RL)** based on the [u2o_zsrl](https://arxiv.org/abs/2408.14785) framework to address the "blind search" problem: instead of training each candidate reward function from scratch, we pretrain a skill-conditioned policy using **Successor Features (SF)** with **HILP** feature learning, then fine-tune it per candidate reward via skill inference, yielding faster evaluation and more stable fitness signals.

```
[Pretrain (once)]                        [Evolution Loop (per candidate)]

HumanoidEnv + random exploration         LLM generates reward function
        |                                        |
HILP feature learner φ(s)               Skill inference: z* = lstsq(φ, r)
        |                                        |
Successor features F(s,z,a)             Fine-tune with online + offline data
        |                                        |
Skill-conditioned actor π(a|s,z)         Evaluate fitness (velocity)
        |
agent_checkpoint.pt + replay_buffer.npz
```

## Setup
```shell
git clone https://github.com/RishiHazra/Revolve.git
cd Revolve
conda create -n "revolve" python=3.10
conda activate revolve
pip install -e .
```

## Run (Original REvolve)

```shell
export ROOT_PATH='Revolve'
export OPENAI_API_KEY='<your openai key>'
python main.py \
        evolution.num_generations=5 \
        evolution.individuals_per_generation=15 \
        database.num_islands=5 \
        database.max_island_size=8 \
        data_paths.run=10 \
        environment.name="HumanoidEnv"
```

## Run (REvolve + U2O)

### Step 1: Pretrain (one-time)

Pretrain an SFAgent with HILP feature learning and successor features. The agent collects random exploration data, then learns temporal-distance features φ(s) and skill-conditioned successor features F(s,z,a) entirely offline.

```shell
export ROOT_PATH='Revolve'
python -m u2o.pretrain \
        --output_dir ./u2o_pretrained \
        --z_dim 100 \
        --hidden_dim 2048 \
        --phi_hidden_dim 1024 \
        --feature_dim 1024 \
        --feature_learner hilp \
        --collection_episodes 10000 \
        --pretrain_steps 5000000 \
        --batch_size 2048 \
        --wandb_project revolve-u2o
```

| Output File | Description |
|-------------|-------------|
| `agent_checkpoint.pt` | Pretrained SFAgent (actor + successor features + HILP feature learner) |
| `replay_buffer.npz` | Collected exploration transitions for offline data mixing |
| `pretrain_config.json` | Full config for reproducibility (obs_dim, action_dim, hyperparams) |

### Step 2: Run Evolution with U2O

```shell
export ROOT_PATH='Revolve'
export OPENAI_API_KEY='<your openai key>'
python main.py \
        u2o.enabled=true \
        u2o.pretrained_dir=./u2o_pretrained \
        evolution.num_generations=5 \
        evolution.individuals_per_generation=15 \
        database.num_islands=5 \
        database.max_island_size=8 \
        data_paths.run=10 \
        environment.name="HumanoidEnv"
```

When `u2o.enabled=true`, each candidate reward function goes through:
1. **Skill inference**: Evaluate candidate reward on replay buffer, solve z* = lstsq(φ, r) using successor features
2. **Fine-tune**: Online data collection with inferred z*, mixed with offline replay buffer data, updating SF + actor with task rewards

When `u2o.enabled=false` (default), the original REvolve pipeline runs unchanged.

## Project Structure

```
Revolve/
├── main.py                     # Main evolutionary loop
├── modules.py                  # Reward generation, policy training, evaluation
├── rewards_database.py         # Island-based population management
├── utils.py                    # Utilities
├── prompts/                    # LLM prompts (mutation, crossover)
├── evolutionary_utils/         # Island/Individual entities
├── rl_agent/                   # RL training and evaluation
│   ├── main.py                 # SAC training + U2O fine-tuning
│   ├── HumanoidEnv.py          # Humanoid environment (obs_dim=376, action_dim=17)
│   ├── AdroitEnv.py            # Adroit manipulation environment
│   └── evaluate.py             # Fitness evaluation
├── u2o/                        # U2O integration (SFAgent-based)
│   ├── agent.py                # SFAgent: actor + successor features + skill inference
│   ├── fb_modules.py           # Network building blocks (Actor, ForwardMap, BackwardMap)
│   ├── networks.py             # Feature learners (HILP, Laplacian, Contrastive, ICM, etc.)
│   ├── replay_buffer.py        # Episode-based replay buffer with future/goal sampling
│   ├── utils.py                # Utilities (TruncatedNormal, soft_update, schedule, etc.)
│   ├── wrappers.py             # EpisodeMonitor wrapper
│   └── pretrain.py             # Pretraining script (data collection + offline training)
├── human_feedback/             # Elo scoring for human evaluation
└── cfg/                        # Hydra configs
    ├── generate.yaml           # Evolution + U2O config
    └── train.yaml              # RL training config
```

## U2O Architecture

The U2O module implements the full **Successor Feature (SF)** framework from [u2o_zsrl](https://arxiv.org/abs/2408.14785):

- **HILP Feature Learner**: Dual φ networks with expectile regression learning temporal-distance value functions. Supports pluggable feature learners (HILP, Laplacian, Contrastive, ICM, AutoEncoder, Identity).
- **Successor Features**: Dual forward maps F(s,z,a) predicting φ(s'), enabling skill-conditioned Q-learning via Q(s,a) = F(s,z,a)^T z.
- **Skill-Conditioned Actor**: Policy π(a|s,z) outputting TruncatedNormal actions, conditioned on observation and skill vector.
- **Skill Inference**: Given a task reward r, solve z* = lstsq(φ, r) to find the optimal skill direction without retraining.
- **Mixed Online/Offline Training**: Fine-tuning combines newly collected online data with pretrained offline replay buffer.

## U2O Configuration

All U2O parameters in `cfg/generate.yaml`:

```yaml
u2o:
  enabled: false                    # toggle U2O on/off
  pretrained_dir: ${root_dir}/u2o_pretrained
  # SFAgent architecture (scaled for Humanoid 376-dim obs)
  z_dim: 100                       # skill vector dimension
  hidden_dim: 2048                 # network hidden size
  phi_hidden_dim: 1024             # feature network hidden size
  feature_dim: 1024                # feature output dimension
  feature_learner: hilp            # feature learning method
  hilp_discount: 0.98              # HILP temporal discount
  hilp_expectile: 0.5              # expectile for value learning
  # Training
  lr: 1e-4                         # learning rate
  batch_size: 2048                 # batch size
  finetune_steps: 50000            # fine-tuning steps per reward function
  discount: 0.98                   # MDP discount
  sf_target_tau: 0.01              # soft update rate for target networks
  # Replay buffer
  future: 0.99                     # future sampling discount
  p_randomgoal: 0.375              # random goal sampling probability
```

Pretrain script arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `./u2o_pretrained` | Output directory |
| `--z_dim` | `100` | Skill vector dimension |
| `--hidden_dim` | `2048` | Network hidden size |
| `--phi_hidden_dim` | `1024` | Feature network hidden size |
| `--feature_dim` | `1024` | Feature output dimension |
| `--feature_learner` | `hilp` | Feature learning method (hilp, laplacian, contrastive, icm, etc.) |
| `--collection_episodes` | `10000` | Episodes of random data to collect |
| `--pretrain_steps` | `5000000` | Offline training steps |
| `--batch_size` | `2048` | Batch size for training |
| `--lr` | `1e-4` | Learning rate |
| `--discount` | `0.98` | MDP discount factor |
| `--hilp_discount` | `0.98` | HILP temporal discount |
| `--hilp_expectile` | `0.5` | Expectile for value learning |
| `--device` | `auto` | `cuda` or `cpu` |

## Other Utilities
* The prompts are listed in `prompts/` folder.
* Elo scoring in `human_feedback/` folder.


## Citation

### To cite the original REvolve paper:
```bibtex
@inproceedings{hazra2025revolve,
	title        = {{RE}volve: Reward Evolution with Large Language Models using Human Feedback},
	author       = {Rishi Hazra and Alkis Sygkounas and Andreas Persson and Amy Loutfi and Pedro Zuidberg Dos Martires},
	year         = 2025,
	booktitle    = {The Thirteenth International Conference on Learning Representations},
	url          = {https://openreview.net/forum?id=cJPUpL8mOw}
}
```

### U2O reference:
```bibtex
@inproceedings{lee2024unsupervised,
	title        = {Unsupervised-to-Online Reinforcement Learning},
	author       = {Junsu Lee and Seohong Park and Sergey Levine},
	year         = 2024,
	url          = {https://arxiv.org/abs/2408.14785}
}
```
