# Zero-Shot Revolve with frozen phi & z_star

## Overview

This project addresses a fundamental inefficiency in LLM-guided reward design: each candidate reward function is evaluated by training a policy from scratch, making the search expensive and the fitness signal noisy. We introduce a system that evolves **both reward functions and policy networks** jointly across generations.

The system combines three components into a unified pipeline:

- **LLM-based reward evolution**: an island-based evolutionary algorithm uses an LLM to generate, mutate, and crossover reward function code, maintaining a population of candidates across parallel islands
- **Unsupervised pretraining**: a skill-conditioned policy is pretrained entirely without task supervision using successor features and HILP feature learning, building a reusable behavioral prior over the state space
- **Policy inheritance**: each candidate fine-tunes from its parent's checkpoint rather than from scratch, conditioned on a task-inferred skill direction z* — so policy weights evolve across generations alongside the reward function

```
[Pretrain (once, per environment)]       [Evolution Loop (per candidate)]

Env (Humanoid / Adroit / ...) +          LLM generates/mutates reward function
random or RND exploration                        |
        |                                Skill inference: z* = lstsq(φ, r)
HILP feature learner φ(s)                       |
        |                                Fine-tune from parent checkpoint
Successor features F(s,z,a)                     |
        |                                Evaluate fitness → update island
Skill-conditioned actor π(a|s,z)
        |
agent_checkpoint.pt + replay_buffer.npz
```

## Setup
```shell
git clone https://github.com/Linqizhe07/zeroshotRevolve.git
cd Unsupervised
conda create -n "revolve" python=3.10
conda activate revolve
pip install -e .
```

## Run

### Supported Environments

| `--env` / `environment.name` | obs_dim | action_dim | Description |
|------------------------------|---------|------------|-------------|
| `humanoid` / `HumanoidEnv` | 376 | 17 | MuJoCo Humanoid locomotion |
| `adroit` / `AdroitHandDoorEnv` | 39 | 28 | 28-DOF dexterous hand, door opening task |

> To add a new environment: (1) implement a `create_dummy_<env>_env()` factory in `u2o/pretrain.py`, (2) register it in `ENV_REGISTRY`, (3) add the env class to `utils.py` load_environment.

---

### Experiment 1: Baseline (no pretraining)

**Humanoid:**
```shell
export ROOT_PATH='/home/ubuntu/adroithand/zeroshotRevolve'
export OPENAI_API_KEY='<your openai key>'
python main.py \
        evolution.baseline=revolve_auto \
        environment.name="HumanoidEnv" \
        evolution.num_generations=5 \
        evolution.individuals_per_generation=12 \
        database.num_islands=3 \
        database.num_gpus=0 \
        data_paths.run=1 \
        u2o.enabled=false \
        wandb.project=humanoid-baseline
```

**Adroit Hand:**
```shell
export ROOT_PATH='/home/ubuntu/adroithand/zeroshotRevolve'
export OPENAI_API_KEY='<your openai key>'
python main.py \
        evolution.baseline=revolve_auto \
        environment.name="AdroitHandDoorEnv" \
        evolution.num_generations=5 \
        evolution.individuals_per_generation=12 \
        database.num_islands=3 \
        database.num_gpus=0 \
        data_paths.run=1 \
        u2o.enabled=false \
        wandb.project=adroit-baseline
```

---

### Experiment 2: Full System (with U2O pretraining)

#### Step 1: Pretrain (one-time, per environment)

Collect exploration data and train the skill-conditioned policy offline. The agent learns temporal-distance features φ(s) and skill-conditioned successor features F(s,z,a) without any task reward.

**Humanoid:**
```shell
export ROOT_PATH='/home/ubuntu/adroithand/zeroshotRevolve'
python -m u2o.pretrain \
        --env humanoid \
        --output_dir ./u2o_pretrained_humanoid \
        --z_dim 50 \
        --hidden_dim 1024 \
        --phi_hidden_dim 512 \
        --feature_dim 512 \
        --feature_learner hilp \
        --collection_episodes 10000 \
        --max_buffer_episodes 10000 \
        --pretrain_steps 1000000 \
        --batch_size 1024 \
        --exploration rnd \
        --wandb_project revolve-u2o-humanoid
```

**Adroit Hand:**
```shell
export ROOT_PATH='/home/ubuntu/adroithand/zeroshotRevolve'
python -m u2o.pretrain \
        --env adroit \
        --output_dir ./u2o_pretrained_adroit \
        --z_dim 50 \
        --hidden_dim 1024 \
        --phi_hidden_dim 512 \
        --feature_dim 512 \
        --feature_learner hilp \
        --collection_episodes 10000 \
        --max_buffer_episodes 10000 \
        --pretrain_steps 1000000 \
        --batch_size 1024 \
        --exploration rnd \
        --wandb_project revolve-u2o-adroit
```

| Output File | Description |
|-------------|-------------|
| `agent_checkpoint.pt` | Pretrained policy (actor + successor features + HILP feature learner) |
| `replay_buffer.npz` | Collected exploration transitions for offline data mixing |
| `pretrain_config.json` | Full config for reproducibility (env, obs_dim, action_dim, hyperparams) |

#### Step 2: Run Evolution

**Humanoid + U2O:**
```shell
export ROOT_PATH='/home/ubuntu/adroithand/zeroshotRevolve'
export OPENAI_API_KEY='<your openai key>'
python main.py \
        u2o.enabled=true \
        u2o.pretrained_dir=./u2o_pretrained_humanoid \
        environment.name="HumanoidEnv" \
        evolution.num_generations=5 \
        evolution.individuals_per_generation=12 \
        database.num_islands=3 \
        database.num_gpus=0 \
        data_paths.run=1 \
        wandb.project=humanoid-u2o
```

**Adroit Hand + U2O:**
```shell
export ROOT_PATH='/home/ubuntu/adroithand/zeroshotRevolve'
export OPENAI_API_KEY='<your openai key>'
python main.py \
        u2o.enabled=true \
        u2o.pretrained_dir=./u2o_pretrained_adroit \
        environment.name="AdroitHandDoorEnv" \
        evolution.num_generations=5 \
        evolution.individuals_per_generation=12 \
        database.num_islands=3 \
        database.num_gpus=0 \
        data_paths.run=1 \
        wandb.project=adroit-u2o
```

Each candidate reward function in the evolutionary loop goes through:
1. **Skill inference**: evaluate the candidate reward on the offline replay buffer, solve z* = lstsq(φ, r) to find the optimal skill direction
2. **Fine-tune**: collect online data with policy conditioned on z*, mix with offline buffer, update successor features and actor with task rewards
3. **Checkpoint**: save policy weights as this individual's checkpoint for the next generation to inherit

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
│   ├── main.py                 # SAC training + fine-tuning
│   ├── HumanoidEnv.py          # Humanoid environment (obs_dim=376, action_dim=17)
│   ├── AdroitEnv.py            # Adroit manipulation environment
│   └── evaluate.py             # Fitness evaluation
├── u2o/                        # Skill-conditioned policy module
│   ├── agent.py                # SFAgent: actor + successor features + skill inference
│   ├── fb_modules.py           # Network building blocks (Actor, ForwardMap, BackwardMap)
│   ├── networks.py             # Feature learners (HILP, Laplacian, Contrastive, ICM, etc.)
│   ├── replay_buffer.py        # Episode-based replay buffer with future/goal sampling
│   ├── utils.py                # Utilities (TruncatedNormal, soft_update, schedule, etc.)
│   ├── wrappers.py             # EpisodeMonitor wrapper
│   └── pretrain.py             # Pretraining script (data collection + offline training)
├── human_feedback/             # Elo scoring for human evaluation
└── cfg/                        # Hydra configs
    ├── generate.yaml           # Evolution + policy config
    └── train.yaml              # RL training config
```

## Architecture

The skill-conditioned policy implements the **Successor Feature (SF)** framework:

- **HILP Feature Learner**: dual φ networks with expectile regression learning temporal-distance value functions. Supports pluggable feature learners (HILP, Laplacian, Contrastive, ICM, AutoEncoder, Identity).
- **Successor Features**: dual forward maps F(s,z,a) predicting φ(s'), enabling skill-conditioned Q-learning via Q(s,a) = F(s,z,a)ᵀ z.
- **Skill-Conditioned Actor**: policy π(a|s,z) outputting TruncatedNormal actions, conditioned on observation and skill vector.
- **Skill Inference**: given a task reward r, solve z* = lstsq(φ, r) to find the optimal skill direction without any additional training.
- **Mixed Online/Offline Training**: fine-tuning combines newly collected online data with the pretrained offline replay buffer (50/50 split).

## Configuration

All parameters in `cfg/generate.yaml`:

```yaml
u2o:
  enabled: false                    # toggle pretraining on/off
  pretrained_dir: ${root_dir}/u2o_pretrained
  # Architecture (must match pretrain)
  z_dim: 50                         # skill vector dimension
  hidden_dim: 1024                  # network hidden size
  phi_hidden_dim: 512               # feature network hidden size
  feature_dim: 512                  # feature output dimension
  feature_learner: hilp             # feature learning method
  hilp_discount: 0.98               # HILP temporal discount
  hilp_expectile: 0.5               # expectile for value learning
  # Training
  lr: 1e-4                          # learning rate
  batch_size: 1024                  # batch size
  finetune_steps: 500000            # fine-tuning steps per reward function
  discount: 0.98                    # MDP discount
  sf_target_tau: 0.01               # soft update rate for target networks
  # Replay buffer
  future: 0.99                      # future sampling discount
  p_randomgoal: 0.375               # random goal sampling probability
```

Pretrain script arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--env` | `humanoid` | Environment: `humanoid`, `adroit` (extensible via `ENV_REGISTRY`) |
| `--output_dir` | `./u2o_pretrained` | Output directory |
| `--z_dim` | `50` | Skill vector dimension |
| `--hidden_dim` | `1024` | Network hidden size |
| `--phi_hidden_dim` | `512` | Feature network hidden size |
| `--feature_dim` | `512` | Feature output dimension |
| `--feature_learner` | `hilp` | Feature learning method (hilp, laplacian, contrastive, icm, etc.) |
| `--collection_episodes` | `10000` | Episodes of random/RND data to collect |
| `--max_buffer_episodes` | `10000` | Replay buffer capacity in episodes |
| `--pretrain_steps` | `1000000` | Offline training steps |
| `--batch_size` | `1024` | Batch size for training |
| `--lr` | `1e-4` | Learning rate |
| `--discount` | `0.98` | MDP discount factor |
| `--hilp_discount` | `0.98` | HILP temporal discount |
| `--hilp_expectile` | `0.5` | Expectile for value learning |
| `--device` | `auto` | `cuda` or `cpu` |

## Other Utilities
* LLM prompts are in `prompts/`.
* Human preference scoring (Elo) is in `human_feedback/`.

## Acknowledgements

Inspired by [REvolve](https://openreview.net/forum?id=cJPUpL8mOw) (Hazra et al., ICLR 2025) and [Unsupervised-to-Online RL](https://arxiv.org/abs/2408.14785) (Lee et al., 2024).
