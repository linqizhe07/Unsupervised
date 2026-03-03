# Zero-Shot Revolve

## Overview

This project addresses a fundamental inefficiency in LLM-guided reward design: each candidate reward function is evaluated by training a policy from scratch, making the search expensive and the fitness signal noisy. We introduce a system that evolves **both reward functions and policy networks** jointly across generations.

The system combines three components:

- **LLM-based reward evolution**: an island-based evolutionary algorithm uses an LLM to generate, mutate, and crossover reward function code across parallel islands
- **Unsupervised pretraining**: a skill-conditioned policy is pretrained without any task reward using HILP feature learning and successor features, building a reusable behavioral prior
- **Policy inheritance**: each candidate fine-tunes from its parent's checkpoint conditioned on a task-inferred skill direction z*, so policy weights evolve across generations alongside the reward function

```
[Pretrain — once per environment]        [Evolution loop — per reward candidate]

  Humanoid: RND exploration  ──┐
                               ├──→ ReplayBuffer
  AdroitHand: D4RL datasets ──┘   (cloned + expert + human, circular buffer)         │
                                    HILP φ(s): temporal-distance features
                                         │
                                    SF F(s,z,a): skill-conditioned Q
                                         │
                                    Actor π(a|s,z)
                                         │
                               agent_checkpoint.pt
                                         │
                        ┌────────────────┘
                        │
              LLM generates reward fn
                        │
              z* = lstsq(φ, r)      ← skill inference, no retraining
                        │
              Fine-tune π from parent checkpoint
                        │
              Evaluate fitness → update island
```

---

## Installation

```shell for running locally
git clone https://github.com/Linqizhe07/zeroshotRevolve.git
cd zeroshotRevolve
conda create -n revolve python=3.10
conda activate revolve
pip install -e .
```

```shell for instance
python3 -m venv venv1(1 for Adroithand 2 for Humanoid)
source venv/bin/activate
pip install -e .
```

### Required Packages

Core dependencies (installed by `pip install -e .`):

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥2.0 | SFAgent, HILP, actor networks |
| `numpy` | — | Array operations |
| `scipy` | — | Skill inference (`lstsq`) |
| `mujoco` | 2.3.7 | Physics simulation |
| `gymnasium[mujoco]` | 0.29.1 | Humanoid environment |
| `gymnasium-robotics` | 1.2.4 | AdroitHand environment |
| `stable-baselines3` | 2.3.2 | SAC training in evolution loop |
| `hydra-core` | 1.3.2 | Config management |
| `openai` | — | LLM reward generation |
| `tensorboard` | ≥2.12.0 | Metrics logging (≥2.12 required for NumPy ≥1.24 compatibility) |
| `h5py` | 3.10.0 | Dataset I/O |

Additional (install as needed):

```shell
# AdroitHand pretraining: D4RL offline datasets
pip install git+https://github.com/Farama-Foundation/d4rl.git

# Experiment tracking (optional)
pip install wandb
```

> `d4rl` is not on PyPI and requires MuJoCo 2.x. Datasets are downloaded automatically on first use and cached in `~/.d4rl/`.

---

## Supported Environments

| `--env` | `environment.name` | obs_dim | action_dim | Description |
|---------|-------------------|---------|------------|-------------|
| `humanoid` | `HumanoidEnv` | 376 | 17 | MuJoCo Humanoid locomotion |
| `adroit_door` | `AdroitHandDoorEnv` | 39 | 28 | 28-DOF dexterous hand, door opening |

---

## Pretraining

Pretraining is **one-time per environment**. It has two phases:

1. **Data collection** — environment-specific (see below)
2. **Offline training** — identical for both environments: train SFAgent on the collected buffer using HILP feature learning and successor features, with no task reward

### Humanoid — ZSRL Pipeline (RND exploration)

The Humanoid environment has no offline dataset. The agent collects its own data using RND (Random Network Distillation): a predictor network is trained to match a fixed random target; states with high prediction error are novel and yield high intrinsic reward, driving the actor toward unexplored regions.

```
HumanoidEnv (obs=376, action=17)
    │
    ├── RND predictor/target networks (obs → 128-dim embedding)
    ├── Lightweight REINFORCE actor trained on intrinsic reward
    │
    └──→ ReplayBuffer (reward stored as 0.0 — only trajectories matter)
```

```shell
export ROOT_PATH='/home/ubuntu/zeroshotRevolve'
python -m u2o.pretrain \
    --env humanoid \
    --output_dir ./u2o_pretrained_humanoid \
    --exploration rnd \
    --collection_episodes 10000 \
    --max_buffer_episodes 10000 \
    --pretrain_steps 1000000 \
    --batch_size 1024 \
    --z_dim 50 \
    --hidden_dim 1024 \
    --phi_hidden_dim 512 \
    --feature_dim 512 \
    --feature_learner hilp \
    --wandb_project pretrain-humanoid
```

### AdroitHand — GCRL Pipeline (D4RL offline dataset)

The AdroitHand environment has high-quality offline datasets from D4RL. The environment is **never instantiated** during pretraining — the dataset is loaded directly into the replay buffer.

```
D4RL dataset (door-human-v1 / door-cloned-v1 / door-expert-v1)
    │
    ├── Split by terminals/timeouts into episodes
    ├── terminal → discount=0.0 | timeout → discount=1.0
    │
    └──→ ReplayBuffer (obs=39, action=28)
```

Available datasets:

| Dataset | Episodes | Notes |
|---------|----------|-------|
| `door-human-v1` | ~25 | MOCO-captured human demos — highest quality but very few |
| `door-cloned-v1` | ~5000 | BC rollouts from human demos — largest, most diverse coverage |
| `door-expert-v1` | ~200 | Expert policy rollouts — reliable successes |

**Multi-dataset loading** (recommended): pass a comma-separated list to `--d4rl_dataset`. Datasets are loaded in order into a circular replay buffer; **load large datasets first and small/high-quality ones last**, so the small datasets are never overwritten.

```
Loading order:  cloned (fills buffer) → expert (overwrites oldest cloned) → human (idem)
Buffer result:  9775 cloned + 200 expert + 25 human  (with max_buffer_episodes=10000)
Sampling:       uniform over all episodes — each dataset contributes proportionally to its size
```

> **Why not equal allocation?** `door-human-v1` has only ~25 episodes. Capping cloned at `max_buffer_episodes / 3 = 3333` to "balance" it would waste 3308 buffer slots and reduce cloned's state-coverage contribution without meaningfully increasing human's share (it would still be < 1 % of training batches).

```shell
pip uninstall h5py -y && pip install h5py
export ROOT_PATH='/home/ubuntu/zeroshotRevolve'
python -m u2o.pretrain \
    --env adroit_door \
    --output_dir ./u2o_pretrained_adroit \
    --exploration d4rl \
    --d4rl_dataset door-cloned-v1,door-expert-v1,door-human-v1 \
    --max_buffer_episodes 10000 \
    --pretrain_steps 1000000 \
    --batch_size 1024 \
    --z_dim 50 \
    --hidden_dim 1024 \
    --phi_hidden_dim 512 \
    --feature_dim 512 \
    --feature_learner hilp \
    --wandb_project pretrain-adroit
```

Single-dataset usage still works unchanged:

```shell
python -m u2o.pretrain --exploration d4rl --d4rl_dataset door-cloned-v1 ...
```

### Offline Training (Phase 2) — Both Environments

Each training step:

1. Sample a batch of `(s, a, r, s', s_future)` from the replay buffer — `s_future` is geometrically sampled within the same episode (`future=0.99`)
2. Normalize observations with a running mean/std tracker (Welford)
3. Sample skill vectors `z` uniformly from the unit sphere; replace 50% with whitened phi embeddings (`mix_ratio=0.5`) to encourage SF learning on meaningful directions
4. **HILP loss**: train dual φ networks to approximate temporal distance `V(s, g) ≈ −d(s→g)` using expectile regression; EMA-update target φ networks (`τ=0.005`)
5. **SF loss**: train dual F networks so `F(s,z,a)ᵀz ≈ Q(s,a)`; bootstrap target uses EMA target φ and EMA successor net (`τ=0.01`)
6. **Actor loss**: maximize `Q = F(s,z,π(s,z))ᵀz`

Outputs:

| File | Description |
|------|-------------|
| `agent_checkpoint.pt` | Pretrained SFAgent (actor + SF networks + HILP φ) |
| `replay_buffer.npz` | Collected/loaded transitions for offline mixing during fine-tuning |
| `pretrain_config.json` | Full hyperparameter record |

---

## Evolution Loop

### Multi-GPU Distribution

Each `TrainPolicy` worker is automatically assigned a GPU via round-robin across all available GPUs:

```
policies[0] → cuda:0
policies[1] → cuda:1
...
policies[7] → cuda:7
policies[8] → cuda:0   (wraps around)
...
```

`database.num_gpus` controls **parallelism**, not GPU count:

| Value | Behaviour |
|-------|-----------|
| `0` | Run all candidates in parallel (one per GPU slot, round-robin) |
| `N` | Run at most N candidates in parallel at a time |

The number of physical GPUs used is always `torch.cuda.device_count()`, regardless of `num_gpus`. On a single-GPU machine every worker uses `cuda:0` (same as before this change).

### Experiment 1: Baseline (no pretraining)

**Humanoid:**
```shell
export ROOT_PATH='/home/ubuntu/zeroshotRevolve'
export OPENAI_API_KEY='<your openai key>'
python main.py \
    evolution.baseline=revolve_auto \
    environment.name="HumanoidEnv" \
    evolution.num_generations=7 \
    evolution.individuals_per_generation=16 \
    database.num_islands=5 \
    database.num_gpus=0 \
    data_paths.run=100 \
    u2o.enabled=false \
    wandb.project=humanoid-baseline
```

**AdroitHand:**
```shell
export ROOT_PATH='/home/ubuntu/zeroshotRevolve'
export OPENAI_API_KEY='<your openai key>'
python main.py \
    evolution.baseline=revolve_auto \
    environment.name="AdroitHandDoorEnv" \
    evolution.num_generations=7 \
    evolution.individuals_per_generation=16 \
    database.num_islands=5 \
    database.num_gpus=0 \
    data_paths.run=200 \
    u2o.enabled=false \
    wandb.project=adroit-baseline
```

### Experiment 2: Full System (with U2O pretraining)

Run pretrain first (see above), then:

**Humanoid + U2O:**
```shell
export ROOT_PATH='/home/ubuntu/zeroshotRevolve'
export OPENAI_API_KEY='<your openai key>'
python main.py \
    u2o.enabled=true \
    u2o.pretrained_dir=./u2o_pretrained_humanoid \
    environment.name="HumanoidEnv" \
    evolution.num_generations=7 \
    evolution.individuals_per_generation=16 \
    database.num_islands=5 \
    database.num_gpus=0 \
    data_paths.run=182 \
    wandb.project=humanoid-u2o
```

**AdroitHand + U2O:**
```shell
export ROOT_PATH='/home/ubuntu/zeroshotRevolve'
export OPENAI_API_KEY='<your openai key>'
python main.py \
    u2o.enabled=true \
    u2o.pretrained_dir=./u2o_pretrained_adroit \
    environment.name="AdroitHandDoorEnv" \
    evolution.num_generations=7 \
    evolution.individuals_per_generation=16 \
    database.num_islands=5 \
    database.num_gpus=0 \
    data_paths.run=126 \
    wandb.project=adroit-u2o
```

Per-candidate fine-tuning:
1. **Skill inference**: evaluate the reward function on the offline replay buffer → solve `z* = lstsq(φ, r)` (no gradient, no retraining)
2. **Fine-tune**: collect online rollouts with π(·|s, z*), mix 50/50 with offline buffer, update SF networks and actor with task rewards
3. **Checkpoint**: save policy weights for the next generation to inherit

### LLM Prompt Selection

The system message sent to the LLM is assembled per environment:

| Environment | System prompt | Env input |
|-------------|--------------|-----------|
| `HumanoidEnv` | `prompts/system_prompt` | `prompts/env_input` |
| `AdroitHandDoorEnv` | `prompts/system_prompt_adroit` | `prompts/env_input_adroit` |

The final LLM system message = `system_prompt` + `\n` + `env_input`. The `system_prompt` files contain the task description and reward-writing rules; the `env_input` files contain the full observation-space documentation (variable names, indices, types) that the LLM uses to write syntactically correct reward functions.

---

## Project Structure

```
Revolve/
├── main.py                     # Evolutionary loop entry point
├── modules.py                  # Reward generation, policy training, fitness evaluation
├── rewards_database.py         # Island-based population management
├── prompts/                    # LLM prompts (selected per environment at runtime)
│   ├── system_prompt           # Humanoid: task description + reward-writing rules
│   ├── system_prompt_adroit    # AdroitHand: task description + reward-writing rules
│   ├── env_input               # Humanoid observation space doc
│   ├── env_input_adroit        # AdroitHand observation space doc (obs indices, action space)
│   ├── mutation_auto           # In-context mutation prompt
│   ├── crossover_auto          # In-context crossover prompt
│   ├── mutation                # Mutation prompt (non-auto baseline)
│   └── crossover               # Crossover prompt (non-auto baseline)
├── rl_agent/
│   ├── main.py                 # SAC training + U2O fine-tuning
│   ├── HumanoidEnv.py          # obs=376, action=17
│   ├── AdroitEnv.py            # obs=39, action=28; fitness = door_hinge angle
│   └── evaluate.py             # Fitness scoring
├── u2o/
│   ├── pretrain.py             # Pretraining script (Phase 1 + Phase 2)
│   ├── agent.py                # SFAgent: HILP + successor features + skill inference
│   ├── networks.py             # Feature learners (HILP, ICM, Laplacian, ...)
│   ├── fb_modules.py           # Actor, ForwardMap building blocks
│   ├── replay_buffer.py        # Episode buffer with geometric future sampling
│   └── utils.py                # TruncatedNormal, soft_update, schedule, ...
├── human_feedback/             # Elo scoring
└── cfg/
    ├── generate.yaml           # Evolution + U2O config
    └── train.yaml              # SAC training config
```

---

## Key Hyperparameters

`cfg/generate.yaml` (U2O section — must match pretrain args):

```yaml
u2o:
  enabled: false
  pretrained_dir: ${root_dir}/u2o_pretrained
  z_dim: 50
  hidden_dim: 1024
  phi_hidden_dim: 512
  feature_dim: 512
  feature_learner: hilp
  hilp_discount: 0.98
  hilp_expectile: 0.5
  lr: 1e-4
  batch_size: 1024
  finetune_steps: 500000
  discount: 0.98
  sf_target_tau: 0.01
  future: 0.99
  p_randomgoal: 0.375
```

`u2o.pretrain` CLI arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--env` | `humanoid` | `humanoid` or `adroit_door` |
| `--exploration` | `random` | `rnd` (Humanoid) or `d4rl` (AdroitHand) |
| `--d4rl_dataset` | — | Required for `d4rl`. Comma-separated list, e.g. `door-cloned-v1,door-expert-v1,door-human-v1`. Load large datasets first so small ones survive in the circular buffer. |
| `--collection_episodes` | `10000` | RND episodes to collect (unused for d4rl) |
| `--max_buffer_episodes` | `10000` | Buffer capacity in episodes |
| `--pretrain_steps` | `1000000` | Offline training steps |
| `--batch_size` | `1024` | Training batch size |
| `--z_dim` | `50` | Skill vector dimension |
| `--hidden_dim` | `1024` | SF/actor hidden size |
| `--phi_hidden_dim` | `512` | HILP φ network hidden size |
| `--feature_dim` | `512` | Intermediate feature dimension |
| `--hilp_discount` | `0.98` | HILP temporal discount γ |
| `--hilp_expectile` | `0.5` | Expectile τ for value regression |
| `--lr` | `1e-4` | Learning rate |
| `--discount` | `0.98` | MDP discount for replay buffer |
| `--device` | `auto` | `cuda`, `mps`, or `cpu` |
| `--wandb_project` | — | W&B project (optional) |

---

## Acknowledgements

Inspired by [REvolve](https://openreview.net/forum?id=cJPUpL8mOw) (Hazra et al., ICLR 2025) and [Unsupervised-to-Online RL](https://arxiv.org/abs/2408.14785) (Lee et al., 2024).
