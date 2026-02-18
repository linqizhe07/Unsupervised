import os
import sys
import traceback
from typing import Optional, Tuple, List, Callable

import numpy as np

sys.path.append(os.environ["ROOT_PATH"])
from rewards_database import RevolveDatabase, EurekaDatabase
from modules import *
import utils
import prompts
from evolutionary_utils.custom_environment import CustomEnvironment
import absl.logging as logging
from functools import partial
from utils import *
import hydra
import os


def load_reward_function(file_path: str) -> Callable:
    """
    Load and return a callable reward function from a file.
    Args:
    - file_path (str): Path to the file containing the reward function.

    Returns:
    - Callable: Executable reward function.
    """
    with open(file_path, "r") as f:
        reward_fn_str = f.read()

    # Use define_function_from_string to make it executable
    reward_func, _ = define_function_from_string(reward_fn_str)

    if reward_func is None:
        raise ValueError("Failed to load a valid reward function.")

    return reward_func


def is_valid_reward_fn(generated_fn: Callable, generated_fn_str: str, args: List[str]):
    """validate generated heuristic function"""
    if generated_fn is None or args is None:
        raise utils.InvalidFunctionError("Generated function has no arguments.")
    env_state = CustomEnvironment().env_state
    env_vars = env_state.keys()
    # check if all args are valid env args
    if set(args).intersection(set(env_vars)) != set(args):
        raise utils.InvalidFunctionError("Generated function uses invalid arguments.")
    # TODO: test the following for REvolve
    # Get the return type annotation
    return_statements = utils.validate_callable_no_signature(generated_fn_str)
    if not return_statements:
        raise utils.InvalidFunctionError(
            "The function does not have any return statements."
        )
    return True


def generate_valid_reward(
        reward_generation: RewardFunctionGeneration,
        in_context_prompt: str,
        max_trials: int = 10,
) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    single function generation until valid
    :param reward_generation: initialized class of RewardFunctionGeneration
    :param in_context_prompt: in context prompt used by the LLM to generate the new fn
    :param max_trials: maximum number of trials to generate
    :return: return valid function string
    """
    # used in case we want to provide python error feedbacks to the LLM
    error_feedback = ""
    trials = 0
    while True:
        try:
            rew_func_str = reward_generation.generate_rf(
                in_context_prompt + error_feedback
            )
            rew_func, args = utils.define_function_from_string(rew_func_str)
            is_valid_reward_fn(rew_func, rew_func_str, args)
            logging.info("Valid reward function generated.")
            error_feedback = ""
            break  # Exit the loop if successful
        except Exception as e:
            logging.info(f"Specific error caught: {e}")
        logging.info("Attempting to generate a new function due to an error.")
        trials += 1
        if trials >= max_trials:
            logging.info("Exceeded max trials.")
            return None, None
    return rew_func_str, args


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.environ["ROOT_PATH"], "cfg"),
    config_name="generate",
)
def main(cfg):
    env_name = cfg.environment.name

    # U2O configuration
    u2o_enabled = cfg.get("u2o", {}).get("enabled", False)
    u2o_pretrained_dir = None
    u2o_cfg = {}
    if u2o_enabled:
        u2o_pretrained_dir = cfg.u2o.pretrained_dir
        u2o_cfg = {
            "z_dim": cfg.u2o.get("z_dim", 50),
            "hidden_dim": cfg.u2o.get("hidden_dim", 1024),
            "phi_hidden_dim": cfg.u2o.get("phi_hidden_dim", 512),
            "feature_dim": cfg.u2o.get("feature_dim", 512),
            "feature_learner": cfg.u2o.get("feature_learner", "hilp"),
            "lr": cfg.u2o.get("lr", 1e-4),
            "batch_size": cfg.u2o.get("batch_size", 1024),
            "finetune_steps": cfg.u2o.get("finetune_steps", 1000),
        }
        print(f"U2O mode enabled. Pretrained dir: {u2o_pretrained_dir}")

    # Initialize wandb
    wandb_run = None
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("enabled", False):
        try:
            import wandb
            run_name = wandb_cfg.get("run_name") or (
                f"{cfg.evolution.baseline}_run{cfg.data_paths.run}"
            )
            wandb_run = wandb.init(
                project=wandb_cfg.get("project", "revolve"),
                entity=wandb_cfg.get("entity"),
                name=run_name,
                config={
                    "baseline": cfg.evolution.baseline,
                    "num_generations": cfg.evolution.num_generations,
                    "individuals_per_generation": cfg.evolution.individuals_per_generation,
                    "num_islands": cfg.database.num_islands,
                    "max_island_size": cfg.database.max_island_size,
                    "crossover_prob": cfg.database.crossover_prob,
                    "migration_prob": cfg.database.migration_prob,
                    "u2o_enabled": u2o_enabled,
                },
            )
            print(f"[wandb] Initialized: {wandb_run.url}")
        except ImportError:
            logging.warning("wandb not installed; disabling wandb logging.")
            wandb_run = None

    system_prompt = prompts.types["system_prompt"]
    env_input_prompt = prompts.types["env_input_prompt"]

    reward_generation = RewardFunctionGeneration(
        system_prompt=system_prompt, env_input=env_input_prompt
    )

    # create log directory
    log_dir = cfg.data_paths.output_logs
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tracker = utils.DataLogger(os.path.join(log_dir, "progress.log"))

    # define a schedule for temperature of sampling
    temp_scheduler = partial(
        utils.linear_decay,
        initial_temp=cfg.database.initial_temp,
        final_temp=cfg.database.final_temp,
        num_iterations=cfg.evolution.num_generations,
    )
    if "revolve" in cfg.evolution.baseline:
        database = partial(
            RevolveDatabase,
            num_islands=cfg.database.num_islands,
            max_size=cfg.database.max_island_size,
            crossover_prob=cfg.database.crossover_prob,
            migration_prob=cfg.database.migration_prob,
            reward_fn_dir=cfg.database.rewards_dir,
            baseline=cfg.evolution.baseline,
        )
    else:
        database = partial(
            EurekaDatabase,
            num_islands=1,  # Eureka for a single island
            max_size=cfg.database.max_island_size,
            reward_fn_dir=cfg.database.rewards_dir,
            baseline=cfg.evolution.baseline,
        )

    for generation_id in range(0, cfg.evolution.num_generations):
        # fix the temperature for sampling
        temperature = temp_scheduler(iteration=generation_id)
        print(
            f"\n========= Generation {generation_id} | Model: {cfg.evolution.baseline} | temperature: {round(temperature, 2)} =========="
        )
        # load all groups if iteration_id > 0, else initialize empty islands
        rewards_database = database(load_islands=not generation_id == 0)

        rew_fn_strings = []  # valid rew fns
        # fitness_scores = []
        island_ids = []
        counter_ids = []
        # metrics_dicts = []
        policies = []

        # for each generation, produce new individuals via mutation or crossover
        for counter_id in range(cfg.evolution.individuals_per_generation):
            if generation_id == 0:  # initially, uniformly populate the islands
                # TODO: to avoid corner cases, populate all islands uniformly
                island_id = random.choice(range(rewards_database.num_islands))
                in_context_samples = (None, None)
                operator_prompt = ""
                parent_checkpoint_path = None
                logging.info(
                    f"Generation {generation_id}, Counter {counter_id}: island_id={island_id}, type={type(island_id)}"
                )

            else:  # gen_id > 0: start the evolutionary process
                (
                    in_context_samples,
                    island_id,
                    operator,
                    parent_checkpoint_path,
                ) = rewards_database.sample_in_context(
                    cfg.few_shot, temperature
                )  # weighted sampling of islands and corresponding individuals
                operator = (
                    f"{operator}_auto" if "auto" in cfg.evolution.baseline else operator
                )
                operator_prompt = prompts.types[operator]

            # each sample in 'in_context_samples' is a tuple of (fn_path: str, fitness_score: float)
            in_context_prompt = RewardFunctionGeneration.prepare_in_context_prompt(
                in_context_samples,
                operator_prompt,
                evolve=generation_id > 0,
                baseline=cfg.evolution.baseline,
            )
            logging.info(f"Designing reward function for counter {counter_id}")
            # generate valid fn str
            reward_func_str, _ = generate_valid_reward(
                reward_generation, in_context_prompt
            )
            if reward_func_str is None:
                logging.info(
                    f"Skipping counter {counter_id}: failed to generate a valid reward function."
                )
                continue

            try:
                # initialize RL agent policy with the generated reward function
                policy = TrainPolicy(
                    reward_func_str,
                    generation_id,
                    counter_id,
                    island_id,
                    cfg.evolution.baseline,  # cfg.evolution.baseline
                    cfg.database.rewards_dir,
                )
                if u2o_enabled:
                    # Build wandb config for fine-tune subprocess logging
                    ft_wandb_cfg = None
                    if wandb_run is not None:
                        ft_wandb_cfg = {
                            "project": wandb_cfg.get("project", "revolve"),
                            "entity": wandb_cfg.get("entity"),
                            "group": wandb_run.group or wandb_run.name,
                        }
                    policy.enable_u2o(
                        pretrained_dir=u2o_pretrained_dir,
                        u2o_cfg=u2o_cfg,
                        parent_checkpoint_path=parent_checkpoint_path,
                        wandb_cfg=ft_wandb_cfg,
                    )
                policies.append(policy)
                island_ids.append(island_id)
                rew_fn_strings.append(reward_func_str)
                counter_ids.append(counter_id)
            except Exception as e:
                logging.info(f"Error initializing TrainPolicy: {e}")
                logging.error("Traceback:")
                logging.error(traceback.format_exc())

                logging.info(
                    "Oops, something broke again :( Let's toss it out the window and call it modern art!"
                )
                continue

        # run policies in parallel
        # if no valid reward fn
        if len(policies) == 0:
            logging.info("No valid reward functions. Hence, no policy trains required.")
            continue
        # train policies in parallel
        logging.info(f"Training {len(policies)} policies in parallel.")
        ckpt_and_performance_paths = train_policies_in_parallel(policies)
        logging.info("Policy training finished.")

        # evaluate performance for generated reward functions
        logging.info("Evaluating trained policies in parallel.")
        metrics_dicts = evaluate_policies_in_parallel(ckpt_and_performance_paths)
        fitness_scores = [metric_dict["fitness"] for metric_dict in metrics_dicts]
        logging.info("Evaluation finished.")

        # store individuals only if it improves overall island fitness
        # for initialization, we don't use this step
        if generation_id > 0:
            rewards_database.add_individuals_to_islands(
                [generation_id] * len(island_ids),
                counter_ids,
                rew_fn_strings,
                fitness_scores,
                metrics_dicts,
                island_ids,
            )
        else:  # initialization step (generation = 0)
            rewards_database.seed_islands(
                [generation_id] * len(island_ids),
                counter_ids,
                rew_fn_strings,
                fitness_scores,
                metrics_dicts,
                island_ids,
            )

        island_info = [
            {
                island_id: {
                    f"{gen_id}_{count_id}": fitness
                    for gen_id, count_id, fitness in zip(
                        island.generation_ids, island.counter_ids, island.fitness_scores
                    )
                }
            }
            for island_id, island in enumerate(rewards_database._islands)
        ]
        tracker.log({"generation": generation_id, "islands": island_info})

        # wandb logging
        if wandb_run is not None:
            all_island_scores = [
                island.fitness_scores for island in rewards_database._islands
                if island.size > 0
            ]
            all_scores_flat = [s for scores in all_island_scores for s in scores]

            log_dict = {
                "generation": generation_id,
                "temperature": temperature,
                # current generation stats
                "gen/num_individuals": len(fitness_scores),
                "gen/mean_fitness": float(np.mean(fitness_scores)) if fitness_scores else 0,
                "gen/max_fitness": float(np.max(fitness_scores)) if fitness_scores else 0,
                "gen/min_fitness": float(np.min(fitness_scores)) if fitness_scores else 0,
                "gen/std_fitness": float(np.std(fitness_scores)) if fitness_scores else 0,
                # global stats across all islands
                "global/best_fitness": float(np.max(all_scores_flat)) if all_scores_flat else 0,
                "global/mean_fitness": float(np.mean(all_scores_flat)) if all_scores_flat else 0,
                "global/total_individuals": sum(
                    island.size for island in rewards_database._islands
                ),
            }

            # per-island stats
            for iid, island in enumerate(rewards_database._islands):
                if island.size > 0:
                    log_dict[f"island_{iid}/best_fitness"] = island.best_fitness_score
                    log_dict[f"island_{iid}/avg_fitness"] = float(island.average_fitness_score)
                    log_dict[f"island_{iid}/size"] = island.size

            wandb_run.log(log_dict, step=generation_id)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
