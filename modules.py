"""
Various stages of individual generation, training, and evaluation:
1. Reward Function Generation
2. Policy Training
3. Policy Evaluation
"""

import concurrent.futures
import json
import multiprocessing
import os
import time
from typing import Tuple, Optional, Dict, List

import absl.logging as logging
import hydra
import openai
from hydra.core.global_hydra import GlobalHydra
from openai import OpenAI

from rl_agent.evaluate import return_score, INVALID_FITNESS
from rl_agent.fitness_score import calculate_fitness_score
# from rl_agent.generate_scores import generate_behaviour
from rl_agent.main import run_training
from utils import parse_llm_output, serialize_dict, format_human_feedback

openai_api_key = os.environ["OPENAI_API_KEY"]
client = OpenAI(
    api_key=openai_api_key,
    base_url = "https://api.openai.com/v1")


# generates reward functions
class RewardFunctionGeneration:
    def __init__(self, system_prompt: str, env_input: str):
        # TODO: change system message based on Eureka
        self.system_prompt = system_prompt
        self.env_input = env_input  # env_class + task
        self.llm = "gpt-5.2-2025-12-11"

    def query_llm(self, in_context_prompt: str) -> Tuple[str, int, int]:
        response = client.chat.completions.create(
            model=self.llm,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt + "\n" + self.env_input,
                },
                {"role": "user", "content": in_context_prompt},
            ],
            max_completion_tokens=4096,
            reasoning_effort="none",
            temperature=1,
            top_p=1,
        )

        content = response.choices[0].message.content
        if content is None:
            raise openai.APIError(
                message="Model returned empty content (possibly truncated or refused)",
                request=None,
                body=None,
            )
        return (
            content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    @staticmethod
    def prepare_in_context_prompt(
            in_context_samples: Optional[List[Tuple[str, float]]],
            operator_prompt: str,
            evolve: bool,
            baseline: str,
    ) -> str:
        # prepares a prompt from in context examples sampled from RewardsDatabase
        in_context_samples_str = ""
        if not evolve:
            return in_context_samples_str
        for filename, fitness_score in in_context_samples:
            reward_history_file = filename.replace(
                "generated_fns", "reward_history"
            ).replace(".txt", ".json")
            # Skip individuals whose training hasn't completed yet (race condition)
            if not os.path.exists(reward_history_file):
                continue

            in_context_samples_str += "\n\n```python\n"
            with open(filename, "r") as f:
                in_context_samples_str += f.read()
            in_context_samples_str += "\n```\n"

            reward_history = []
            with open(reward_history_file, "r") as f:
                for line in f:
                    reward_history.append(json.loads(line.strip()))

            combined_components = {}
            for entry in reward_history:
                for key, value in entry["episode_components"].items():
                    if key not in combined_components:
                        combined_components[key] = []
                    combined_components[key].append(value)
            in_context_samples_str += f"fitness score: {fitness_score}"
            in_context_samples_str += f"\n{serialize_dict(combined_components)}"
            if "auto" not in baseline:
                # human feedback
                human_feedback_file = filename.replace(
                    "generated_fns", "human_feedback"
                )
                human_feedback = open(human_feedback_file, "r").read()
                human_feedback = format_human_feedback(human_feedback)
                in_context_samples_str += f"\nhuman feedback: {human_feedback}"
        operator_prompt = operator_prompt.replace(
            "\n\n<EXAMPLES>", in_context_samples_str
        )
        operator_prompt = operator_prompt.replace("<EPISODES>", "100")
        return operator_prompt

    def generate_rf(self, in_context_prompt: str, max_retries: int = 30) -> str:
        for attempt in range(max_retries):
            try:
                raw_llm_output, _, _ = self.query_llm(in_context_prompt)
                parsed_function_str = parse_llm_output(raw_llm_output)
                return parsed_function_str
            except (openai.RateLimitError, openai.APIError, openai.APITimeoutError) as e:
                logging.info(f"API error (attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(min(10 * (2 ** min(attempt, 4)), 160))  # exponential backoff, cap at 160s
                continue
        raise openai.APIError(
            message=f"Failed after {max_retries} retries",
            request=None,
            body=None,
        )


class TrainPolicy:
    """
    Train RL Policy
    """

    def __init__(
            self,
            reward_fn_str: str,
            generation_id: int,
            counter_id: int,
            island_id: int,
            baseline: str,
            output_log: str,
            env_name: str = "HumanoidEnv",
            gpu_id: int = 0,
    ):
        self.train_cfg = None
        self._load_train_cfg()

        self.reward_func_str = reward_fn_str
        self.island_id = island_id
        self.generation_id = generation_id
        self.counter_id = counter_id
        self.baseline = baseline  # ['revolve', 'revolve_auto', 'eureka', 'eureka_auto']
        self.output_log = output_log
        self.env_name = env_name
        self.gpu_id = gpu_id
        # U2O parameters (optional, set via enable_u2o())
        self.u2o_enabled = False
        self.pretrained_dir = None
        self.u2o_cfg = {}
        self.parent_checkpoint_path = None
        # wandb config for fine-tune logging
        self.wandb_cfg = None
        logging.info(
            f"Initializing TrainPolicy: generation_id={generation_id}, island_id={island_id}, type(island_id)={type(island_id)}"
        )

    def enable_u2o(self, pretrained_dir: str, u2o_cfg: dict = None,
                   parent_checkpoint_path: str = None, wandb_cfg: dict = None):
        """Enable U2O mode: SFAgent skill inference + fine-tune."""
        self.u2o_enabled = True
        self.pretrained_dir = pretrained_dir
        self.u2o_cfg = u2o_cfg or {}
        self.parent_checkpoint_path = parent_checkpoint_path
        self.wandb_cfg = wandb_cfg

    def _load_train_cfg(self):
        logging.info("Loading train cfg")

        # Ensure ROOT_PATH exists
        root_path = os.environ.get("ROOT_PATH")
        if not root_path:
            raise EnvironmentError("ROOT_PATH environment variable is not set.")

        # Convert absolute path to relative
        config_relative_path = os.path.relpath(
            os.path.join(root_path, "cfg"), start=os.getcwd()
        )

        # Clear Hydra global state if already initialized
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # Initialize Hydra with the relative config path
        with hydra.initialize(config_path=config_relative_path):
            self.train_cfg = hydra.compose(config_name="train")
            logging.info("Training Config loaded")

    def train_policy(self) -> Tuple[str, str, str]:
        # This will define the compute_reward function dynamically

        reward_history_file = os.path.join(
            self.output_log,
            f"island_{self.island_id}/reward_history/{self.generation_id}_{self.counter_id}.json",
        )

        checkpoint_file = os.path.join(
            self.output_log,
            f"island_{self.island_id}/model_checkpoints/{self.generation_id}_{self.counter_id}.zip",
        )

        velocity_file_path = os.path.join(
            self.output_log,
            f"island_{self.island_id}/velocity_logs/velocity_{self.generation_id}_{self.counter_id}.txt",
        )

        # Adroit success log: same dir as reward_history_file but .txt extension.
        # Written by AdroitHandDoorEnv.step() as
        # "Episode finished at step {steps}: Success=True/False".
        # Used by calculate_fitness_score() to compute the paper's fitness formula.
        adroit_success_log_path = os.path.join(
            self.output_log,
            f"island_{self.island_id}/reward_history/{self.generation_id}_{self.counter_id}.txt",
        )

        model_checkpoint_path = os.path.join(
            self.output_log, f"island_{self.island_id}/model_checkpoints"
        )

        fitness_file = os.path.join(
            self.output_log,
            f"island_{self.island_id}/fitness_scores/{self.generation_id}_{self.counter_id}.txt",
        )

        log_dir = os.path.join(
            self.output_log,
            f"island_{self.island_id}/log_dir/{self.generation_id}_{self.counter_id}",
        )

        # Select the evaluation log that matches the fitness function for this env:
        #   AdroitHandDoorEnv → success log (paper's formula: σ = -steps/700 + 75/70)
        #   HumanoidEnv       → velocity log (average goal_distance around peak)
        eval_log_path = (
            adroit_success_log_path
            if self.env_name == "AdroitHandDoorEnv"
            else velocity_file_path
        )

        if self.u2o_enabled:
            # U2O path: SFAgent skill inference + fine-tune
            from rl_agent.main import run_training_u2o

            run_training_u2o(
                self.reward_func_str,
                self.island_id,
                self.generation_id,
                self.counter_id,
                reward_history_file,
                model_checkpoint_path,
                fitness_file,
                velocity_file_path,
                self.output_log,
                log_dir,
                pretrained_dir=self.pretrained_dir,
                u2o_cfg=self.u2o_cfg,
                parent_checkpoint_path=self.parent_checkpoint_path,
                wandb_cfg=self.wandb_cfg,
                env_name=self.env_name,
                gpu_id=self.gpu_id,
            )
        else:
            # Original path: from-scratch (or fine-tune from parent checkpoint)
            run_training(
                self.reward_func_str,
                self.island_id,
                self.generation_id,
                self.counter_id,
                reward_history_file,
                model_checkpoint_path,
                fitness_file,
                velocity_file_path,
                self.output_log,
                log_dir,
                env_name=self.env_name,
                wandb_cfg=self.wandb_cfg,
                parent_checkpoint_path=self.parent_checkpoint_path,
                gpu_id=self.gpu_id,
            )
        return checkpoint_file, eval_log_path, self.env_name


# human evaluation, fitness functions
class RewardFunctionEvaluation:
    """
    Fitness Function Evaluator
    """

    def __init__(self, baseline: str):
        self.baseline = baseline

    # def generate_behavior(self, filename: str) -> Dict:
    #     # be provided
    #     reward_history_dict = json.load(open(filename, "r"))
    #     return reward_history_dict

    @staticmethod
    def evaluate_behavior(
            eval_log_path: str,
            env_name: str = "HumanoidEnv",
    ) -> Dict[str, float]:
        if env_name == "AdroitHandDoorEnv":
            # Paper's formula: σ = -steps/700 + 75/70 (success), 0 (failure)
            # averaged over all episodes. Returns 0 if no episodes logged.
            try:
                fitness_score = calculate_fitness_score(eval_log_path)
            except FileNotFoundError:
                logging.error(
                    f"Adroit success log not found: {eval_log_path}; "
                    f"returning fallback fitness {INVALID_FITNESS}."
                )
                fitness_score = INVALID_FITNESS
        else:
            fitness_score = return_score(eval_log_path)
        return {"fitness": fitness_score}


def train_policies_in_parallel(
        policy_classes: List[TrainPolicy],
        max_workers: int = None,
) -> List[Tuple[str, str, str]]:
    """
    submit multiple training policies in parallel
    max_workers: limit parallel processes to avoid OOM (None = all at once)
    Returns list of (checkpoint_path, eval_log_path, env_name) tuples.
    """
    multiprocessing.set_start_method("spawn", force=True)
    if max_workers is None:
        max_workers = len(policy_classes)
    # Allocate result slots so the returned list preserves submission order
    # even though we collect completions out of order.
    results = [None] * len(policy_classes)
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
    ) as executor:
        future_to_idx = {
            executor.submit(policy_class.train_policy): i
            for i, policy_class in enumerate(policy_classes)
        }
        for future in concurrent.futures.as_completed(future_to_idx):
            i = future_to_idx[future]
            try:
                results[i] = future.result()
            except Exception as e:
                logging.error(
                    f"Training worker {i} failed (island={policy_classes[i].island_id}, "
                    f"gen={policy_classes[i].generation_id}, "
                    f"counter={policy_classes[i].counter_id}): {e}"
                )
                # Empty eval_log_path → FileNotFoundError → INVALID_FITNESS in evaluate_behavior.
                results[i] = (None, "", policy_classes[i].env_name)
    return results


def evaluate_policies_in_parallel(
        ckpt_and_performance_paths: List[Tuple[str, str, str]]
) -> List[Dict[str, float]]:
    """
    Submit evaluation tasks in parallel with checkpoint paths, eval log paths, and env names.
    """
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(ckpt_and_performance_paths)
    ) as executor:
        futures = [
            executor.submit(RewardFunctionEvaluation.evaluate_behavior, eval_log_path, env_name)
            for _, eval_log_path, env_name in ckpt_and_performance_paths
        ]

        fitness_dicts = [future.result() for future in futures]
    return fitness_dicts
