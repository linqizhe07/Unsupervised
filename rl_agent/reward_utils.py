import inspect
import typing as tp

import numpy as np


def _as_float32_array(x: tp.Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _add_humanoid_features(env_state: tp.Dict[str, tp.Any], obs: np.ndarray) -> None:
    dim = int(obs.shape[0])
    if dim not in (376, 378):
        return

    if dim == 376:
        torso_z_idx = 0
        orient_start = 1
        vel_start = 22
    else:
        torso_z_idx = 2
        orient_start = 3
        vel_start = 24

    env_state["torso_z"] = float(obs[torso_z_idx])
    env_state["torso_orientation"] = obs[orient_start : orient_start + 4]

    if obs.shape[0] > vel_start + 5:
        env_state["x_velocity"] = float(obs[vel_start])
        env_state["y_velocity"] = float(obs[vel_start + 1])
        env_state["z_velocity"] = float(obs[vel_start + 2])
        env_state["root_angular_velocity"] = obs[vel_start + 3 : vel_start + 6]


def _add_adroit_features(env_state: tp.Dict[str, tp.Any], obs: np.ndarray) -> None:
    if int(obs.shape[0]) != 39:
        return

    env_state["hand_qpos"] = obs[:27]
    env_state["latch_pos"] = float(obs[27])
    env_state["door_pos"] = float(obs[28])
    env_state["door_hinge"] = float(obs[28])
    env_state["palm_pos"] = obs[29:32]
    env_state["handle_pos"] = obs[32:35]
    env_state["palm_handle_delta"] = obs[35:38]
    env_state["door_open"] = float(obs[38])


def build_env_state_from_transition(
    obs: tp.Any,
    action: tp.Optional[tp.Any],
    next_obs: tp.Optional[tp.Any],
    *,
    reward_on: str = "next",
    joint_velocities: tp.Optional[tp.Any] = None,
    joint_forces: tp.Optional[tp.Any] = None,
) -> tp.Dict[str, tp.Any]:
    """Build a rich env_state dict from a transition.

    reward_on controls which observation is exposed as "observation":
    - "next": align with standard env.step reward timing r(s_{t+1}, ...)
    - "obs": use current state observation r(s_t, ...)
    """
    curr_obs = _as_float32_array(obs)
    nxt_obs = curr_obs if next_obs is None else _as_float32_array(next_obs)
    act = None if action is None else _as_float32_array(action)

    if reward_on == "next":
        reward_obs = nxt_obs
    elif reward_on == "obs":
        reward_obs = curr_obs
    else:
        raise ValueError(f"Unknown reward_on={reward_on}")

    env_state: tp.Dict[str, tp.Any] = {
        "observation": reward_obs,
        "obs": curr_obs,
        "next_obs": nxt_obs,
        "state": reward_obs,
        "current_observation": curr_obs,
        "next_observation": nxt_obs,
    }
    if act is not None:
        env_state["action"] = act
        env_state["act"] = act
    if joint_velocities is not None:
        env_state["joint_velocities"] = _as_float32_array(joint_velocities)
    if joint_forces is not None:
        env_state["joint_forces"] = _as_float32_array(joint_forces)

    env_state["observation_dim"] = int(reward_obs.shape[0])
    _add_humanoid_features(env_state, reward_obs)
    _add_adroit_features(env_state, reward_obs)
    return env_state


def call_reward_func_dynamically(
    reward_func: tp.Callable, env_state: tp.Dict[str, tp.Any]
) -> tp.Tuple[float, tp.Dict[str, tp.Any]]:
    params = inspect.signature(reward_func).parameters
    args_to_pass: tp.Dict[str, tp.Any] = {}
    missing: tp.List[str] = []
    for param_name, param in params.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param_name in env_state:
            args_to_pass[param_name] = env_state[param_name]
        elif param.default is inspect._empty:
            missing.append(param_name)

    if missing:
        raise KeyError(
            f"Missing required reward args: {missing}. "
            f"Available keys: {sorted(env_state.keys())[:40]}"
        )

    reward, reward_components = reward_func(**args_to_pass)
    if reward_components is None:
        reward_components = {}
    return float(reward), reward_components
