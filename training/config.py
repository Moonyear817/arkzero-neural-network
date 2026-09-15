"""Small validated AlphaZero configuration; no simulator or Qt dependencies."""

import math
from copy import deepcopy
from pathlib import Path

import yaml

DEFAULT_CONFIG = {
    "stage": "0-1",
    "squad": ["char_500_noirc", "char_208_melan"],
    "device": "auto",
    "num_threads": 1,
    "seed": 12345,
    "iterations": 10,
    "episodes_per_iteration": 5,
    "mcts_simulations": 32,
    "max_depth": 128,
    "max_decisions": 512,
    "horizon": 300,
    "c_puct": 1.5,
    "dirichlet_alpha": 0.3,
    "dirichlet_epsilon": 0.25,
    "prior_uniform_mix": 0.5,
    "temperature": 1.0,
    "temperature_schedule": [
        {"iteration": 0, "value": 1.0},
        {"iteration": 6, "value": 0.75},
        {"iteration": 11, "value": 0.5},
    ],
    "uniform_mix_schedule": [
        {"iteration": 0, "value": 0.5},
        {"iteration": 3, "value": 0.3},
        {"iteration": 6, "value": 0.15},
        {"iteration": 11, "value": 0.05},
    ],
    "skip_forced_actions": True,
    "future_events_enabled": True,
    "planner_enabled": True,
    "top_k_actions": 8,
    "progressive_widening": True,
    "widening_coefficient": 2.0,
    "widening_exponent": 0.5,
    "decision_log_top_k": 8,
    "joint_training_mode": "alternating",
    "curriculum": False,
    "curriculum_success_rate": 0.8,
    "curriculum_evaluations": 3,
    "batch_size": 32,
    "replay_buffer_size": 50000,
    "training_steps_per_iteration": 50,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "value_loss_weight": 1.0,
    "gradient_clip": 5.0,
    "evaluation_episodes": 10,
    "evaluation_seed": 80000,
    "checkpoint_interval": 1,
    "promote_on_equal": True,
    "checkpoint_dir": "checkpoints/alphazero_001",
    "output_dir": "outputs/alphazero_001",
    "training_data_dir": None,
    "data_dir": None,
    "resume_from": None,
    "operator_progression": {},
    "skill_overrides": {},
}


def default_config():
    return deepcopy(DEFAULT_CONFIG)


def validate_config(config):
    config = deepcopy(config)
    legacy_root = "/Users/yihao/Documents/ChatGPT/Ark Zero"
    current_root = Path(__file__).resolve().parents[1]
    for key in ("data_dir", "checkpoint_dir", "output_dir", "training_data_dir", "resume_from", "warm_start", "evaluation_checkpoint"):
        value = config.get(key)
        if isinstance(value, str) and value.startswith(legacy_root + "/"):
            config[key] = str(current_root / value[len(legacy_root) + 1:])

    result = default_config()
    result.update(deepcopy(config))
    if result["training_data_dir"] is not None and (
        not isinstance(result["training_data_dir"], (str, Path))
        or not str(result["training_data_dir"]).strip()
    ):
        raise ValueError("training_data_dir must be a nonempty path or null")
    if result["training_data_dir"] is not None:
        result["training_data_dir"] = str(result["training_data_dir"])
    from arknights_sim.data.progression import Progression
    result['operator_progression'] = Progression.from_dict(result['operator_progression']).to_dict()
    if not isinstance(result['skill_overrides'], dict):
        raise ValueError('skill_overrides must map operator IDs to zero-based skill indices')
    for key, index in result['skill_overrides'].items():
        if not isinstance(key, str) or not key:
            raise ValueError('Invalid skill override operator ID')
        Progression(skill_index=index)
    for key in (
        "iterations",
        "episodes_per_iteration",
        "mcts_simulations",
        "max_depth",
        "max_decisions",
        "batch_size",
        "replay_buffer_size",
        "training_steps_per_iteration",
        "evaluation_episodes",
        "num_threads",
        "checkpoint_interval",
        "curriculum_evaluations",
        "top_k_actions",
        "decision_log_top_k",
    ):
        if type(result[key]) is not int or result[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("seed", "evaluation_seed"):
        if type(result[key]) is not int or result[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in (
        "horizon",
        "c_puct",
        "dirichlet_alpha",
        "dirichlet_epsilon",
        "prior_uniform_mix",
        "temperature",
        "learning_rate",
        "weight_decay",
        "value_loss_weight",
        "gradient_clip",
        "widening_coefficient",
        "widening_exponent",
    ):
        v = result[key]
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or v < 0
        ):
            raise ValueError(f"{key} must be a finite nonnegative number")
    for key in ("learning_rate", "horizon", "dirichlet_alpha", "gradient_clip"):
        if result[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("dirichlet_epsilon", "prior_uniform_mix"):
        if result[key] > 1:
            raise ValueError(f"{key} must be at most 1")
    for key in ('future_events_enabled', 'planner_enabled', 'progressive_widening'):
        if type(result[key]) is not bool:
            raise ValueError(f'{key} must be boolean')
    if result['widening_coefficient'] <= 0 or not 0 < result['widening_exponent'] <= 1:
        raise ValueError('Widening coefficient must be positive and exponent in (0, 1]')
    result["device"] = str(result["device"]).lower()
    if result["device"] not in ("auto", "cpu", "mps"):
        raise ValueError("device must be auto, cpu or mps")
    if (
        not isinstance(result["squad"], list)
        or (not result["squad"] and not result.get("auto_squad", False)
            and result.get('training_scope') != 'account_guards_chapter8')
        or len(result["squad"]) > 12
        or len(set(result["squad"])) != len(result["squad"])
    ):
        raise ValueError("squad must contain 1–12 distinct operator IDs")
    if type(result.get("auto_squad", False)) is not bool:
        raise ValueError("auto_squad must be boolean")
    if result['joint_training_mode'] not in ('alternating', 'joint'):
        raise ValueError('joint_training_mode must be alternating or joint')
    if type(result['curriculum']) is not bool:
        raise ValueError('curriculum must be boolean')
    threshold = result['curriculum_success_rate']
    if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not 0 < threshold <= 1:
        raise ValueError('curriculum_success_rate must be in (0, 1]')
    stages = result.get('stage_pool') or [result['stage']]
    if not isinstance(stages, list) or not all(isinstance(s, str) and s for s in stages) or len(set(stages)) != len(stages):
        raise ValueError('stage_pool must contain distinct stage IDs')
    if result['curriculum'] and result['evaluation_episodes'] < len(stages):
        raise ValueError('Curriculum evaluation must cover every stage in stage_pool')
    if type(result.get("squad_size", 6)) is not int or not 1 <= result.get("squad_size", 6) <= 12:
        raise ValueError("squad_size must be 1–12")
    for key in ("temperature_schedule", "uniform_mix_schedule"):
        previous = -1
        for item in result[key]:
            if (
                set(item) != {"iteration", "value"}
                or type(item["iteration"]) is not int
                or item["iteration"] < previous
            ):
                raise ValueError(f"{key} requires ordered iteration/value mappings")
            if (
                not math.isfinite(item["value"])
                or item["value"] < 0
                or (key == "uniform_mix_schedule" and item["value"] > 1)
            ):
                raise ValueError(f"Invalid {key} value")
            previous = item["iteration"]
    return result


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError("Config must contain a mapping")
    return validate_config(value)


def scheduled_value(config, field, iteration):
    schedule = (
        "uniform_mix_schedule"
        if field == "prior_uniform_mix"
        else "temperature_schedule"
    )
    value = config[field]
    for item in config[schedule]:
        if iteration >= item["iteration"]:
            value = item["value"]
    return value


def resolve_device(requested="auto"):
    import torch

    requested = str(requested).lower()
    mps = torch.backends.mps.is_available()
    if requested == "mps" and not mps:
        raise RuntimeError("MPS was requested but is unavailable")
    if requested not in ("auto", "cpu", "mps"):
        raise ValueError("Unsupported device")
    return torch.device(
        "mps" if requested == "mps" or (requested == "auto" and mps) else "cpu"
    )
