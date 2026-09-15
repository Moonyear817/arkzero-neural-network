"""Small YAML settings adapter; no tensor/core imports on the GUI thread."""

import math
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import ClassVar

import yaml

from desktop.paths import DATA_DIR, WORKSPACE_ROOT


class ConfigService:
    DEFAULTS: ClassVar[dict] = {
        "stage": "0-1",
        "squad": ["char_500_noirc", "char_208_melan"],
        "device": "auto",
        "iterations": 1,
        "episodes_per_iteration": 2,
        "mcts_simulations": 16,
        "batch_size": 32,
        "replay_buffer_size": 50000,
        "learning_rate": 0.001,
        "c_puct": 1.5,
        "dirichlet_alpha": 0.3,
        "dirichlet_epsilon": 0.25,
        "temperature": 1.0,
        "seed": 12345,
        "training_steps_per_iteration": 2,
        "evaluation_episodes": 1,
        "checkpoint_interval": 1,
        "checkpoint_dir": "outputs/desktop_training/checkpoints",
        "output_dir": "outputs/desktop_training",
        "training_data_dir": "训练数据",
        "num_threads": 1,
        "max_decisions": 512,
        "horizon": 300,
    }

    def __init__(self, root=None):
        self.root = Path(root) if root is not None else WORKSPACE_ROOT
        self.config_dir = self.root / "configs"
        self.default_path = self.config_dir / "desktop_default.yaml"
        self.session_path = self.config_dir / "gui_session.yaml"

    def list_configs(self):
        return sorted(self.config_dir.glob("*.yaml"))

    def load(self, path=None):
        path = Path(path) if path else self.default_path
        if not path.is_absolute():
            path = self.root / path
        with path.open(encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream)
        if not isinstance(loaded, dict):
            raise TypeError("Training config must contain a YAML mapping")
        result = deepcopy(self.DEFAULTS)
        result.update(loaded)
        return self.validate(result)

    def validate(self, config):
        result = deepcopy(self.DEFAULTS)
        result.update(deepcopy(config))
        if not isinstance(result.get("stage"), str) or not result["stage"].strip():
            raise ValueError("stage must be a nonempty string")
        if not isinstance(result.get("squad"), list) or (not result["squad"] and not result.get("auto_squad", False)
                and result.get('training_scope') != 'account_guards_chapter8'):
            raise ValueError("squad must be a nonempty list of operator IDs")
        if any(
            not isinstance(item, str) or not item.strip() for item in result["squad"]
        ):
            raise ValueError("squad must contain nonempty operator IDs")
        if len(set(result["squad"])) != len(result["squad"]):
            raise ValueError("squad cannot contain duplicate operator IDs")
        result["device"] = str(result["device"]).lower()
        if result["device"] not in ("auto", "cpu", "mps"):
            raise ValueError("device must be Auto, CPU or MPS")
        for key in (
            "iterations",
            "mcts_simulations",
            "episodes_per_iteration",
            "batch_size",
            "replay_buffer_size",
            "training_steps_per_iteration",
            "evaluation_episodes",
            "checkpoint_interval",
        ):
            if type(result.get(key)) is not int or result[key] <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(result.get("seed")) is not int or result["seed"] < 0:
            raise ValueError("seed must be a nonnegative integer")
        for key in (
            "learning_rate",
            "c_puct",
            "dirichlet_alpha",
            "dirichlet_epsilon",
            "temperature",
        ):
            value = result.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{key} must be a finite number")
            if value < 0 or (
                key in ("learning_rate", "dirichlet_alpha") and value == 0
            ):
                raise ValueError(f"{key} is outside its valid range")
        if result["dirichlet_epsilon"] > 1:
            raise ValueError("dirichlet_epsilon must be between 0 and 1")
        return result

    def prepare_runtime(self, config):
        result = self.validate(config)
        for key in (
            "checkpoint_dir",
            "output_dir",
            "training_data_dir",
            "evaluation_checkpoint",
            "resume_from",
            "data_dir",
            "account_snapshot",
            "account_draft",
        ):
            value = result.get(key)
            if value:
                path = Path(value).expanduser()
                result[key] = str(
                    (path if path.is_absolute() else self.root / path).resolve()
                )
        if not result.get("data_dir"):
            local_data = self.root / "data/real"
            result["data_dir"] = str(local_data if local_data.is_dir() else DATA_DIR)
        if not result.get("training_data_dir"):
            result["training_data_dir"] = str(self.root / "训练数据")
        return result

    def save_session(self, config, path=None):
        config = self.validate(config)
        destination = Path(path) if path else self.session_path
        if not destination.is_absolute():
            destination = self.root / destination
        if destination.resolve() == self.default_path.resolve():
            raise ValueError(
                "The default configuration is read-only; save a new GUI session"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                suffix=".yaml",
                dir=destination.parent,
                delete=False,
            ) as stream:
                temporary = Path(stream.name)
                yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
        return destination
