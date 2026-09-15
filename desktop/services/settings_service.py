"""Validated, atomic local settings with explicit workspace paths."""

import json
import os
from pathlib import Path

from desktop.paths import DATA_DIR, WORKSPACE_ROOT


class SettingsService:
    def __init__(self, root=None):
        self.root = Path(root or WORKSPACE_ROOT)
        self.path = self.root / "configs/desktop_settings.json"
        self.defaults = {
            "device": "Auto",
            "config": str(self.root / "configs/desktop_default.yaml"),
            "data_dir": str(DATA_DIR if root is None else self.root / "data/real"),
            "checkpoint_dir": str(self.root / "checkpoints"),
            "log_dir": str(self.root / "logs"),
            "theme": "System",
            "language": "zh_CN",
            "auto_save_checkpoint": True,
            "refresh_ms": 1000,
            "max_log_lines": 10000,
            "visualization_fps": 20,
        }

    def load(self):
        if not self.path.exists():
            return dict(self.defaults)
        result = {**self.defaults, **json.loads(self.path.read_text())}
        self.validate(result)
        return result

    def validate(self, values):
        if values.get("language", "zh_CN") not in ("en", "zh_CN"):
            raise ValueError("Unknown interface language")
        if values["device"] not in ("Auto", "CPU", "MPS") or values["theme"] not in (
            "System",
            "Light",
            "Dark",
        ):
            raise ValueError("Unknown device or theme")
        for key, low, high in (
            ("refresh_ms", 250, 10000),
            ("max_log_lines", 100, 100000),
            ("visualization_fps", 1, 60),
        ):
            if type(values[key]) is not int or not low <= values[key] <= high:
                raise ValueError(f"Invalid {key}")
        for key in ("config", "data_dir", "checkpoint_dir", "log_dir"):
            if not isinstance(values[key], str) or not values[key].strip():
                raise ValueError(f"Missing {key}")

    def save(self, values):
        result = {**self.defaults, **values}
        self.validate(result)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(temporary, self.path)
        return result
