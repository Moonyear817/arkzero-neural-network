from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Signal

from desktop.paths import DATA_DIR, WORKSPACE_ROOT
from desktop.services.research_service import (
    evaluate_model,
    inspect_model,
    list_checkpoints,
)

from .research_controller import ResearchController


class ModelController(ResearchController):
    models_changed = Signal(object)
    model_selected = Signal(object)
    model_loaded = Signal(object)
    evaluation_ready = Signal(object)
    comparison_ready = Signal(object)

    def __init__(self, checkpoint_dir=None, data_dir=None, parent=None):
        super().__init__(parent)
        self.checkpoint_dir = Path(checkpoint_dir or WORKSPACE_ROOT / "checkpoints")
        self.data_dir = Path(data_dir or DATA_DIR)
        self.models = ()
        self.selected = None
        self.loaded = None

    def refresh(self):
        directory = self.checkpoint_dir
        return self._start(
            lambda check: list_checkpoints(directory, check), self._receive
        )

    def _receive(self, models):
        self.models = models
        self.models_changed.emit(models)

    def select(self, path):
        target = str(Path(path).resolve())
        match = next((model for model in self.models if model.path == target), None)
        if match is None:
            raise ValueError("Select a listed checkpoint first")
        self.selected = match
        self.model_selected.emit(match)
        return match

    def load_model(self, path=None):
        target = str(path) if path else (self.selected.path if self.selected else None)
        if target is None:
            return False
        return self._start(
            lambda check: inspect_model(target, self.data_dir, check), self._loaded
        )

    def _loaded(self, snapshot):
        self.loaded = snapshot
        self.model_loaded.emit(snapshot)

    def evaluate(self, path=None, simulations=16, seeds=(80000,)):
        target = str(path) if path else (self.selected.path if self.selected else None)
        if target is None:
            return False
        return self._start(
            lambda check: evaluate_model(
                target,
                self.data_dir,
                simulations,
                tuple(seeds),
                check,
                WORKSPACE_ROOT / "outputs/desktop/evaluations",
            ),
            self._evaluated,
        )

    def _evaluated(self, result):
        metrics = dict(result.metrics)
        self.models = tuple(
            replace(model, metrics=tuple({**dict(model.metrics), **metrics}.items()))
            if model.path == result.path
            else model
            for model in self.models
        )
        self.models_changed.emit(self.models)
        self.evaluation_ready.emit(result)

    def compare(self, paths):
        requested = {str(Path(path).resolve()) for path in paths}
        if len(requested) != 2:
            raise ValueError("Choose exactly two different checkpoints")
        rows = tuple(model for model in self.models if model.path in requested)
        if len(rows) != 2:
            raise ValueError("Compare needs two listed checkpoints")
        self.comparison_ready.emit(rows)
        return rows
