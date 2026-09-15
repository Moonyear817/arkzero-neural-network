from pathlib import Path

from PySide6.QtCore import Signal

from desktop.paths import DATA_DIR, WORKSPACE_ROOT
from desktop.services.research_service import search_mcts, search_neural_mcts

from .research_controller import ResearchController


class MCTSController(ResearchController):
    result_ready = Signal(object)
    model_changed = Signal(str)

    def __init__(self, data_dir=None, parent=None):
        super().__init__(parent)
        self.data_dir = Path(data_dir or DATA_DIR)
        default = WORKSPACE_ROOT / "checkpoints/best.pt"
        self.checkpoint = str(default) if default.exists() else None
        self.result = None

    def set_model(self, path):
        self.checkpoint = str(Path(path).resolve())
        self.model_changed.emit(self.checkpoint)

    def search(
        self, stage_id="0-1", seed=12345, simulations=16, mode="puct", checkpoint=None
    ):
        selected = checkpoint or self.checkpoint
        if mode not in ("puct", "uct"):
            raise ValueError("Search mode must be puct or uct")

        def operation(check):
            if mode == "puct":
                if not selected:
                    raise ValueError(
                        "Load or select a checkpoint on Models before neural PUCT search"
                    )
                return search_neural_mcts(
                    self.data_dir, selected, stage_id, seed, simulations, check
                )
            return search_mcts(self.data_dir, stage_id, seed, simulations, check)

        return self._start(operation, self._receive)

    def _receive(self, result):
        self.result = result
        self.result_ready.emit(result)
