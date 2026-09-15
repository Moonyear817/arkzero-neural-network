from pathlib import Path

from PySide6.QtCore import Signal

from desktop.paths import DATA_DIR
from desktop.services.research_service import load_game_data

from .research_controller import ResearchController


class DataController(ResearchController):
    loaded = Signal(object)

    def __init__(self, data_dir=None, root=None, parent=None):
        super().__init__(parent)
        self.data_dir = (
            Path(data_dir)
            if data_dir
            else (Path(root) / "data/real" if root else DATA_DIR)
        )
        self.snapshot = None

    def load(self):
        directory = self.data_dir
        return self._start(
            lambda check: load_game_data(directory, check), self._receive
        )

    def _receive(self, result):
        self.snapshot = result
        self.loaded.emit(result)
