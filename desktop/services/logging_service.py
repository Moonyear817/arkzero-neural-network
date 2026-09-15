"""Full disk logs, bounded queued delivery to the UI, no per-event redraw."""

import logging
import queue
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LogEntry:
    time: str
    category: str
    level: str
    text: str


class CategoryFilter(logging.Filter):
    def __init__(self, category=None, errors=False):
        super().__init__()
        self.category, self.errors = category, errors

    def filter(self, record):
        return record.levelno >= logging.ERROR if self.errors else getattr(record, "category", "APP") == self.category


class UIQueueHandler(logging.Handler):
    def __init__(self, mailbox):
        super().__init__()
        self.mailbox = mailbox

    def emit(self, record):
        category = "ERROR" if record.levelno >= logging.ERROR else getattr(record, "category", "APP")
        entry = LogEntry(self.formatter.formatTime(record, "%H:%M:%S"), category,
                         record.levelname, self.format(record))
        try:
            self.mailbox.put_nowait(entry)
        except queue.Full:
            # Disk handlers have already persisted the full event stream.
            pass


class LoggingService:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.mailbox = queue.Queue(maxsize=20000)
        self.logger = logging.getLogger("arknights.desktop")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.handlers = []
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for file, category in (("app", "APP"), ("simulator", "SIM"), ("training", "TRAIN"), ("mcts", "MCTS"), ("errors", "ERROR")):
            handler = logging.FileHandler(self.directory / (file + ".log"), encoding="utf-8")
            handler.setFormatter(formatter)
            handler.addFilter(CategoryFilter(category, errors=category == "ERROR"))
            self.logger.addHandler(handler)
            self.handlers.append(handler)
        ui = UIQueueHandler(self.mailbox)
        ui.setFormatter(formatter)
        self.logger.addHandler(ui)
        self.handlers.append(ui)

    def log(self, category, message, error=False):
        self.logger.log(logging.ERROR if error else logging.INFO, message, extra={"category": category})

    def drain(self, limit=500):
        result = []
        for _ in range(limit):
            try:
                result.append(self.mailbox.get_nowait())
            except queue.Empty:
                break
        return result

    def close(self):
        for handler in self.handlers:
            self.logger.removeHandler(handler)
            handler.close()
        self.handlers.clear()

