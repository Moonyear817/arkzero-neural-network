#!/usr/bin/env python3
"""Matched real trainer timing, either headless or through the visible desktop.

Run each mode in a fresh process with the same repeats/seed and no other compute
job. Output includes every run; no best-run selection. This is a UI overhead
measurement, not a new learning-quality experiment.
"""

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cli", "gui"), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from desktop.paths import WORKSPACE_ROOT, DATA_DIR
    from training import AlphaZeroTrainer
    import psutil
    import torch
    torch.set_num_threads(1)
    config = dict(stage="0-1", squad=["char_500_noirc", "char_208_melan"],
        device="cpu", seed=54321, iterations=1, episodes_per_iteration=2,
        mcts_simulations=8, training_steps_per_iteration=2, batch_size=8,
        replay_buffer_size=1024, evaluation_episodes=2, num_threads=1,
        data_dir=str(DATA_DIR))
    work = WORKSPACE_ROOT / "outputs/desktop/overhead" / args.mode
    app = window = None
    ui_ticks = []
    if args.mode == "gui":
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QTimer, QEventLoop
        from PySide6.QtTest import QTest
        from desktop.main_window import MainWindow
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.show()
        def wait_for(predicate, timeout=180):
            loop = QEventLoop()
            timer = QTimer()
            timer.setInterval(10)
            timer.timeout.connect(lambda: loop.quit() if predicate() else None)
            deadline = QTimer()
            deadline.setSingleShot(True)
            deadline.timeout.connect(loop.quit)
            timer.start()
            deadline.start(timeout * 1000)
            if not predicate():
                loop.exec()
            timer.stop()
            deadline.stop()
            if not predicate():
                raise TimeoutError("Desktop benchmark operation timed out")
        wait_for(lambda: window.ui_state.mps != "Checking" and not window.data.is_busy() and not window.models.is_busy(), 60)
        ui_timer = QTimer(window)
        ui_timer.setInterval(100)
        def tick():
            ui_ticks.append(time.perf_counter())
            # Exercise navigation while computation is active.
            window.navigate("training" if len(ui_ticks) % 2 else "models")
        ui_timer.timeout.connect(tick)
        ui_timer.start()
    rows = []
    for index in range(args.repeats + 1):
        run = {**config, "checkpoint_dir": str(work / f"run_{index}/checkpoints"),
               "output_dir": str(work / f"run_{index}/metrics")}
        gc.collect()
        started = time.perf_counter()
        if args.mode == "cli":
            trainer = AlphaZeroTrainer(run)
            result = trainer.train()
            metrics = result["history"][-1]
            del trainer
        else:
            events, errors = [], []
            window.training.metrics_updated.connect(events.append)
            window.training.error.connect(errors.append)
            if not window.training.start(run):
                raise RuntimeError("Training controller refused start")
            wait_for(lambda: not window.training.is_busy)
            app.processEvents()
            window.training.metrics_updated.disconnect(events.append)
            window.training.error.disconnect(errors.append)
            if errors or not events:
                raise RuntimeError(str(errors) or "No metrics from GUI training")
            metrics = dict(events[-1])
        rows.append({"run": index, "warmup": index == 0,
            "wall_seconds": time.perf_counter() - started,
            "rss_mb_after": psutil.Process().memory_info().rss / 1024 ** 2,
            "metrics": {k: metrics.get(k) for k in (
                "policy_loss", "value_loss", "total_loss", "replay_size",
                "evaluation_success_rate", "selfplay_success_rate")}})
    output = {"mode": args.mode, "config": config, "runs": rows,
        "mean_wall_seconds": statistics.mean(r["wall_seconds"] for r in rows if not r["warmup"]),
        "median_wall_seconds": statistics.median(r["wall_seconds"] for r in rows if not r["warmup"]),
        "ui_timer_ticks": len(ui_ticks),
        "maximum_ui_tick_gap_seconds": max((b-a for a,b in zip(ui_ticks,ui_ticks[1:])), default=None)}
    if window:
        ui_timer.stop()
        window.dashboard.flush()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        window.navigate("training")
        QTest.qWait(100)
        window.grab().save(str(args.output.with_suffix(".png")))
        window.begin_shutdown()
        wait_for(lambda: window._shutdown_done)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
