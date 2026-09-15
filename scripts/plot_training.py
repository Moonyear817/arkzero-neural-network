"""Plot recorded AlphaZero iterations, retaining failed and zero-success runs."""

import argparse
import json
from pathlib import Path
from common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="?", type=Path, default=ROOT / "outputs/alphazero_001/training_metrics.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.metrics.read_text().splitlines() if line.strip()]
    if not records:
        parser.error("No iterations in metrics file")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = args.output or args.metrics.parent / "plots"
    destination.mkdir(parents=True, exist_ok=True)
    iterations = [row["iteration"] for row in records]
    charts = (
        ("success_rate", "Zero-leak clear rate", [("selfplay_success_rate", "Self-play"), ("evaluation_success_rate", "Evaluation")]),
        ("loss", "Training loss", [("policy_loss", "Policy"), ("value_loss", "Value"), ("total_loss", "Total")]),
        ("replay_size", "Replay samples", [("replay_size", "Buffer size")]),
    )
    for filename, ylabel, series in charts:
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for key, label in series:
            pairs = [(x, row.get(key)) for x, row in zip(iterations, records) if row.get(key) is not None]
            if pairs:
                ax.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", markersize=4, label=label)
        ax.set(xlabel="Iteration", ylabel=ylabel, title="Arknights Zero · " + ylabel)
        if filename == "success_rate":
            ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
        fig.savefig(destination / (filename + ".png"), dpi=160)
        plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
