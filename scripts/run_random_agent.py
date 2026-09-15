"""Run a complete, seeded, uniformly random legal-action episode."""

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.episode import run_episode
from agents.random_agent import RandomAgent
from arknights_sim.environment import ArknightsEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="0-1")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-decisions", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    env = ArknightsEnv()
    episode = run_episode(
        env, RandomAgent(args.seed), stage_id=args.stage, seed=args.seed,
        max_decisions=args.max_decisions,
    )
    result = env.result(episode.final_state)
    result_data = asdict(result) if is_dataclass(result) else result
    output = {
        "agent": "RandomAgent", "stage": args.stage, "seed": args.seed,
        "result": result_data, "decisions": episode.decisions,
        "wall_time": episode.wall_time,
        "average_legal_actions": sum(r.legal_action_count for r in episode.history) / max(1, episode.decisions),
        "maximum_legal_actions": max((r.legal_action_count for r in episode.history), default=0),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            **output,
            "history": [record.to_dict() for record in episode.history],
        }, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
