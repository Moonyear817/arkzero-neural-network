"""Compare fixed-seed evaluation, with regression agents isolated from training."""

import argparse
import json
from pathlib import Path
from common import ROOT


def summarize_baseline(label, factory, seeds, stage):
    from arknights_sim.environment import ArknightsEnv
    from agents.episode import run_episode
    trials = []
    for seed in seeds:
        env = ArknightsEnv()
        episode = run_episode(env, factory(seed), stage_id=stage, seed=seed, record_keys=False)
        trials.append({"seed": seed, **env.result(episode.final_state).to_dict(),
                       "decisions": episode.decisions, "wall_time": episode.wall_time})
    n = len(trials)
    return {"agent": label, "success_rate": sum(r["success"] for r in trials) / n,
            "average_kills": sum(r["kills"] for r in trials) / n,
            "average_leaks": sum(r["leaks"] for r in trials) / n,
            "average_decisions": sum(r["decisions"] for r in trials) / n,
            "wall_time": sum(r["wall_time"] for r in trials), "episodes": trials}


def main():
    from training.config import load_config
    from training.trainer import AlphaZeroTrainer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/alphazero_001.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20001, 20002, 20003])
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/alphazero/agent_comparison.json")
    args = parser.parse_args()
    config = load_config(args.config)
    config.update(device=args.device, mcts_simulations=args.simulations, evaluation_episodes=len(args.seeds))
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def retain(row):
        rows.append(row)
        document = {"stage": config.get("stage_id", "0-1"), "seeds": args.seeds,
                    "mcts_simulations": args.simulations, "device": args.device,
                    "checkpoint": str(args.checkpoint), "results": rows,
                    "evaluation_noise": False, "evaluation_temperature": 0,
                    "note": "No evaluation episode is added to replay. Scripted baseline is loaded only after neural evaluations."}
        args.output.write_text(json.dumps(document, indent=2, default=str) + "\n")
        print(json.dumps(row, default=str), flush=True)

    trainer = AlphaZeroTrainer(config)
    retain({"agent": "Untrained neural + PUCT", **trainer.evaluate(seeds=args.seeds)})
    retain({"agent": "Current AlphaZero", **trainer.evaluate(checkpoint=args.checkpoint, seeds=args.seeds)})
    # Baseline strategies never enter the trainer or neural search.
    from agents.random_agent import RandomAgent
    from agents.mcts_agent import PlainMCTSAgent
    from agents.mcts import MCTSConfig
    retain(summarize_baseline("RandomAgent", RandomAgent, args.seeds, config.get("stage_id", "0-1")))
    retain(summarize_baseline("Plain UCT + tactical rollout", lambda seed: PlainMCTSAgent(MCTSConfig(
        mcts_simulations=args.simulations, seed=seed)), args.seeds, config.get("stage_id", "0-1")))
    from agents.scripted_agent import ScriptedAgent
    retain(summarize_baseline("Scripted regression only", lambda seed: ScriptedAgent(), args.seeds, config.get("stage_id", "0-1")))


if __name__ == "__main__":
    main()

