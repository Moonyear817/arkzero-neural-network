"""Discover a solution through simulator search; never imports scripted agent."""

import argparse, json
from dataclasses import asdict
from common import ROOT
from arknights_sim.environment import ArknightsEnv
from arknights_sim.environment.replay import save_episode, replay_solution
from agents.episode import run_episode
from agents.mcts_agent import PlainMCTSAgent
from agents.mcts import MCTSConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="0-1")
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--rollout-limit", type=int, default=160)
    parser.add_argument(
        "--rollout-policy", choices=["random", "tactical"], default="tactical"
    )
    parser.add_argument("--output", default="outputs/mcts_0-1_solution.json")
    args = parser.parse_args()
    env = ArknightsEnv(trace=True)
    initial = env.reset(args.stage, seed=args.seed)
    config = MCTSConfig(
        mcts_simulations=args.simulations,
        max_depth=args.max_depth,
        rollout_limit=args.rollout_limit,
        seed=args.seed,
        rollout_policy=args.rollout_policy,
    )
    agent = PlainMCTSAgent(config)
    episode = run_episode(env, agent, initial, seed=args.seed)
    path = ROOT / args.output
    save_episode(
        path,
        env,
        initial,
        episode,
        stage_id=args.stage,
        seed=args.seed,
        metadata={
            "agent": "PlainMCTSAgent",
            "config": asdict(config),
            "strategy_source": "search-derived only; no scripted input",
        },
        search_statistics=agent.stats_history,
    )
    path.with_name(path.stem + "_search.txt").write_text(
        "\n\n".join(
            f"Decision {i + 1}\n" + s.format()
            for i, s in enumerate(agent.stats_history)
        )
        + "\n"
    )
    verification = replay_solution(path)
    path.with_name(path.stem + "_replay.json").write_text(
        json.dumps(verification, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "result": env.result(episode.final_state).to_dict(),
                "decisions": episode.decisions,
                "wall_time": episode.wall_time,
                "searches": sum(s.simulations > 0 for s in agent.stats_history),
                "simulations": sum(s.simulations for s in agent.stats_history),
                "nodes": sum(s.nodes for s in agent.stats_history),
                "replay": verification,
                "output": str(path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
