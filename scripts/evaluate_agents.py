"""Reproducible multi-seed agent evaluation; every attempted trial is retained."""

import argparse
import json
import platform
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT
from agents.mcts import MCTSConfig
from agents.mcts_agent import PlainMCTSAgent
from agents.random_agent import RandomAgent
from arknights_sim.environment import ArknightsEnv


def _run_trial(agent, args, seed, *, timeout=None):
    """Strategy-free driver; traces and per-transition key recording are off.

    Scripted regression can use its explicit scheduling hook. Search receives
    only the environment and its own state and cannot receive baseline history.
    A wall-time cutoff is checked between decisions, so one search can overshoot
    it; the actual measured time is always retained in the row.
    """
    env = ArknightsEnv(trace=False, max_decisions=args.max_decisions)
    state = env.reset(stage_id=args.stage, seed=seed)
    agent.reset()
    started = time.perf_counter()
    legal_counts = []
    status = "completed"
    error = None
    try:
        while not env.is_terminal(state):
            if timeout and time.perf_counter() - started >= timeout:
                status = "timeout"
                break
            if hasattr(agent, "prepare_state"):
                state = agent.prepare_state(env, state)
            if env.is_terminal(state):
                break
            legal = env.legal_actions(state)
            action = agent.select_action(env, state)
            if action not in legal:
                raise ValueError(f"Agent selected illegal action: {action}")
            state = env.step(state, action)
            legal_counts.append(len(legal))
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"
    wall_time = time.perf_counter() - started
    result = env.result(state).to_dict()
    stats = list(getattr(agent, "stats_history", ()))
    fresh = [item for item in stats if item.simulations]
    simulations = sum(item.simulations for item in stats)
    nodes = sum(item.nodes for item in stats)
    root_searches = []
    for decision, item in enumerate(stats):
        if not item.simulations:
            continue
        record = item.to_dict()
        record.pop("actions")
        record["decision"] = decision
        root_searches.append(record)
    return {
        "agent": type(agent).__name__,
        "stage": args.stage,
        "seed": seed,
        "status": status,
        "error": error,
        **result,
        "success": status == "completed" and result["success"],
        "decisions": len(legal_counts),
        "wall_time": wall_time,
        "simulation_time": state.game.current_time,
        "average_legal_actions": statistics.mean(legal_counts) if legal_counts else 0,
        "maximum_legal_actions": max(legal_counts, default=0),
        "decision_legal_action_counts": legal_counts,
        "simulations": simulations,
        "nodes": nodes,
        "fresh_searches": len(fresh),
        "reused_steps": sum(item.reused_discovered_plan for item in stats),
        "average_depth": (
            sum(item.average_depth * item.simulations for item in fresh) / simulations
            if simulations else 0
        ),
        "maximum_depth": max((item.maximum_depth for item in fresh), default=0),
        "rollout_steps": sum(item.rollout_steps for item in fresh),
        "successful_rollouts": sum(item.successful_rollouts for item in fresh),
        "search_wall_time": sum(item.wall_time for item in fresh),
        "average_decision_latency": wall_time / max(1, len(legal_counts)),
        "average_fresh_search_latency": (
            statistics.mean(item.wall_time for item in fresh) if fresh else 0
        ),
        "selection_reasons": dict(Counter(item.selection_reason for item in stats)),
        "root_searches": root_searches,
        "final_state_key": env.state_key(state),
        "trace_enabled": False,
        "record_keys": False,
        "timeout_seconds": timeout,
    }


def run_scripted_baseline(args, seed):
    # This import is intentionally local to the historical regression baseline.
    # No schedule, decision history, or baseline result is passed into MCTS.
    from agents.scripted_agent import ScriptedAgent

    return _run_trial(ScriptedAgent(), args, seed)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["agent"], row["rollout_policy"], row["budget"])].append(row)
    result = []
    for (agent, policy, budget), records in groups.items():
        completed = [row for row in records if row["status"] == "completed"]
        entry = {
            "agent": agent, "rollout_policy": policy, "budget": budget,
            "trials": len(records), "seeds": [row["seed"] for row in records],
            "completed": len(completed),
            "timeouts": sum(row["status"] == "timeout" for row in records),
            "errors": sum(row["status"] == "error" for row in records),
            "successes": sum(row["success"] for row in records),
            "success_rate": sum(row["success"] for row in records) / len(records),
        }
        for name in (
            "kills", "leaks", "decisions", "wall_time", "average_legal_actions",
            "simulations", "nodes", "fresh_searches", "reused_steps",
            "average_depth", "average_decision_latency", "rollout_steps",
        ):
            label = name if name.startswith("average_") else "average_" + name
            entry[label] = statistics.mean(row[name] for row in records)
        entry["maximum_legal_actions"] = max(row["maximum_legal_actions"] for row in records)
        entry["maximum_depth"] = max(row["maximum_depth"] for row in records)
        entry["wall_time_min"] = min(row["wall_time"] for row in records)
        entry["wall_time_max"] = max(row["wall_time"] for row in records)
        result.append(entry)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="0-1")
    parser.add_argument("--budgets", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--seeds", type=int, nargs="+", default=[12345, 12346, 12347])
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--rollout-limit", type=int, default=160)
    parser.add_argument("--max-decisions", type=int, default=512)
    parser.add_argument("--uniform-timeout", type=float, default=60.0)
    parser.add_argument("--trial-timeout", type=float, default=None)
    parser.add_argument("--skip-uniform", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--append", action="store_true",
                        help="Retain existing JSON trials when extending a sweep")
    parser.add_argument("--output", type=Path, default=ROOT / "research/v02/agent_evaluation.json")
    args = parser.parse_args()
    if (any(budget < 1 for budget in args.budgets) or args.uniform_timeout <= 0
            or (args.trial_timeout is not None and args.trial_timeout <= 0)):
        parser.error("Budgets and uniform timeout must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw_path = args.output.with_suffix(".jsonl")
    run_id = datetime.now(timezone.utc).isoformat()
    rows = (
        json.loads(args.output.read_text())["trials"]
        if args.append and args.output.exists() else []
    )

    def retain(row, policy=None, budget=None):
        row.update(run_id=run_id, rollout_policy=policy, budget=budget)
        rows.append(row)
        with raw_path.open("a") as raw:
            raw.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            raw.flush()
        document = {
            "schema_version": 1,
            "run_id": run_id,
            "run_ids": list(dict.fromkeys(row["run_id"] for row in rows)),
            "stage": args.stage,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "settings": {
                "budgets": list(dict.fromkeys(
                    [row["budget"] for row in rows if row["rollout_policy"] == "tactical"]
                    + args.budgets
                )), "seeds": args.seeds,
                "max_depth": args.max_depth, "rollout_limit": args.rollout_limit,
                "max_decisions": args.max_decisions, "trace": False,
                "record_keys": False, "reuse_successful_plan": True,
                "uniform_timeout": args.uniform_timeout,
                "trial_timeout": args.trial_timeout,
            },
            "notes": [
                "Success means terminal 11/11 kills and zero leaks for this stage.",
                "All attempted trials, including failures and timeouts, are retained.",
                "Timeouts count as unsuccessful in success_rate; partial kills/leaks are labeled by status.",
                "Tactical rollout uses only generic routes/ranges and retains every legal action.",
                "A verified search-derived plan can be reused; reused decisions run zero additional simulations.",
                "Average legal actions refer to actual episode decisions; tree depth excludes rollout steps.",
                "Timing excludes initial data loading; raw JSONL appends every run with run_id.",
            ],
            "summary": summarize(rows),
            "trials": rows,
        }
        args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({key: row[key] for key in (
            "agent", "rollout_policy", "budget", "seed", "status", "success",
            "kills", "leaks", "decisions", "wall_time", "simulations", "nodes",
        )}), flush=True)

    # Each run constructs an independent environment and agent. Baseline
    # results are serialized only; they are not inputs to search configuration.
    if not args.skip_baselines:
        for seed in args.seeds:
            retain(_run_trial(RandomAgent(seed), args, seed))
        for seed in args.seeds:
            retain(run_scripted_baseline(args, seed))
    for budget in args.budgets:
        for seed in args.seeds:
            agent = PlainMCTSAgent(MCTSConfig(
                mcts_simulations=budget, exploration_constant=2 ** 0.5,
                max_depth=args.max_depth, rollout_limit=args.rollout_limit,
                seed=seed, rollout_policy="tactical",
            ))
            retain(_run_trial(agent, args, seed, timeout=args.trial_timeout), "tactical", budget)
    if not args.skip_uniform:
        for seed in args.seeds:
            agent = PlainMCTSAgent(MCTSConfig(
                mcts_simulations=16, max_depth=args.max_depth,
                rollout_limit=args.rollout_limit, seed=seed, rollout_policy="random",
            ))
            retain(_run_trial(agent, args, seed, timeout=args.uniform_timeout), "random", 16)
    print(f"Saved {len(rows)} trials to {args.output}", flush=True)


if __name__ == "__main__":
    main()
