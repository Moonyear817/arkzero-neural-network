"""Measure simulator, state-copy, legal-action and complete MCTS episode costs.

Timing runs never enable tracemalloc. Python allocation peak is measured in a
separate complete episode and is not substituted for process resident memory.
"""

import argparse
import gc
import json
import platform
import statistics
import time
import tracemalloc
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from common import ROOT
from agents.mcts import MCTSConfig
from agents.mcts_agent import PlainMCTSAgent
from arknights_sim import Simulator
from arknights_sim.environment import ArknightsEnv


def mean(values):
    return statistics.mean(values) if values else 0.0


def rate(count, seconds):
    return count / seconds if seconds else 0.0


def inspect_tree(root):
    """Inspect allocated tree nodes only; rollout states are not tree nodes."""
    if root is None:
        return 0, 0, 0
    stack = [root]
    count = depth_sum = maximum = 0
    while stack:
        node = stack.pop()
        count += 1
        depth_sum += node.depth
        maximum = max(maximum, node.depth)
        stack.extend(node.children.values())
    return count, depth_sum, maximum


def state_complexity(state):
    game = state.game
    return (
        sum(unit.alive for unit in game.enemies.values())
        + sum(unit.alive for unit in game.operators.values())
        + len(game.queue.heap)
    )


def run_mcts_episode(stage, squad, args, config, *, measure=True):
    env = ArknightsEnv(
        stage, squad, trace=False, max_decisions=args.max_decisions,
        horizon=args.horizon,
    )
    state = env.reset(seed=args.seed)
    agent = PlainMCTSAgent(config)
    representative = state
    highest_complexity = state_complexity(state)
    legal_counts = []
    decision_latencies = []
    selection_latencies = []
    transition_latencies = []
    tree_nodes = tree_depth_sum = tree_maximum = 0
    started = time.perf_counter() if measure else None
    while not env.is_terminal(state):
        decision_started = time.perf_counter() if measure else None
        legal = env.legal_actions(state)
        selection_started = time.perf_counter() if measure else None
        action = agent.select_action(env, state)
        selection_finished = time.perf_counter() if measure else None
        if action not in legal:
            raise ValueError(f"MCTS selected illegal action: {action}")
        next_state = env.step(state, action)
        finished = time.perf_counter() if measure else None
        if measure:
            decision_latencies.append(finished - decision_started)
            selection_latencies.append(selection_finished - selection_started)
            transition_latencies.append(finished - selection_finished)
            nodes, depth_sum, maximum = inspect_tree(agent.last_root)
            tree_nodes += nodes
            tree_depth_sum += depth_sum
            tree_maximum = max(tree_maximum, maximum)
        legal_counts.append(len(legal))
        complexity = state_complexity(state)
        if complexity > highest_complexity:
            representative, highest_complexity = state, complexity
        state = next_state
    wall_time = time.perf_counter() - started if measure else None
    searches = [record for record in agent.stats_history if record.simulations]
    simulations = sum(record.simulations for record in searches)
    row = {
        "seed": args.seed,
        "result": env.result(state).to_dict(),
        "simulated_seconds": state.game.current_time,
        "decisions": len(legal_counts),
        "decision_legal_action_counts": legal_counts,
        "actual_searches": len(searches),
        "actual_simulations": simulations,
        "actual_nodes": sum(record.nodes for record in searches),
        "plan_reuse_decisions": sum(record.reused_discovered_plan for record in agent.stats_history),
        "selection_reasons": dict(Counter(record.selection_reason for record in agent.stats_history)),
        "rollout_steps": sum(record.rollout_steps for record in searches),
        "successful_rollouts": sum(record.successful_rollouts for record in searches),
    }
    if measure:
        if tree_nodes != row["actual_nodes"]:
            raise AssertionError("Reported MCTS node count differs from retained tree")
        row.update(
            episode_wall_seconds=wall_time,
            decision_latencies=decision_latencies,
            selection_latencies=selection_latencies,
            transition_latencies=transition_latencies,
            search_wall_seconds=sum(record.wall_time for record in searches),
            fresh_search_latencies=[record.wall_time for record in searches],
            simulation_tree_depth_sum=sum(record.average_depth * record.simulations for record in searches),
            maximum_depth=max((record.maximum_depth for record in searches), default=0),
            allocated_node_depth_sum=tree_depth_sum,
            allocated_node_maximum_depth=tree_maximum,
        )
    return row, representative


def benchmark_simulator(stage, squad, args):
    runs = []
    for _ in range(args.repeats):
        sim = Simulator(stage, squad, seed=args.seed, trace=False)
        ticks_before = sim.state.clock.tick
        events_before = sim.state.events_processed
        started = time.perf_counter()
        sim.run(args.horizon)
        elapsed = time.perf_counter() - started
        ticks = sim.state.clock.tick - ticks_before
        total = sim.state.events_processed - events_before
        runs.append({
            "wall_seconds": elapsed,
            "simulated_seconds": sim.current_time,
            "fixed_ticks": ticks,
            "queued_events": total - ticks,
            "total_events": total,
            "terminal": sim.state.done,
        })
    seconds = sum(run["wall_seconds"] for run in runs)
    return {
        "scenario": "Real stage, available squad, no deployments; advance to battle terminal or horizon",
        "event_definition": "One fixed physics/combat tick or one popped queue event, including cancelled hits",
        "events_per_second": rate(sum(run["total_events"] for run in runs), seconds),
        "simulation_seconds_per_real_second": rate(sum(run["simulated_seconds"] for run in runs), seconds),
        "runs": runs,
    }


def benchmark_state_and_legal(env, representative, args):
    # Each clone is discarded; the input is a real nonterminal state encountered
    # by MCTS, chosen by live units + queued events rather than a scripted plan.
    clone_samples = []
    legal_samples = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        for _ in range(args.clone_iterations):
            representative.game.clone()
        clone_samples.append(time.perf_counter() - started)
        started = time.perf_counter()
        for _ in range(args.legal_iterations):
            env.legal_actions(representative)
        legal_samples.append(time.perf_counter() - started)
    game = representative.game
    return {
        "state_selection": "Highest live-unit-plus-queue count among actual MCTS decision states, first timing episode",
        "time": game.current_time,
        "enemies_stored": len(game.enemies),
        "enemies_alive": sum(unit.alive for unit in game.enemies.values()),
        "operators_stored": len(game.operators),
        "operators_alive": sum(unit.alive for unit in game.operators.values()),
        "queue_length": len(game.queue.heap),
        "legal_action_count": len(env.legal_actions(representative)),
        "game_state_clones_per_second": rate(args.clone_iterations * args.repeats, sum(clone_samples)),
        "legal_actions_calls_per_second": rate(args.legal_iterations * args.repeats, sum(legal_samples)),
        "clone_iterations_per_repeat": args.clone_iterations,
        "legal_iterations_per_repeat": args.legal_iterations,
        "clone_wall_seconds": clone_samples,
        "legal_actions_wall_seconds": legal_samples,
    }


def summarize_mcts(rows):
    simulations = sum(row["actual_simulations"] for row in rows)
    searches = sum(row["actual_searches"] for row in rows)
    nodes = sum(row["actual_nodes"] for row in rows)
    search_seconds = sum(row["search_wall_seconds"] for row in rows)
    legal_counts = [count for row in rows for count in row["decision_legal_action_counts"]]
    decision_latencies = [latency for row in rows for latency in row["decision_latencies"]]
    selection_latencies = [latency for row in rows for latency in row["selection_latencies"]]
    fresh_latencies = [latency for row in rows for latency in row["fresh_search_latencies"]]
    return {
        "actual_searches": searches,
        "actual_simulations": simulations,
        "actual_nodes_including_roots": nodes,
        "decisions": sum(row["decisions"] for row in rows),
        "plan_reuse_decisions": sum(row["plan_reuse_decisions"] for row in rows),
        "successes": sum(row["result"]["success"] for row in rows),
        "episodes": len(rows),
        "search_wall_seconds": search_seconds,
        "mcts_simulations_per_second": rate(simulations, search_seconds),
        "mcts_nodes_per_second": rate(nodes, search_seconds),
        "average_nodes_per_search": rate(nodes, searches),
        "average_depth": rate(sum(row["simulation_tree_depth_sum"] for row in rows), simulations),
        "maximum_depth": max((row["maximum_depth"] for row in rows), default=0),
        "average_allocated_node_depth": rate(sum(row["allocated_node_depth_sum"] for row in rows), nodes),
        "maximum_allocated_node_depth": max((row["allocated_node_maximum_depth"] for row in rows), default=0),
        "average_legal_actions_per_decision_state": mean(legal_counts),
        "maximum_legal_actions_per_decision_state": max(legal_counts, default=0),
        "average_decision_latency_seconds": mean(decision_latencies),
        "maximum_decision_latency_seconds": max(decision_latencies, default=0),
        "average_action_selection_latency_seconds": mean(selection_latencies),
        "average_actual_search_latency_seconds": mean(fresh_latencies),
        "average_episode_wall_seconds": mean([row["episode_wall_seconds"] for row in rows]),
        "rollout_steps": sum(row["rollout_steps"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="0-1")
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=64)
    parser.add_argument("--rollout-limit", type=int, default=160)
    parser.add_argument("--max-decisions", type=int, default=512)
    parser.add_argument("--horizon", type=float, default=300)
    parser.add_argument("--rollout-policy", choices=("random", "tactical"), default="tactical")
    parser.add_argument("--no-plan-reuse", action="store_true")
    parser.add_argument("--clone-iterations", type=int, default=500)
    parser.add_argument("--legal-iterations", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=ROOT / "research/v02/benchmark.json")
    args = parser.parse_args()
    if min(args.repeats, args.clone_iterations, args.legal_iterations) <= 0:
        parser.error("Repeats and microbenchmark iteration counts must be positive")
    config = MCTSConfig(
        mcts_simulations=args.simulations, seed=args.seed,
        max_depth=args.max_depth, rollout_limit=args.rollout_limit,
        rollout_policy=args.rollout_policy, reuse_successful_plan=not args.no_plan_reuse,
    )
    # Read and normalize GameData once, outside every reported timer.
    loader_env = ArknightsEnv(trace=False, horizon=args.horizon, max_decisions=args.max_decisions)
    initial = loader_env.reset(stage_id=args.stage, seed=args.seed)
    stage = initial.game.stage
    squad = tuple(initial.game.squad.values())
    simulator = benchmark_simulator(stage, squad, args)
    rows = []
    representative = None
    for repeat in range(args.repeats):
        gc.collect()
        row, sample = run_mcts_episode(stage, squad, args, config)
        row["repeat"] = repeat + 1
        rows.append(row)
        if representative is None:
            representative = sample
    micro = benchmark_state_and_legal(loader_env, representative, args)
    summary = summarize_mcts(rows)

    # Tracing allocations substantially affects execution speed. This pass is
    # strictly separate from all latency/throughput samples above.
    gc.collect()
    tracemalloc.start()
    try:
        memory_row, memory_sample = run_mcts_episode(stage, squad, args, config, measure=False)
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    memory = {
        "method": "tracemalloc, separate one-episode pass after static GameData loading",
        "peak_traced_python_bytes": peak_bytes,
        "current_traced_python_bytes": current_bytes,
        "peak_traced_python_mib": peak_bytes / (1024 * 1024),
        "includes": "Episode state, search trees, rollouts, agent search history and Python allocations during that pass",
        "excludes": "Preloaded GameData, earlier timed samples, native allocations and full process RSS",
        "excluded_from_all_latency_and_throughput_metrics": True,
        "result": memory_row["result"],
        "actual_simulations": memory_row["actual_simulations"],
        "actual_searches": memory_row["actual_searches"],
        "decisions": memory_row["decisions"],
        "matches_first_timing_run": all(
            memory_row[key] == rows[0][key]
            for key in ("result", "actual_simulations", "actual_searches", "decisions", "decision_legal_action_counts")
        ),
    }
    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "stage": args.stage,
        "stage_data_id": stage.id,
        "squad": [{"id": unit.id, "name": unit.name} for unit in squad],
        "scenario": f"Real {args.stage} GameData, independent complete MCTS episodes; no scripted strategy",
        "config": asdict(config),
        "environment_config": {
            "horizon": args.horizon,
            "max_decisions": args.max_decisions,
            "dt": loader_env.dt,
            "windup": loader_env.windup,
        },
        "repeats": args.repeats,
        "seed": args.seed,
        "trace_enabled": False,
        "record_keys": False,
        "simulator": simulator,
        "state_and_legal_actions": micro,
        "mcts": summary,
        "memory": memory,
        "timing_runs": rows,
        "metric_definitions": {
            "average_depth": "Simulation-weighted selected/expanded tree-leaf depth; excludes all rollout steps",
            "average_allocated_node_depth": "Mean depth of all allocated MCTS nodes including depth-zero roots",
            "legal_actions": "Counts at actual episode decisions, with each repeated timing episode included",
            "average_decision_latency": "Legal enumeration, action selection/plan validation and env.step; excludes statistics bookkeeping",
            "average_actual_search_latency": "MCTSSearch.search wall time for fresh searches only; excludes reused-plan decisions",
            "mcts_throughput": "Actual simulations/nodes divided by fresh-search seconds; roots included in node count",
            "repeated_seed": "Same configured seed each repeat measures timing repeatability; this is not a success-rate study",
            "event_rate": "Core only, no deployments; separate scenario from MCTS timing",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
