"""Audit recorded evaluation actions in fresh simulator instances, without search.

This checks stored action times, legality, action counts and final outcomes.
Original evaluation files contain no battle traces or intermediate state hashes;
this audit therefore cannot claim equality of those unrecorded observations.
"""

import argparse
import hashlib
import json
import traceback
from time import perf_counter

from common import ROOT

from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader
from arknights_sim.environment import Action, ArknightsEnv


def replay_episode(record, config):
    """Construct and load a new environment for this one recorded episode."""
    env = ArknightsEnv(
        data_dir=config.get("data_dir"),
        horizon=config["horizon"],
        max_decisions=config["max_decisions"],
        trace=False,
    )
    loader = OperatorLoader(
        env.data_dir / "character_table.json",
        env.data_dir / "range_table.json",
        SkillLoader(env.data_dir / "skill_table.json"),
    )
    env.squad = tuple(loader.load(key) for key in config["squad"])
    state = env.reset(stage_id=record["stage"], seed=record["seed"])
    history = record["history"]
    for index, decision in enumerate(history):
        if env.is_terminal(state):
            raise ValueError(
                f"Decision {index}: history continues after terminal state"
            )
        if state.game.current_time != decision["time"]:
            raise ValueError(
                f"Decision {index}: time {state.game.current_time!r} != recorded {decision['time']!r}"
            )
        legal = env.legal_actions(state)
        if len(legal) != decision["legal_actions"]:
            raise ValueError(
                f"Decision {index}: legal action count {len(legal)} != recorded {decision['legal_actions']}"
            )
        action = Action.from_dict(decision["action"])
        if action not in legal:
            raise ValueError(f"Decision {index}: recorded action is illegal: {action}")
        state = env.step(state, action)
    result = {
        **env.result(state).to_dict(),
        "decisions": len(history),
        "game_time": state.game.current_time,
    }
    fields = (
        "success",
        "kills",
        "leaks",
        "life",
        "total_enemies",
        "terminal",
        "termination",
        "simulator_result",
        "decisions",
        "game_time",
    )
    differences = {
        field: {"recorded": record[field], "replayed": result[field]}
        for field in fields
        if record[field] != result[field]
    }
    if differences:
        raise ValueError(f"Final outcome differs: {differences}")
    if not env.is_terminal(state):
        raise ValueError("History ends before a terminal environment state")
    return {
        "matched": True,
        "seed": record["seed"],
        "stage": record["stage"],
        "episode": record["episode"],
        "decisions_checked": len(history),
        "action_times_exact": True,
        "all_actions_legal": True,
        "legal_action_counts_exact": True,
        "recorded_result_exact": True,
        "result": result,
        "replayed_final_state_key": env.state_key(state),
        "original_final_state_key_available": False,
        "original_trace_available": False,
        "action_history_sha256": hashlib.sha256(
            json.dumps(history, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluations", default="outputs/alphazero/smoke/evaluations.jsonl"
    )
    parser.add_argument(
        "--checkpoint", help="Read configuration only; never instantiate a network"
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    source = ROOT / args.evaluations
    checkpoint = (
        ROOT / args.checkpoint
        if args.checkpoint
        else source.parent / "checkpoints/latest.pt"
    )
    output = (
        ROOT / args.output
        if args.output
        else source.parent / "evaluation_replay_audit.json"
    )
    # Configuration is stored with checkpoints. Loading tensors here only reads
    # the file; no network/evaluator/search/trainer modules are imported.
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = dict(payload["config"])
    del payload
    source_bytes = source.read_bytes()
    evaluations = [
        json.loads(line) for line in source_bytes.splitlines() if line.strip()
    ]
    started = perf_counter()
    audited = []
    aggregates = []
    for evaluation in evaluations:
        group = []
        for episode in evaluation["episode_results"]:
            try:
                audit = replay_episode(episode, config)
            except Exception:  # noqa: BLE001 -- persist every failed episode audit
                audit = {
                    "matched": False,
                    "seed": episode.get("seed"),
                    "episode": episode.get("episode"),
                    "error": traceback.format_exc(),
                }
            audit.update(
                evaluation_iteration=evaluation["iteration"],
                evaluation_kind=evaluation["kind"],
            )
            audited.append(audit)
            group.append(audit)
        matched = all(row["matched"] for row in group)
        success_rate = (
            sum(row.get("result", {}).get("success", False) for row in group)
            / len(group)
            if group
            else 0.0
        )
        aggregates.append(
            {
                "iteration": evaluation["iteration"],
                "kind": evaluation["kind"],
                "episodes": len(group),
                "all_episodes_matched": matched,
                "recorded_success_rate": evaluation["success_rate"],
                "replayed_success_rate": success_rate if matched else None,
                "success_rate_exact": matched
                and success_rate == evaluation["success_rate"],
            }
        )
    report = {
        "status": "passed"
        if all(row["matched"] for row in audited)
        and all(row["success_rate_exact"] for row in aggregates)
        else "failed",
        "source": str(source),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "configuration_source": str(checkpoint),
        "scope": "Exact recorded action times, legality, legal counts, final outcomes and aggregate success rates; fresh simulator per episode; no search",
        "limitations": "Original traces and intermediate/final state hashes were not recorded, so trace/hash equality with the original run is not asserted",
        "environment_config": {
            key: config.get(key)
            for key in (
                "stage",
                "squad",
                "seed",
                "horizon",
                "max_decisions",
                "data_dir",
            )
        },
        "episodes_checked": len(audited),
        "episodes_matched": sum(row["matched"] for row in audited),
        "wall_seconds": perf_counter() - started,
        "evaluation_groups": aggregates,
        "episodes": audited,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "episodes_checked",
                    "episodes_matched",
                    "wall_seconds",
                )
            },
            indent=2,
        )
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
