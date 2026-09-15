"""Solution artifacts and strict fresh-instance replay, without agent dependencies."""

import hashlib, json, math
from dataclasses import replace
from pathlib import Path
from .action import Action
from .env import ArknightsEnv
from arknights_sim.data.operator_loader import OperatorLoader
from arknights_sim.data.skill_loader import SkillLoader


def trace_hash(trace):
    return hashlib.sha256(
        json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def save_episode(
    path,
    env,
    initial_state,
    episode,
    *,
    stage_id="0-1",
    seed=12345,
    metadata=None,
    search_statistics=(),
):
    final = episode.final_state
    document = {
        "schema_version": 2,
        "stage": stage_id,
        "seed": seed,
        "squad": list(initial_state.game.squad),
        "operator_progression": {key: {field: getattr(op, field) for field in
            ('elite', 'level', 'potential', 'skill_index', 'skill_level')}
            for key, op in initial_state.game.squad.items()},
        "operator_notes": {key: list(op.simulation_notes) for key, op in initial_state.game.squad.items()},
        "environment": {
            "dt": initial_state.game.clock.dt,
            "windup": initial_state.game.windup,
            "horizon": initial_state.horizon,
            "max_decisions": initial_state.max_decisions,
        },
        "initial_state_key": env.state_key(initial_state),
        "history": [row.to_dict() for row in episode.history],
        "result": env.result(final).to_dict(),
        "final_state_key": env.state_key(final),
        "final_game_hash": final.game.stable_hash(),
        "trace_hash": trace_hash(final.trace) if final.trace_enabled else None,
        "trace": list(final.trace) if final.trace_enabled else None,
        "decisions": episode.decisions,
        "wall_time": episode.wall_time,
        "metadata": metadata or {},
        "search_statistics": [s.to_dict() for s in search_statistics],
        "mechanics_status": {
            "continuous_turning": "ASSUMED",
            "blocking_radius": "ASSUMED",
            "attack_windup": "ASSUMED",
            "same_frame_event_priority": "ASSUMED",
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    lines = [
        f"Stage: {stage_id}",
        f"Seed: {seed}",
        f"Metadata: {json.dumps(document['metadata'], ensure_ascii=False)}",
        "",
    ]
    for row in episode.history:
        label = str(row.action)
        for key, data in initial_state.game.squad.items():
            label = label.replace(key, data.name)
        lines.append(f"{row.time:09.3f} {label}")
    result = document["result"]
    lines.extend(
        [
            "",
            f"RESULT: {'SUCCESS' if result['success'] else 'FAILURE'}",
            f"Kills: {result['kills']} / {result['total_enemies']}",
            f"Leaks: {result['leaks']}",
            f"Life: {result['life']}",
            f"Decisions: {episode.decisions}",
            f"Wall time: {episode.wall_time:.6f}s",
            f"MCTS simulations: {sum(s.simulations for s in search_statistics)}",
            f"MCTS nodes: {sum(s.nodes for s in search_statistics)}",
            "Mechanics: four uncalibrated rules remain ASSUMED.",
        ]
    )
    path.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    return document


def replay_solution(path, *, env_factory=None):
    document = json.loads(Path(path).read_text())
    if document.get("schema_version") != 2:
        raise ValueError("Unknown replay format")
    if document["decisions"] != len(document["history"]):
        raise ValueError("Decision count mismatch")
    if (
        document["trace_hash"] is not None
        and trace_hash(document["trace"]) != document["trace_hash"]
    ):
        raise ValueError("Stored battle trace mismatch")
    env = (
        env_factory(document)
        if env_factory
        else ArknightsEnv(
            **document["environment"], trace=document["trace_hash"] is not None
        )
    )
    if env_factory:
        initial = env.reset(stage_id=document["stage"], seed=document["seed"])
    else:
        data = env.data_dir
        loader = OperatorLoader(
            data / "character_table.json",
            data / "range_table.json",
            SkillLoader(data / "skill_table.json"),
        )
        squad = [loader.load(key, **document.get("operator_progression", {}).get(key, {})) for key in document["squad"]]
        squad = [replace(op, simulation_notes=tuple(document.get('operator_notes', {}).get(op.id, op.simulation_notes))) for op in squad]
        initial = env.reset(
            stage_id=document["stage"], squad=squad, seed=document["seed"]
        )
    if env.state_key(initial) != document["initial_state_key"]:
        raise ValueError("Initial setup/key mismatch")
    state = initial
    for index, row in enumerate(document["history"]):
        if row["decision"] != index:
            raise ValueError(f"Decision index mismatch at decision {index}")
        if "external_advance_to" in row:
            state = env.advance_to_time(state, row["external_advance_to"])
        if (
            not math.isfinite(row["time"])
            or abs(state.game.current_time - row["time"]) > 1e-8
        ):
            raise ValueError(f"Replay time mismatch at decision {index}")
        action = Action.from_dict(row["action"])
        legal = env.legal_actions(state)
        if row["legal_action_count"] != len(legal):
            raise ValueError(f"Legal action count mismatch at decision {index}")
        if action not in legal:
            raise ValueError(f"Illegal replay action at decision {index}")
        state = env.step(state, action)
        if row["next_state_key"] and env.state_key(state) != row["next_state_key"]:
            raise ValueError(f"Replay state mismatch at decision {index}")
        if (
            "next_decision_event" in row
            and state.last_decision.to_dict() != row["next_decision_event"]
        ):
            raise ValueError(f"Decision event mismatch at decision {index}")
    if (
        env.result(state).to_dict() != document["result"]
        or env.state_key(state) != document["final_state_key"]
        or state.game.stable_hash() != document["final_game_hash"]
    ):
        raise ValueError("Final replay mismatch")
    if (
        document["trace_hash"] is not None
        and trace_hash(state.trace) != document["trace_hash"]
    ):
        raise ValueError("Battle trace mismatch")
    return {
        "identical": True,
        "state_key": env.state_key(state),
        "game_hash": state.game.stable_hash(),
        "trace_identical": document["trace_hash"] is not None,
        "result": env.result(state).to_dict(),
        "decisions": state.decision_count,
    }
