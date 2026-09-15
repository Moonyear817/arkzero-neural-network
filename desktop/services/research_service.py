"""Application services for existing data loaders, checkpoint inference and search.

All values crossing to Qt are immutable DTOs. Checkpoint browsing reads only
sidecars; explicit loading uses the actual neural backend with weights_only=True.
"""

import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataRecord:
    category: str
    id: str
    name: str
    fields: tuple[tuple[str, str], ...]
    stage_map: object = None


@dataclass(frozen=True)
class GameDataSnapshot:
    records: tuple[DataRecord, ...]
    directory: str


def _fields(**values):
    return tuple((k.replace("_", " "), str(v)) for k, v in values.items())


def operator_summary(raw, record, skills, op):
    """Readable source facts; descriptions do not imply implemented effects."""
    wiki = record.get("wiki", {})
    def clean(text):
        return re.sub(r"<(?:/?[a-zA-Z][^>]*|@[^>]*|/)>", "", text or "")
    lines = [f"{op.name} · {op.rarity}星 · {wiki.get('profession', op.profession)} / {wiki.get('subprofession', op.subprofession)}",
             f"外文名：{wiki.get('en', '')}　日文名：{wiki.get('ja', '')}",
             f"标签：{wiki.get('tag', '')}", f"获取方式：{wiki.get('obtain_method', '')}",
             f"特性：{wiki.get('trait', clean(raw.get('description')))}", "",
             "当前模拟范围", "；".join(op.simulation_notes) or "已验证的精英0、1级基础回归干员", "",
             "游戏资料 · 各精英阶段（潜能1、信赖0）"]
    for phase, data in enumerate(raw['phases']):
        for frame in data['attributesKeyFrames']:
            a = frame['data']
            lines.append(f"精英{phase} Lv{frame['level']}：生命 {a['maxHp']} / 攻击 {a['atk']} / 防御 {a['def']} / 法抗 {a['magicResistance']:g} / 费用 {a['cost']} / 阻挡 {a['blockCnt']}")
    lines += ["", "技能资料（效果是否实现以当前模拟范围为准）"]
    for index, ref in enumerate(raw['skills'], 1):
        skill = skills.data.get(ref['skillId'])
        if not skill:
            continue
        for level, data in enumerate(skill['levels'], 1):
            bb = {v['key']: v['value'] for v in data['blackboard']}
            def value(match):
                key, spec = match.group(1), match.group(2)
                if key not in bb:
                    return match.group(0)
                number = bb[key]
                return f"{number * 100:g}%" if spec and '%' in spec else f"{number:g}"
            from arknights_sim.data.skill_description import describe_level
            description = describe_level(data)
            try:
                skills.load(ref['skillId'], level)
                description += "\n模拟支持：已实现简化效果，非官方完整验证"
            except NotImplementedError:
                description += "\n模拟支持：UNSUPPORTED（效果尚未实现）"
            sp = data['spData']
            rank = str(level) if level <= 7 else f"专精{level - 7}"
            lines.append(f"技能{index} · {data['name']} · 等级{rank}\n{description}\n初始技力 {sp['initSp']} / 消耗 {sp['spCost']} / 持续 {data['duration']:g}秒")
    lines += ["", "天赋资料（当前未模拟）"]
    for talent in raw.get('talents') or []:
        for candidate in talent.get('candidates') or []:
            if candidate.get('description'):
                cond = candidate['unlockCondition']
                lines.append(f"{candidate.get('name', '')} · 精英{cond['phase'].split('_')[-1]} Lv{cond['level']} · 潜能{candidate['requiredPotentialRank'] + 1}\n{clean(candidate['description'])}")
    lines += ["", f"来源：{record.get('source_url', '')}", f"页面版本：{record.get('revision_id', '')} · {record.get('revision_timestamp', '')}"]
    return '\n\n'.join(lines)


def load_game_data(data_dir, check=lambda: None):
    from arknights_sim.data.operator_loader import OperatorLoader
    from arknights_sim.data.skill_loader import SkillLoader
    from arknights_sim.data.stage_loader import StageLoader
    from desktop.models.simulation_state import snapshot_from_stage

    directory = Path(data_dir)
    check()
    stages = StageLoader(directory / "enemy_database.json")
    skills = SkillLoader(directory / "skill_table.json")
    operators = OperatorLoader(
        directory / "character_table.json", directory / "range_table.json", skills
    )
    records = []
    for path in sorted(directory.glob("level_*.json")):
        check()
        stage_id = "0-1" if path.stem == "level_main_00-01" else path.stem
        try:
            stage = stages.load(path)
        except NotImplementedError as exc:
            records.append(
                DataRecord(
                    "Stages",
                    stage_id,
                    stage_id,
                    _fields(Stage_ID=stage_id, Support=f"Unsupported: {exc}"),
                )
            )
            continue
        records.append(
            DataRecord(
                "Stages",
                stage_id,
                stage_id,
                _fields(
                    Stage_ID=stage_id,
                    Source_ID=stage.id,
                    Map_Size=f"{stage.map.width} × {stage.map.height}",
                    Waves=len(stage.waves),
                    Total_enemies=len(stage.spawns),
                    Routes=len(stage.routes),
                    Initial_DP=stage.initial_cost,
                    Max_DP=stage.max_cost,
                    DP_recovery=f"1 DP / {stage.cost_interval:g}s",
                    Life=stage.life,
                    Deployment_limit=stage.character_limit,
                    Mechanics="Current simulator assumptions; real-game validation incomplete",
                ),
                snapshot_from_stage(stage),
            )
        )
        for enemy in stage.enemies:
            matching = [s for s in stage.spawns if s.enemy_id == enemy.id]
            records.append(
                DataRecord(
                    "Enemies",
                    f"{stage_id}:{enemy.id}",
                    enemy.name,
                    _fields(
                        Enemy_ID=enemy.id,
                        Stage=stage_id,
                        HP=enemy.hp,
                        ATK=enemy.atk,
                        DEF=enemy.defense,
                        Move_Speed=enemy.speed,
                        Attack_Interval=f"{enemy.interval:g}s",
                        Routes=", ".join(str(s.route_index) for s in matching),
                        Spawn_Times=", ".join(f"{s.time:g}s" for s in matching),
                    ),
                )
            )
    for key in operators.available_ids():
        check()
        op = operators.load(key)
        records.append(
            DataRecord(
                "Operators",
                key,
                op.name,
                _fields(
                    Name=op.name,
                    Internal_ID=op.id,
                    Profession=operators.catalog.get(key, {}).get("wiki", {}).get("profession", op.profession),
                    Branch=operators.catalog.get(key, {}).get("wiki", {}).get("subprofession", op.subprofession),
                    Rarity=op.rarity,
                    Position=op.position_type,
                    HP=op.hp,
                    ATK=op.atk,
                    DEF=op.defense,
                    Attack_Interval=f"{op.interval:g}s",
                    Block=op.block_count,
                    DP_Cost=op.cost,
                    Skill=op.skill.id if op.skill else "None",
                    Scope="E0 level 1; potential 1; trust 0",
                    Support="；".join(op.simulation_notes) or "已验证的基础回归干员",
                    PRTS=operators.catalog.get(key, {}).get("source_url", ""),
                    Operator_details=operator_summary(operators.chars[key], operators.catalog.get(key, {}), skills, op),
                    Wiki_details=json.dumps(operators.catalog.get(key, {}).get("wiki", {}), ensure_ascii=False, indent=2),
                    Full_wiki_text=operators.catalog.get(key, {}).get("wikitext", ""),
                    GameData=json.dumps(operators.chars[key], ensure_ascii=False, indent=2),
                    All_skills=json.dumps({s["skillId"]: skills.data.get(s["skillId"]) for s in operators.chars[key]["skills"]}, ensure_ascii=False, indent=2),
                ),
            )
        )
    from arknights_sim.data.skill_description import skill_details
    owners = {}
    for key in operators.available_ids():
        for ref in operators.chars[key]["skills"]:
            owners.setdefault(ref["skillId"], []).append(operators.catalog.get(key, {}).get("name", operators.chars[key]["name"]))
    for key, names in sorted(owners.items()):
        check()
        entry = skills.data.get(key)
        supported = 0
        for rank in range(1, len(entry["levels"]) + 1 if entry else 1):
            try:
                skills.load(key, rank)
                supported += 1
            except NotImplementedError:
                pass
        first = entry["levels"][0] if entry else {}
        records.append(DataRecord("Skills", key, first.get("name", key), _fields(
            Skill_ID=key, Operators="、".join(names),
            Support=f"已实现简化效果 {supported} / {len(entry['levels']) if entry else 0} 个技能等级；非官方完整验证",
            Skill_details=skill_details(skills, key),
        )))
    from arknights_sim.data.summon_catalog import summon_records
    for summon in summon_records(operators):
        check()
        descriptions = []
        for ref in summon['token_skills'] or []:
            if ref.get('skillId'):
                descriptions.append(skill_details(skills, ref['skillId']))
            else:
                descriptions.append('此技能槽没有独立 skillId；需要核对召唤师联动，不能当作普通主动技能。')
        records.append(DataRecord('Summons', summon['owner_id'] + ':' + summon['token_id'],
            summon['owner_name'] + ' · ' + summon['token_name'], _fields(
                Owner=summon['owner_name'], Token_ID=summon['token_id'],
                Support='UNSUPPORTED_RUNTIME — 资料已收录，放置/作战/联动尚未实现',
                Summon_details=json.dumps(summon, ensure_ascii=False, indent=2) + '\n\n' + '\n\n'.join(descriptions))))
    from arknights_sim.data.map_catalog import MapCatalog
    catalog = MapCatalog(directory)
    for key, entry in catalog.records.items():
        check()
        records.append(DataRecord("Stages", key, f"{entry['code']} · {entry['name']}",
            _fields(Map_ID=key, Difficulty=entry['difficulty'], Map_Size=f"{entry['width']} × {entry['height']}",
                    Waves=entry['waves'], Routes=entry['routes'], Enemy_types=entry['enemy_types'],
                    Support="可模拟" if entry['simulation_supported'] else entry['support_reason'],
                    Source=entry['source_url'])))
    coverage=directory.parent/'mechanics/coverage.json'
    if coverage.exists():
        for rule in json.loads(coverage.read_text())['rules']:
            records.append(DataRecord('Mechanics',rule['id'],rule['name'],(('验证状态',rule['status']),('规则与范围',rule['description']),('来源',rule['source']))))
    return GameDataSnapshot(tuple({(r.category, r.id): r for r in records}.values()), str(directory))


@dataclass(frozen=True)
class CheckpointInfo:
    path: str
    name: str
    iteration: str
    created_time: float
    size: int
    metrics: tuple[tuple[str, str], ...] = ()
    best: bool = False
    latest: bool = False
    metadata_note: str = ""

    def metric(self, key):
        return dict(self.metrics).get(key, "UNKNOWN")


METRICS = (
    "evaluation_score",
    "success_rate",
    "policy_loss",
    "value_loss",
    "average_leaks",
    "average_decisions",
    "value_estimate",
    "policy_divergence",
    "inference_latency",
)


def _display_value(value):
    if value is None:
        return "UNKNOWN"
    if isinstance(value, float) and not math.isfinite(value):
        return "UNKNOWN"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return "UNKNOWN"


def list_checkpoints(checkpoint_dir, check=lambda: None):
    folder = Path(checkpoint_dir)
    if not folder.exists():
        return ()
    if not folder.is_dir():
        raise ValueError(f"Checkpoint directory is not a directory: {folder}")
    result = []
    for path in sorted(folder.rglob("*")):
        check()
        if not path.is_file() or path.suffix.lower() not in {
            ".pt",
            ".pth",
            ".ckpt",
            ".safetensors",
            ".npz",
        }:
            continue
        metadata, note = {}, ""
        candidates = tuple(
            dict.fromkeys(
                (
                    path.with_suffix(path.suffix + ".json"),
                    path.with_suffix(".json"),
                    path.with_suffix(path.suffix + ".yaml"),
                    path.with_suffix(".yaml"),
                )
            )
        )
        for sidecar in candidates:
            if sidecar.is_file():
                try:
                    if sidecar.stat().st_size > 2_000_000:
                        raise ValueError("metadata exceeds 2 MB")
                    raw = sidecar.read_text(encoding="utf-8")
                    parsed = (
                        json.loads(raw)
                        if sidecar.suffix == ".json"
                        else yaml.safe_load(raw)
                    )
                    if not isinstance(parsed, dict):
                        raise TypeError("metadata must be a mapping")
                    metadata = parsed
                except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
                    note = f"Metadata unavailable: {exc}"
                break
        st = path.stat()
        metrics = metadata.get("evaluation", {})
        metrics = dict(metrics) if isinstance(metrics, dict) else {}
        if isinstance(metadata.get("metrics"), dict):
            metrics.update(metadata["metrics"])
        metrics.update({k: metadata[k] for k in METRICS if k in metadata})
        metrics.setdefault("evaluation_score", metrics.get("average_return"))
        metrics.setdefault("value_estimate", metrics.get("mean_value"))
        result.append(
            CheckpointInfo(
                str(path.resolve()),
                path.name,
                _display_value(metadata.get("iteration")),
                getattr(st, "st_birthtime", st.st_mtime),
                st.st_size,
                tuple((k, _display_value(metrics.get(k))) for k in METRICS),
                metadata.get("best") is True
                or metadata.get("is_best") is True
                or path.name == "best.pt",
                path.name == "latest.pt",
                note,
            )
        )
    if result:
        latest_path = max(
            result, key=lambda m: (Path(m.path).stat().st_mtime_ns, m.path)
        ).path
        result = [replace(m, latest=m.latest or m.path == latest_path) for m in result]
    return tuple(result)


@dataclass(frozen=True)
class PolicyRow:
    action: str
    neural_probability: float | None
    mcts_probability: float
    visits: int
    q_value: float


@dataclass(frozen=True)
class TreeNodeSnapshot:
    id: int
    parent_id: int | None
    action: str
    visits: int
    q_value: float
    terminal: bool
    children: tuple[int, ...]


@dataclass(frozen=True)
class MCTSSnapshot:
    stage_id: str
    game_time: float
    seed: int
    simulations: int
    selected_action: str
    wall_time: float
    node_count: int
    terminal: bool
    rows: tuple[PolicyRow, ...]
    tree: tuple[TreeNodeSnapshot, ...]
    network_value: float | None = None
    algorithm: str = "Plain single-agent UCT · tactical rollout"


class _CancellableEnvironment:
    def __init__(self, env, check):
        self.env, self.check = env, check

    def __getattr__(self, name):
        attribute = getattr(self.env, name)
        if not callable(attribute):
            return attribute

        def guarded(*args, **kwargs):
            self.check()
            return attribute(*args, **kwargs)

        return guarded


def search_mcts(
    data_dir, stage_id="0-1", seed=12345, simulations=16, check=lambda: None
):
    from agents.mcts.search import MCTSConfig, MCTSSearch
    from arknights_sim.environment import ArknightsEnv
    from arknights_sim.environment.action import ActionType

    check()
    env = _CancellableEnvironment(ArknightsEnv(data_dir=data_dir), check)
    state = env.reset(stage_id=stage_id, seed=seed)
    while not env.is_terminal(state) and not any(
        a.type == ActionType.DEPLOY for a in env.legal_actions(state)
    ):
        state = env.advance_to_next_decision_event(state)
    check()
    result = MCTSSearch(MCTSConfig(mcts_simulations=simulations, seed=seed)).search(
        env, state
    )
    check()
    stats = result.stats
    total = sum(a.visit_count for a in stats.actions)
    rows = tuple(
        PolicyRow(
            str(a.action),
            None,
            a.visit_count / total if total else 0.0,
            a.visit_count,
            a.mean_value,
        )
        for a in stats.actions
    )
    # Send only a bounded tree made of scalar DTOs. The GUI renders one level at
    # a time; omitted low-visit children remain in the full root policy table.
    nodes = []
    queue = [(result.root, None)]
    while queue and len(nodes) < 2000:
        check()
        node, parent_id = queue[len(nodes)]
        node_id = len(nodes)
        children = sorted(
            node.children.values(), key=lambda n: (-n.visit_count, -n.mean_value)
        )[:20]
        remaining = 2000 - len(queue)
        child_ids = tuple(
            range(len(queue), len(queue) + min(len(children), max(0, remaining)))
        )
        queue.extend((child, node_id) for child in children[: len(child_ids)])
        nodes.append(
            TreeNodeSnapshot(
                node_id,
                parent_id,
                str(node.action) if node.action else "ROOT",
                node.visit_count,
                node.mean_value,
                node.terminal,
                child_ids,
            )
        )
        if len(nodes) >= len(queue):
            break
    return MCTSSnapshot(
        stage_id,
        state.game.current_time,
        seed,
        stats.simulations,
        str(result.action),
        stats.wall_time,
        stats.nodes,
        env.is_terminal(state),
        rows,
        tuple(nodes),
    )


@dataclass(frozen=True)
class LoadedModelSnapshot:
    path: str
    name: str
    iteration: str
    parameter_count: int
    device: str
    value: float


@dataclass(frozen=True)
class EvaluationSnapshot:
    path: str
    seeds: tuple[int, ...]
    simulations: int
    metrics: tuple[tuple[str, str], ...]
    episodes: tuple[tuple[tuple[str, str], ...], ...]
    saved_path: str = ""


def load_model(path, check=lambda: None):
    """Load only the selected checkpoint in a worker; never execute pickle code."""
    import torch

    from network import PolicyValueNetwork

    check()
    path = Path(path)
    payload = torch.load(path, weights_only=True, map_location="cpu")
    check()
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError("Checkpoint does not contain model_state_dict")
    # Meta construction avoids consuming process-wide RNG while training runs
    # in a different worker. Loaded tensors supply every actual parameter.
    with torch.device("meta"):
        model = PolicyValueNetwork(future_events_enabled=payload.get(
            "network_metadata", {}).get("future_events_enabled", True))
    model.load_state_dict(payload["model_state_dict"], strict=True, assign=True)
    model.reward_version = payload.get('reward_version', 1)
    model.eval()
    iteration = _display_value(payload.get("iteration"))
    del payload
    check()
    return model, iteration


def inspect_model(path, data_dir, check=lambda: None):
    from arknights_sim.environment import ArknightsEnv
    from network import NeuralEvaluator

    model, iteration = load_model(path, check)
    env = _CancellableEnvironment(ArknightsEnv(data_dir=data_dir), check)
    state = env.reset()
    _, value = NeuralEvaluator(model, "cpu").evaluate(
        env, state, env.legal_actions(state)
    )
    check()
    return LoadedModelSnapshot(
        str(Path(path).resolve()),
        Path(path).name,
        iteration,
        model.parameter_count,
        "CPU",
        value,
    )


def evaluate_model(
    path, data_dir, simulations=16, seeds=(80000,), check=lambda: None, output_dir=None
):
    from time import perf_counter

    from arknights_sim.environment import ArknightsEnv
    from network import NeuralEvaluator
    from training.config import default_config
    from training.self_play import play_episode

    model, _ = load_model(path, check)
    evaluator = NeuralEvaluator(model, "cpu")
    config = default_config()
    config.update(
        mcts_simulations=simulations,
        temperature=0.0,
        dirichlet_epsilon=0.0,
        prior_uniform_mix=0.05,
    )
    rows = []
    started = perf_counter()
    for seed in seeds:
        check()
        env = _CancellableEnvironment(ArknightsEnv(data_dir=data_dir), check)
        episode = play_episode(
            env, evaluator, config, seed=seed, stage="0-1", training=False
        )
        rows.append(episode.metrics)
    if not rows:
        raise ValueError("Evaluation needs at least one seed")
    metrics = {
        "success_rate": sum(r["success"] for r in rows) / len(rows),
        "average_leaks": sum(r["leaks"] for r in rows) / len(rows),
        "average_kills": sum(r["kills"] for r in rows) / len(rows),
        "average_decisions": sum(r["decisions"] for r in rows) / len(rows),
        "mean_value": sum(r["mean_value"] for r in rows) / len(rows),
        "average_decision_time": sum(r["average_decision_time"] for r in rows)
        / len(rows),
        "wall_time": perf_counter() - started,
        "stage": "0-1",
        "device": "CPU",
        "noise": False,
        "temperature": 0.0,
    }
    saved = ""
    if output_dir is not None:
        import time

        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{Path(path).stem}_{time.time_ns()}.json"
        target.write_text(
            json.dumps(
                {
                    "checkpoint": str(path),
                    "seeds": list(seeds),
                    "simulations": simulations,
                    "metrics": metrics,
                    "episodes": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        saved = str(target)
    return EvaluationSnapshot(
        str(Path(path).resolve()),
        tuple(seeds),
        simulations,
        tuple((k, _display_value(v)) for k, v in metrics.items()),
        tuple(
            tuple((k, _display_value(v)) for k, v in row.items() if k != "history")
            for row in rows
        ),
        saved,
    )


def search_neural_mcts(
    data_dir, checkpoint, stage_id="0-1", seed=12345, simulations=16, check=lambda: None
):
    from arknights_sim.environment import ArknightsEnv
    from arknights_sim.environment.action import ActionType
    from mcts import PUCTConfig, PUCTSearch
    from network import NeuralEvaluator

    model, _ = load_model(checkpoint, check)
    env = _CancellableEnvironment(ArknightsEnv(data_dir=data_dir), check)
    state = env.reset(stage_id=stage_id, seed=seed)
    while not env.is_terminal(state) and not any(
        a.type == ActionType.DEPLOY for a in env.legal_actions(state)
    ):
        state = env.advance_to_next_decision_event(state)
    result = PUCTSearch(
        NeuralEvaluator(model, "cpu"),
        PUCTConfig(
            simulations=simulations,
            seed=seed,
            prior_uniform_mix=0.05,
            temperature=0.0,
            dirichlet_epsilon=0.0,
        ),
    ).search(env, state, training=False)
    rows = tuple(
        PolicyRow(
            str(a.action), a.neural_prior, a.visit_fraction, a.visit_count, a.mean_value
        )
        for a in result.stats.actions
    )
    nodes = []
    queue = [(result.root, None, "ROOT", result.root.N, result.root.Q)]
    while len(nodes) < len(queue) and len(nodes) < 2000:
        check()
        node, parent_id, label, visits, q_value = queue[len(nodes)]
        node_id = len(nodes)
        children = (
            sorted(node.edges.values(), key=lambda edge: (-edge.N, -edge.Q))[:20]
            if node
            else []
        )
        remaining = max(0, 2000 - len(queue))
        children = children[:remaining]
        child_ids = tuple(range(len(queue), len(queue) + len(children)))
        queue.extend(
            (edge.child, node_id, str(edge.action), edge.N, edge.Q) for edge in children
        )
        nodes.append(
            TreeNodeSnapshot(
                node_id,
                parent_id,
                label,
                visits,
                q_value,
                node.terminal if node else False,
                child_ids,
            )
        )
    return MCTSSnapshot(
        stage_id,
        state.game.current_time,
        seed,
        result.stats.simulations,
        str(result.selected_action),
        result.stats.wall_time,
        result.stats.nodes,
        env.is_terminal(state),
        rows,
        tuple(nodes),
        result.root_value,
        f"Neural PUCT · {Path(checkpoint).name} · evaluation, no root noise",
    )
