"""Agent decision granularity, distinct from the simulator's physics tick."""

from dataclasses import dataclass
from math import floor
from .legal_actions import deployable_operators, skill_ready
from arknights_sim.summons.runtime import available_cards


@dataclass(frozen=True)
class DecisionEvent:
    time: float
    reasons: tuple[str, ...]

    def to_dict(self):
        return {"time": self.time, "reasons": list(self.reasons)}


def signature(game):
    # Cell membership is coarse topology, not each tiny change in position/HP.
    active = tuple(
        e for e in sorted(game.enemies.values(), key=lambda e: e.id) if e.alive
    )
    progression = game.stage_events
    from arknights_sim.core.stage_events import blockers
    return {
        "STAGE_PROGRESSION": ((progression.wave_index, progression.fragment_index,
                               progression.phase, progression.revision, blockers(game))
                              if progression else None),
        "OPERATOR_DANGER": tuple((o.id, sum(o.hp / max(o.data.hp, 1) <= threshold
                                             for threshold in (0.5, 0.25, 0.1)))
                                 for o in sorted(game.allies.values(), key=lambda o: o.id)
                                 if o.alive),
        "ENEMY_SPAWN": game.spawned,
        "ENEMY_DEATH": game.killed,
        "ENEMY_ESCAPE": game.escaped,
        "ESCORT_STATE": (game.escorts_saved, game.escorts_dead),
        "ROUTE_VISIBILITY": tuple((e.id, e.hidden) for e in active),
        "SUMMON_STATE": tuple((u.id, u.alive) for u in sorted(game.summons.values(), key=lambda u: u.id)),
        "SUMMON_DEPLOY_AVAILABLE": available_cards(game),
        "OPERATOR_DEATH": tuple(
            (o.id, o.alive, o.generation)
            for o in sorted(game.operators.values(), key=lambda o: o.id)
        ),
        "BLOCKING_CHANGED": tuple((e.id, e.blocked_by) for e in active),
        "KEY_TILE_CHANGED": tuple(
            (e.id, floor(e.position[0] + 0.5), floor(e.position[1] + 0.5))
            for e in active
        ),
        "WAYPOINT_WAIT_CHANGED": tuple(
            (e.id, e.wait_until > game.current_time + 1e-9) for e in active
        ),
        "DEPLOY_AVAILABLE": deployable_operators(game),
        "SKILL_READY": tuple(
            o.id
            for o in sorted(game.allies.values(), key=lambda o: o.id)
            if skill_ready(o, game.current_time)
        ),
        "SKILL_STATE_CHANGED": tuple(
            (o.id, o.skill.is_active(o.data.skill, game.current_time+1e-9))
            for o in sorted(game.allies.values(), key=lambda o: o.id)
            if o.alive and o.data.skill
        ),
        "REDEPLOY_READY": tuple(
            key
            for key, t in sorted(game.redeploy_at.items())
            if game.current_time + 1e-9 >= t
        ),
        "BATTLE_END": game.done,
        "DEVICE_STATE": (game.mechanic_revision,tuple((k,d.sp+1e-8>=d.data.charge,game.current_time<d.active_until) for k,d in sorted(game.devices.items()) if d.data.charge),tuple((d.id,game.dp>=d.cost and game.current_time>=game.device_ready[d.id]) for d in game.stage.devices if d.position is None)),
    }


def changed(before, after):
    return tuple(key for key in before if before[key] != after[key])
