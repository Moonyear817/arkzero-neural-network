"""Hybrid fixed-grid simulation with an exact-time queue for spawn, hit and skill events.

RULE-004 collision radius; RULE-005 target ties; RULE-006 simplified windup.
"""

import copy, math
from functools import lru_cache
from .event_priority import SIMULATION_PHASES
from .state import GameState
from . import stage_events
from arknights_sim.entities.unit import Unit
from arknights_sim.map.route import compile_route, WAIT_KINDS
from arknights_sim.combat.blocking import release, try_block
from arknights_sim.combat.targeting import OperatorTargetSelector, EnemyTargetSelector
from arknights_sim.combat.damage import DamageCalculator
from arknights_sim.combat.attack import AttackEvent
from arknights_sim.skills.modifier import Modifier, modified
from arknights_sim.skills.skill import SkillState
from arknights_sim.deployment.cost import deployment_cost, retreat_refund
from arknights_sim.debug.trace import BattleTrace
from arknights_sim.mechanics import runtime as mechanics
from arknights_sim.mechanics import terrain
from arknights_sim.summons import runtime as summons
from arknights_sim.skills.runtime import apply_timed_skill, regenerate
from arknights_sim.talents import runtime as talents


@lru_cache(maxsize=32)
def _compiled_paths(stage):
    """Profile-guided cache: StageData and all returned waypoints are frozen."""
    return tuple(
        (i, compile_route(stage.map, stage.routes[i]))
        for i in sorted({spawn.route_index for spawn in stage.spawns})
    )


class Simulator:
    event_phases = SIMULATION_PHASES

    def __init__(self, stage, squad=(), seed=12345, dt=1 / 60, trace=False, windup=0.2):
        if not math.isfinite(dt) or dt <= 0 or not math.isfinite(windup) or windup < 0:
            raise ValueError("Timing configuration")
        if len({o.id for o in squad}) != len(squad):
            raise ValueError("Duplicate operator")
        self.state = GameState(stage, squad, seed, dt)
        self.state.windup = windup
        self.trace = BattleTrace(trace)
        self.paths = dict(_compiled_paths(stage))
        if stage.devices:self.paths=dict(enumerate(mechanics.paths_for(self.state)))
        if self.state.stage_events is None:
            for s in stage.spawns:
                self.state.queue.push(s.time, "SPAWN", s)
        self.trace.log(0, "BATTLE_START", stage=stage.id, seed=seed)
        for device in self.state.devices.values():
            if device.data.kind in ('friendly_frost','friendly_altar'):
                self.state.queue.push(device.data.charge,'DEVICE_PULSE',device.id)

    @classmethod
    def from_state(cls, state, trace=False):
        sim = cls.__new__(cls)
        sim.state = state
        sim.trace = BattleTrace(trace)
        sim.paths = dict(_compiled_paths(state.stage))
        if state.stage.devices:sim.paths=dict(enumerate(mechanics.paths_for(state)))
        return sim

    @property
    def current_time(self):
        return self.state.current_time

    def clone(self):
        return copy.deepcopy(self)

    def snapshot(self):
        return self.state.snapshot()

    def stable_hash(self):
        return self.state.stable_hash()

    def _log(self, kind, **kw):
        self.trace.log(self.current_time, kind, **kw)

    def deploy(self, key, position, direction=0):
        s = self.state
        if s.done:
            raise ValueError("Battle ended")
        if key not in s.squad:
            raise ValueError("Operator not in squad")
        if direction not in range(4) or any(int(v) != v for v in position):
            raise ValueError("Position/direction")
        if key in s.operators and s.operators[key].alive:
            raise ValueError("Already deployed")
        if self.current_time + 1e-9 < s.redeploy_at.get(key, 0):
            raise ValueError("Redeploy cooldown")
        active = [o for o in s.allies.values() if o.alive]
        if s.occupied_deployment_slots >= s.stage.character_limit or any(
            o.position == tuple(position) for o in active
        ):
            raise ValueError("Deployment limit/occupied tile")
        d = s.squad[key]
        tile = s.stage.map.tile(*position)
        if tuple(position) in mechanics.solid_positions(s):raise ValueError('Device occupies tile')
        if tile.buildable not in [d.position_type, "ALL"]:
            raise ValueError("Invalid deployment tile")
        count = s.deploy_counts.get(key, 0)
        cost = deployment_cost(d.cost, count)
        if s.dp + 1e-9 < cost:
            raise ValueError("Insufficient DP")
        s.dp -= cost
        s.deploy_counts[key] = count + 1
        op = Unit(
            key,
            d,
            d.hp,
            tuple(position),
            direction=direction,
            deployed_at=self.current_time,
            paid_cost=cost,
            generation=count + 1,
            skill=SkillState(d.skill.initial_sp if d.skill else 0),
        )
        s.operators[key] = op
        talents.on_deploy(self,op)
        summons.owner_deployed(s, op)
        mechanics.gravity_switches(s)
        self._log("DEPLOY", unit=key, position=position, direction=direction, cost=cost)
        return op

    def deploy_summon(self, card_id, position, direction=0):
        return summons.deploy(self, card_id, position, direction)

    def retreat_summon(self, key):
        return summons.retreat(self, key)

    def retreat(self, key):
        s = self.state
        op = s.operators.get(key)
        if s.done or op is None or not op.alive:
            raise ValueError("Operator unavailable")
        refund = retreat_refund(op.paid_cost)
        s.dp = min(s.stage.max_cost, s.dp + refund)
        self._remove_operator(op)
        talents.refresh(self)
        mechanics.gravity_switches(s)
        self._log("RETREAT", unit=key, refund=refund)

    def _remove_operator(self, op):
        s = self.state
        op.alive = False
        s.redeploy_at[op.id] = self.current_time + op.data.redeploy
        for eid in list(op.blocked_enemies):
            release(s.enemies[eid], s.allies)
        summons.owner_removed(self, op)

    def add_modifier(self, modifier):
        units = {**self.state.enemies, **self.state.allies}
        valid = {
            f"{k}_{suffix}"
            for k in ["ATK", "DEF", "ASPD", "MOVE_SPEED", "BLOCK_COUNT", "DAMAGE_SCALE", "HP_REGEN"]
            for suffix in ["ADD", "MULTIPLY"]
        }
        if modifier.target not in units or modifier.kind not in valid:
            raise ValueError("Modifier target/kind")
        if (
            not all(
                math.isfinite(v)
                for v in [modifier.value, modifier.start_time, modifier.end_time]
            )
            or modifier.start_time < self.current_time
            or modifier.end_time <= modifier.start_time
        ):
            raise ValueError("Modifier interval/value")
        units[modifier.target].modifiers.append(modifier)
        self.state.queue.push(modifier.start_time, "MODIFIER_BOUNDARY", modifier.target)
        self.state.queue.push(modifier.end_time, "MODIFIER_BOUNDARY", modifier.target)

    def activate_skill(self, key):
        op = self.state.allies.get(key)
        if self.state.done or not op or not op.alive or not op.data.skill:
            raise ValueError("No usable skill")
        data = op.data.skill
        if (data.trigger != "MANUAL_TRIGGER" or (op.skill.is_active(data,self.current_time) and data.mode!='TOGGLE')
                or op.skill.sp + 1e-8 < data.cost):
            raise ValueError("Skill not ready")
        self._start_skill(op)

    def _start_skill(self, op):
        from arknights_sim.skills.runtime import heal_self
        data = op.data.skill
        op.skill.sp = 0
        if data.mode=='TOGGLE' and op.skill.active:
            source=f'{op.id}:{op.generation}:{data.id}'
            op.modifiers[:]=[m for m in op.modifiers if m.source!=source]
            op.skill.active=False
            self._log('SKILL_END',unit=op.id)
            return
        if data.mode == 'INSTANT_HEAL':
            heal_self(self,op,op.max_hp*data.heal_ratio+data.heal_flat)
            self._log('SKILL_START',unit=op.id,skill=data.id,end=self.current_time)
            self._log('SKILL_END',unit=op.id)
            return
        end = self.current_time + data.duration
        op.skill.active_until = end
        op.skill.active = True
        talents.on_skill(self,op)
        if data.effect_target == "OWNER_SUMMONS":
            targets = [u for u in self.state.summons.values() if u.alive and u.owner_id == op.id]
        else:
            targets = [op]
        for target in targets:
            apply_timed_skill(target, op, self.current_time)
        if data.mode not in ('TOGGLE','INFINITE'):
            self.state.queue.push(end, "SKILL_END", op.id, op.generation)
        self._log("SKILL_START", unit=op.id, skill=data.id, end=end)

    def remaining(self, e):
        pos = e.position
        dist = 0
        for node in mechanics.unit_path(self.state,e,self.paths)[e.node :]:
            if node.kind in WAIT_KINDS:
                continue
            if node.kind == 'DISAPPEAR': continue
            if node.kind == 'APPEAR_AT_POS':
                pos = node.position
                continue
            dist += math.dist(pos, node.position)
            pos = node.position
        return dist

    def _move(self, e, dt, start):
        if not e.alive or e.escaped or e.blocked_by or start < e.frozen_until or mechanics.roadblock_binds(self.state,e):
            return
        speed = (
            modified(e.data.speed, "MOVE_SPEED", e.modifiers, start)
            * self.state.stage.move_multiplier
        )
        if speed <= 0:
            return
        budget = dt
        elapsed = 0
        path = mechanics.unit_path(self.state,e,self.paths)
        while budget > 1e-10 and e.node < len(path):
            now = start + elapsed
            if e.wait_until > now + 1e-10:
                wait = min(budget, e.wait_until - now)
                budget -= wait
                elapsed += wait
                if now + wait >= e.wait_until - 1e-9:
                    self.trace.log(now + wait, "MOVING", unit=e.id)
                continue
            node = path[e.node]
            if node.kind == 'DISAPPEAR':
                e.hidden = True
                e.node += 1
                release(e, {**self.state.allies, **self.state.devices})
                self.trace.log(now, 'DISAPPEAR', unit=e.id)
                continue
            if node.kind == 'APPEAR_AT_POS':
                e.hidden = False
                e.position = node.position
                e.node += 1
                self.trace.log(now, 'APPEAR', unit=e.id, position=e.position)
                continue
            if node.kind in WAIT_KINDS:
                e.wait_until = stage_events.route_wait_deadline(self.state, node, now)
                e.node += 1
                self.trace.log(now, "WAIT", unit=e.id, duration=max(0, e.wait_until-now),
                               wait_kind=node.kind, deadline=e.wait_until)
                continue
            distance = math.dist(e.position, node.position)
            factor=mechanics.speed_factor(self.state,e,(node.position[0]-e.position[0],node.position[1]-e.position[1]))
            travel = distance / (speed*factor)
            if travel <= budget + 1e-10:
                e.position = node.position
                e.node += 1
                budget = max(0, budget - travel)
                elapsed += travel
                if node.kind == "GOAL":
                    e.escaped = True
                    e.alive = False
                    if e.data.faction == 'ESCORT':
                        self.state.escorts_saved += 1
                        self.trace.log(start + elapsed, 'ESCORT_SAVED', unit=e.id)
                    else:
                        self.state.escaped += 1
                        self.state.life = max(0, self.state.life - e.data.life_cost)
                        self.trace.log(start + elapsed, "ESCAPE", unit=e.id)
                    return
                if node.kind == "MOVE":
                    self.trace.log(
                        start + elapsed, "WAYPOINT", unit=e.id, position=node.position
                    )
            else:
                fraction = budget / travel
                e.position = tuple(
                    a + (b - a) * fraction for a, b in zip(e.position, node.position)
                )
                budget = 0

    def _advance(self, time):
        s = self.state
        old = self.current_time
        dt = time - old
        mechanics.advance(s,old,time)
        terrain.advance(self,dt)
        regenerate(self, old, time)
        s.dp = min(s.stage.max_cost, s.dp + dt / s.stage.cost_interval)
        for op in sorted(s.allies.values(), key=lambda u: u.id):
            if op.alive:
                # Recover only the part outside an active duration.
                op.skill.recover(
                    op.data.skill, max(0, time - max(old, op.skill.active_until))*talents.natural_sp_rate(op), time
                )
        for e in sorted(s.enemies.values(), key=lambda u: u.id):
            self._move(e, dt, old)
        s.clock.time = time

    def _event(self, event):
        s = self.state
        s.events_processed += 1
        if event.kind == "SPAWN":
            spawn = event.payload[0]
            d = next(d for d in s.stage.enemies if d.id == spawn.enemy_id)
            route = s.stage.routes[spawn.route_index]
            pos = tuple(
                v + s.rng.uniform(-r, r) if r else v
                for v, r in zip(route.start, route.random_range)
            )
            eid = f"enemy_{s.spawned:04d}"
            s.spawned += 1
            e = Unit(eid, d, d.hp, pos, route_index=spawn.route_index,
                     deployed_at=self.current_time, self_damage_at=self.current_time+1)
            s.enemies[eid] = e
            e.enemy_skill_ready={i:self.current_time+skill.initial for i,skill in enumerate(d.skills)}
            if d.self_damage_per_second:
                s.queue.push(self.current_time+1,'SELF_DAMAGE',eid)
            stage_events.register_spawn(s, spawn, eid)
            self._log(
                "ENEMY_SPAWN",
                unit=eid,
                enemy=d.id,
                route=spawn.route_index,
                position=pos,
            )
        elif event.kind == "HIT":
            from arknights_sim.combat.runtime import resolve_hit
            resolve_hit(self,event.payload[0])
        elif event.kind == 'TALENT_EVENT':
            talents.event(self,event.payload)
        elif event.kind == 'DEVICE_PULSE':
            from arknights_sim.mechanics.chapter_devices import pulse
            pulse(self,event.payload[0])
        elif event.kind == 'BURN_TICK':
            from arknights_sim.mechanics.escorts import burn_tick
            burn_tick(self,event.payload)
        elif event.kind == 'SELF_DAMAGE':
            unit=s.enemies[event.payload[0]]
            if unit.alive:
                from arknights_sim.combat.runtime import deal_damage
                if talents.effect_enabled(unit,'self_damage',self.current_time):
                    deal_damage(self,unit,unit.data.self_damage_per_second,'TRUE',event_kind='SELF_DAMAGE',defense_sp=False)
                if unit.alive:s.queue.push(self.current_time+1,'SELF_DAMAGE',unit.id)
        elif event.kind == 'AREA_HIT':
            from arknights_sim.combat.runtime import area_hit
            area_hit(self,event.payload)
        elif event.kind == "MODIFIER_BOUNDARY":
            self._log("MODIFIER_BOUNDARY", unit=event.payload[0])
        elif event.kind == "SKILL_END":
            key, generation = event.payload
            op = s.allies[key]
            if op.alive and op.generation == generation:
                op.skill.active = False
                self._log("SKILL_END", unit=key)
        else:
            raise ValueError(event.kind)

    def _attack(self, unit, target, *, effect=None, extra_targets=()):
        if target is None or self.current_time + 1e-9 < unit.ready_at or self.current_time < unit.frozen_until:
            return
        if unit.id in self.state.enemies and unit.blocked_by and unit.tremble_until>self.current_time:
            return
        base_speed = getattr(unit.data,'base_attack_speed',100)
        speed = modified(base_speed, "ASPD", unit.modifiers, self.current_time)
        if self.current_time < unit.cold_until:speed-=30
        if unit.id in self.state.allies:speed+=mechanics.operator_as(self.state,unit)
        if speed <= 0:
            return
        interval = unit.data.interval * base_speed / speed
        windup = min(self.state.windup, interval)
        unit.ready_at = self.current_time + interval
        procs,extra_hits=talents.on_attack(self,unit) if unit.id in self.state.allies else ((),0)
        if unit.id in self.state.enemies and unit.data.projectile_delay:
            atk=modified(unit.data.atk,'ATK',unit.modifiers,self.current_time)
            self.state.queue.push(self.current_time+unit.data.projectile_delay,'AREA_HIT',
                                  unit.id,tuple(target.position),atk,unit.data.projectile_radius)
            self._log('PROJECTILE_LAUNCH',unit=unit.id,position=target.position,
                      impact=self.current_time+unit.data.projectile_delay)
            return
        if effect and effect.mode=='NEXT_ATTACK':
            unit.skill.sp-=effect.cost
            self._log('SKILL_TRIGGER',unit=unit.id,skill=effect.id)
        hits=effect.hits if effect and (effect.mode=='NEXT_ATTACK' or effect.hits!=1) else getattr(unit.data,'normal_hits',1)
        hits+=extra_hits
        # MULTIHIT-001 ASSUMED: stagger by 0.1s until prefab animation hit
        # markers are calibrated. Distinct hits apply defense separately.
        for ti,victim in enumerate((target,*extra_targets)):
            scale=effect.attack_scale if effect else 1
            if unit.id in self.state.enemies and not unit.blocked_by:
                scale*=unit.data.ranged_damage_scale
            if unit.id in self.state.allies and victim.blocked_by != unit.id and not (effect and effect.ranged_penalty_removed):
                scale*=unit.data.ranged_attack_scale
            for hi in range(hits):
                event=AttackEvent(unit.id,victim.id,self.current_time,
                    self.current_time+windup+hi*min(.1,interval/max(1,hits)),unit.ready_at,
                    unit.generation,victim.generation,effect,hi,ti==0,scale,
                    attack_revision=unit.attack_revision,talent_procs=procs,target_index=ti)
                self.state.queue.push(event.hit,'HIT',event)
                unit.last_attack_end=event.hit
            self._log('ATTACK_START',attacker=unit.id,target=victim.id,hit=self.current_time+windup)

    def _operator_attack(self, op):
        from arknights_sim.skills.runtime import active_effect
        from arknights_sim.combat.targeting import in_range
        effect=active_effect(op,self.current_time)
        data=op.data.skill
        if data and data.mode=='NEXT_ATTACK' and op.skill.sp+1e-8>=data.cost:
            effect=data
        if effect and effect.stop_attack:return
        capacity=max(0,int(modified(op.data.block_count,'BLOCK_COUNT',op.modifiers,self.current_time)))
        limit=capacity if op.data.target_mode=='BLOCK_COUNT' else 100000 if op.data.target_mode=='ALL' else 1
        if effect:
            limit=max(limit,capacity if effect.max_targets==-1 else effect.max_targets)
        candidates=sorted((e for e in self.state.enemies.values() if e.alive and e.data.faction=='HOSTILE'
                           and not (e.dying_until and talents.talents(op,'dying'))
                           and mechanics.revealed(self.state,e) and in_range(op,e)),
                          key=lambda e:(e.blocked_by!=op.id,
                                        bool(talents.talents(op,'drone_priority')) and e.data.motion!='FLY',
                                        self.remaining(e),e.id))[:limit]
        if candidates:
            self._attack(op,candidates[0],effect=effect,extra_targets=tuple(candidates[1:]))
        else:
            self._attack(op,mechanics.roadblock_target(self.state,op),effect=effect)

    def _remove_enemy(self, unit, killer=None):
        if not unit.alive: return
        unit.alive = False
        if unit.data.faction == 'ESCORT':
            self.state.escorts_dead += 1
            self.state.life = max(0, self.state.life-unit.data.death_life_cost)
            self._log('ESCORT_DEATH', unit=unit.id, life_cost=unit.data.death_life_cost)
        else:
            self.state.killed += 1
        release(unit, {**self.state.allies, **self.state.devices})
        talents.on_death(self,unit,killer)

    def _tick(self):
        s = self.state
        mechanics.gravity_switches(s)
        for op in sorted(s.allies.values(),key=lambda u:u.id):
            data=op.data.skill
            if op.alive and data and data.mode in ('TIMED','INFINITE') and data.trigger=='AUTO_TRIGGER' and not op.skill.is_active(data,self.current_time) and op.skill.sp+1e-8>=data.cost:
                self._start_skill(op)
        blockers={**s.allies,**{k:d for k,d in s.devices.items() if d.data.kind=='crate'}}
        for op in sorted(s.allies.values(), key=lambda u: u.id):
            capacity = max(
                0,
                modified(
                    op.data.block_count, "BLOCK_COUNT", op.modifiers, self.current_time
                ),
            )
            while (
                sum(s.enemies[e].data.block_weight for e in op.blocked_enemies)
                > capacity
            ):
                release(s.enemies[op.blocked_enemies[-1]], s.allies)
        for e in sorted(s.enemies.values(), key=lambda u: u.id):
            if e.alive:
                op = try_block(e, blockers, s.enemies, self.current_time)
                if op:
                    self._log("BLOCK", enemy=e.id, operator=op.id)
        talents.refresh(self)
        for op in sorted(s.allies.values(), key=lambda u: u.id):
            if op.alive:
                if op.data.damage_type == "NONE":
                    continue
                if op.data.damage_type == "HEAL":
                    from arknights_sim.combat.targeting import HealingTargetSelector
                    self._attack(op, HealingTargetSelector.select(op, s.allies.values()))
                    continue
                self._operator_attack(op)
        for e in sorted(s.enemies.values(), key=lambda u: u.id):
            if e.data.faction=='ESCORT':
                from arknights_sim.mechanics.escorts import tick
                tick(self,e)
            if e.alive and not e.hidden and e.data.faction == 'HOSTILE':
                candidates={k:o for k,o in blockers.items() if k in s.devices or not terrain.concealed(s,o)}
                candidates.update({k:o for k,o in s.enemies.items() if o.data.faction == 'ESCORT'})
                targets=[]
                for _ in range(e.data.max_targets):
                    chosen=EnemyTargetSelector.select(e,candidates)
                    if chosen is None:break
                    targets.append(chosen);candidates.pop(chosen.id)
                if targets:self._attack(e,targets[0],extra_targets=tuple(targets[1:]))
        s.events_processed += 1

    def _finish(self):
        s = self.state
        if s.life <= 0 or (
            stage_events.scheduling_complete(s)
            and s.spawned == len(s.stage.spawns)
            and not any(e.alive and e.data.faction == 'HOSTILE' for e in s.enemies.values())
        ):
            s.done = True
            s.result = "LOSS" if s.life <= 0 else "WIN"
            self._log(
                "BATTLE_END",
                result=s.result,
                killed=s.killed,
                escaped=s.escaped,
                life=s.life,
            )

    def _settle_events(self):
        # A stage gate can enqueue a spawn after a lethal HIT. Drain to a fixed
        # point while retaining the established priorities within every drain.
        while True:
            stage_events.settle(self)
            if self.state.queue.peek() > self.current_time + 1e-9:
                break
            while self.state.queue.peek() <= self.current_time + 1e-9:
                self._event(self.state.queue.pop())

    def run_until(self, target, *, stop_when=None):
        s = self.state
        if not math.isfinite(target) or target < self.current_time - 1e-9:
            raise ValueError("Time cannot go backwards")
        # Public action boundaries lie on the fixed clock, independent of step partitioning.
        if abs(target / s.clock.dt - round(target / s.clock.dt)) > 1e-6:
            raise ValueError("Time must align to dt grid")
        while not s.done:
            next_time = min(s.queue.peek(), s.clock.next_tick(), stage_events.next_boundary(s))
            if next_time > target + 1e-9:
                break
            self._advance(next_time)
            self._settle_events()
            if abs(s.clock.next_tick() - next_time) < 1e-8:
                s.clock.tick += 1
                self._tick()
            # Zero-windup events generated on this tick are resolved before terminal check.
            self._settle_events()
            self._finish()
            # Optional observer only at settled tick boundaries: core knows no
            # agent concepts. Returning True lets an outer driver pause safely.
            if (
                stop_when is not None
                and (
                    s.done
                    or abs(
                        s.current_time / s.clock.dt - round(s.current_time / s.clock.dt)
                    )
                    < 1e-6
                )
                and stop_when(s)
            ):
                break
        return s

    def step(self, seconds=None):
        return self.run_until(
            self.current_time + (self.state.clock.dt if seconds is None else seconds)
        )

    def run(self, max_seconds=300):
        return self.run_until(
            round(max_seconds / self.state.clock.dt) * self.state.clock.dt
        )
