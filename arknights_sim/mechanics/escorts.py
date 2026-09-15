"""Neutral escort actions. They share event/HP state, never the player squad.

ASSUMED: skill polls use the combat tick, each phase uses its own authored
cooldown, and burn's first damage tick is one second after application.
"""
import math
from arknights_sim.combat.attack import AttackEvent


def targets(sim, unit, radius):
    return sorted((e for e in sim.state.enemies.values() if e.alive and not e.hidden
                   and e.data.faction=='HOSTILE' and math.dist(e.position,unit.position)<=radius),
                  key=lambda e:(sim.remaining(e),e.id))


def apply_burn(sim, target, spec):
    target.burn_started=sim.current_time
    target.burn_until=sim.current_time+spec['dragon_fire.duration']
    target.burn_base=spec['dragon_fire.baseDamage']
    target.burn_growth=spec['dragon_fire.addOnDamage']
    target.burn_ramp=spec['dragon_fire.addOnDuration']
    target.burn_generation+=1
    sim.state.queue.push(sim.current_time+1,'BURN_TICK',target.id,target.generation,target.burn_generation)
    sim._log('BURN_APPLIED',unit=target.id,until=target.burn_until)


def burn_tick(sim, payload):
    key,generation,burn_generation=payload
    target=sim.state.enemies.get(key) or sim.state.allies.get(key)
    if target is None or not target.alive or target.generation!=generation or target.burn_generation!=burn_generation:return
    if sim.current_time >= target.burn_until:return
    amount=target.burn_base+target.burn_growth*min(1,(sim.current_time-target.burn_started)/max(target.burn_ramp,1e-8))
    from arknights_sim.combat.runtime import deal_damage
    deal_damage(sim,target,amount,'TRUE',event_kind='BURN_DAMAGE',defense_sp=False)
    if target.alive:sim.state.queue.push(sim.current_time+1,'BURN_TICK',key,generation,burn_generation)


def tick(sim, unit):
    if not unit.alive or unit.hidden or unit.data.faction!='ESCORT' or unit.data.atk<=0:return
    available=targets(sim,unit,unit.data.attack_range)
    if available:sim._attack(unit,available[0])
    half=unit.hp<unit.data.hp/2
    for index,skill in enumerate(unit.data.skills):
        if skill.half_hp != half or sim.current_time+1e-8<unit.enemy_skill_ready[index]:continue
        bb=dict(skill.parameters)
        if skill.kind=='DragonFire':
            candidates=[e for e in available if sim.current_time>=e.burn_until]
            if not candidates:continue
            apply_burn(sim,candidates[0],bb)
        elif skill.kind=='DanceFire':
            candidates=targets(sim,unit,bb['range_radius'])
            if not candidates:continue
            for e in candidates:
                sim.state.queue.push(sim.current_time,'HIT',AttackEvent(unit.id,e.id,sim.current_time,sim.current_time,sim.current_time,
                    unit.generation,e.generation,attack_scale=bb['atk_scale']))
        else:raise ValueError('Unsupported escort skill '+skill.kind)
        unit.enemy_skill_ready[index]=sim.current_time+skill.cooldown
        sim._log('ESCORT_SKILL',unit=unit.id,skill=skill.kind,half_hp=half)
