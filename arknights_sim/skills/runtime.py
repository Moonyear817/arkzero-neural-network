"""Timed effect application, independent of UI and action selection."""
from .modifier import Modifier, modified


def apply_timed_skill(unit, owner, start):
    data = owner.data.skill
    source = f"{owner.id}:{owner.generation}:{data.id}"
    end=None if data.mode in ('TOGGLE','INFINITE') else owner.skill.active_until
    for kind, value in (("ATK_PERCENT", data.attack_multiplier-1),
                        ("DEF_PERCENT", data.defense_multiplier-1),
                        ("ASPD_ADD", (data.attack_speed_multiplier-1)*100),
                        ("HP_REGEN_ADD", data.hp_recovery_per_sec)):
        unit.modifiers.append(Modifier(source, unit.id, kind, value,
                                       start, end))
    block = data.block_delta if data.block_override is None else data.block_override-unit.data.block_count
    if block:
        unit.modifiers.append(Modifier(source,unit.id,'BLOCK_COUNT_ADD',block,start,end))
    if data.regen_ratio:
        unit.modifiers.append(Modifier(source,unit.id,'HP_REGEN_ADD',unit.data.hp*data.regen_ratio,start,end))


def active_effect(unit, time):
    data=unit.data.skill
    return data if unit.skill.is_active(data,time) else None


def heal_self(sim, unit, amount, *, as_healing=False, healer=None):
    if amount <= 0 or not unit.alive: return
    from arknights_sim.talents.runtime import max_hp, healing_scale
    if as_healing:
        if not getattr(unit.data,'healable',True) or (unit.id in sim.state.allies and active_effect(unit,sim.current_time) and unit.data.skill.prohibit_healing):return
        amount*=healing_scale(unit)
    before=unit.hp
    unit.hp=min(max_hp(unit),unit.hp+amount)
    sim._log('HEAL' if as_healing else 'HP_RECOVERY',unit=unit.id,target=unit.id,
             attacker=healer,amount=unit.hp-before,hp=unit.hp)


def hit_effects(sim, attacker, target, attack, damage):
    data=attack.skill_effect
    if not data:return
    if attack.hit_index == 0 and attack.sp_eligible:
        heal_self(sim,attacker,attacker.max_hp*data.heal_ratio+data.heal_flat)
    if data.lifesteal:heal_self(sim,attacker,damage*data.lifesteal,as_healing=True)
    if data.debuff_duration and target.alive:
        source=f'{attacker.id}:{data.id}'
        # Refresh the same debuff; repeated applications must not stack.
        target.modifiers[:]=[m for m in target.modifiers if m.source!=source]
        for kind,value in [('ATK_MULTIPLY',data.target_atk_multiplier),('MOVE_SPEED_MULTIPLY',data.target_speed_multiplier)]:
            if value != 1:
                target.modifiers.append(Modifier(source,target.id,kind,value,sim.current_time,sim.current_time+data.debuff_duration))
        sim.state.queue.push(sim.current_time+data.debuff_duration,'MODIFIER_BOUNDARY',target.id)
    if data.kill_refill and not target.alive:
        attacker.skill.sp=attacker.data.skill.cost


def regenerate(sim, old, time):
    # Queue boundaries split each active interval exactly. HP recovery is a
    # stat, not a healing action; unhealable summons still recover this way.
    for unit in sim.state.allies.values():
        if not unit.alive:
            continue
        rate = modified(0, "HP_REGEN", unit.modifiers, old)
        if rate > 0 and unit.hp < unit.max_hp:
            before = unit.hp
            unit.hp = min(unit.max_hp, unit.hp + rate * (time - old))
            sim.trace.log(time, "HP_RECOVERY", unit=unit.id,
                          amount=unit.hp - before, hp=unit.hp)
