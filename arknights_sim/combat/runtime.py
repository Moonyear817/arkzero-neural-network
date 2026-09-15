"""Shared damage pipeline for attacks, projectiles, DoT and map devices."""
import math
from .damage import DamageCalculator
from arknights_sim.skills.modifier import modified
from arknights_sim.talents import runtime as talents
from arknights_sim.mechanics import runtime as mechanics, terrain
from arknights_sim.summons import runtime as summons


def attack_power(sim, unit):
    value=modified(unit.data.atk,'ATK',unit.modifiers,sim.current_time)
    if unit.id in sim.state.enemies and unit.hp<talents.max_hp(unit)/2 and talents.effect_enabled(unit,'low_hp_attack_bonus',sim.current_time):
        value*=1+unit.data.low_hp_attack_bonus
    if unit.id in sim.state.allies:value*=mechanics.operator_attack(sim.state,unit)
    return value


def defenses(sim, target):
    defense=modified(target.data.defense,'DEF',target.modifiers,sim.current_time)
    resistance=target.data.resistance
    if getattr(target,'frozen_until',0)>sim.current_time:resistance-=15
    if target.id in sim.state.enemies and target.hp<talents.max_hp(target)/2:
        defense*=1+target.data.half_hp_defense_bonus
        resistance*=1+target.data.half_hp_resistance_bonus
    if target.id in sim.state.allies:defense=terrain.defense(sim.state,target,defense)
    return defense,resistance


def remove_dead(sim, target, source):
    state=sim.state
    if target.id in state.operators:
        sim._remove_operator(target)
        talents.refresh(sim)
    elif target.id in state.summons:summons.remove(sim,target,'DEATH')
    elif target.id in state.devices:
        if target.data.kind=='roadblock':mechanics.destroy_roadblock(sim,target.id)
        else:mechanics.remove(sim,target.id)
    else:sim._remove_enemy(target,source)
    sim._log('DEATH',unit=target.id)


def deal_damage(sim, target, amount, kind, source=None, *, event_kind='DAMAGE',
                reflect=True, can_dodge=True, defense_sp=True):
    """Receive hooks see every attempt, even an evaded hit (Ulpianus)."""
    if not target.alive:return 0.
    if target.id in sim.state.allies:
        talents.pre_receive(sim,target)
        if can_dodge and talents.dodge(sim,target,kind):
            sim._log('DODGE',unit=target.id,attacker=source.id if source else None)
            return 0.
    scale=talents.incoming_scale(sim,target,source,kind)
    # Fragile takes the strongest active source, never multiplies itself.
    if kind in ('PHYSICAL','ARTS'):
        scale*=max((m.value for m in target.modifiers if m.kind=='FRAGILE_MULTIPLY'
                    and m.start_time<=sim.current_time and (m.end_time is None or sim.current_time<m.end_time)),default=1.)
    amount=max(0.,amount*scale)
    floor=talents.hp_floor(sim,target,source,amount)
    target.hp=max(floor,target.hp-amount)
    sim._log(event_kind,attacker=source.id if source else None,target=target.id,amount=amount,hp=target.hp)
    if target.id in sim.state.allies and amount>0 and defense_sp:
        target.skill.gain(target.data.skill,'DEFENSE_RECOVERY',sim.current_time)
    talents.after_receive(sim,target,source,amount,reflect=reflect)
    if target.hp<=0:remove_dead(sim,target,source)
    return amount


def resolve_hit(sim, attack):
    s=sim.state
    units={**s.enemies,**s.allies,**s.devices}
    attacker=units.get(attack.attacker);target=units.get(attack.target)
    if attacker is None or target is None:return
    if ((not attacker.alive and not attack.is_dot) or attacker.generation!=attack.generation
            or not target.alive or getattr(target,'hidden',False) or target.generation!=attack.target_generation
            or (not attack.is_dot and getattr(attacker,'attack_revision',0)!=attack.attack_revision)):
        return
    talents.refresh(sim)
    from arknights_sim.skills.runtime import hit_effects, heal_self
    if attacker.data.damage_type=='HEAL' and attack.damage_type is None:
        heal_self(sim,target,attack_power(sim,attacker),as_healing=True,healer=attacker.id)
        return
    kind=attack.damage_type or (attack.skill_effect.damage_type if attack.skill_effect and attack.skill_effect.damage_type else attacker.data.damage_type)
    scale=talents.before_hit(sim,attacker,target,attack)
    atk=(attack.fixed_atk if attack.fixed_atk is not None else attack_power(sim,attacker))*attack.attack_scale*scale
    defense,resistance=defenses(sim,target)
    atk,defense,resistance=talents.offensive(sim,attacker,target,attack,atk,defense,resistance)
    if attack.sp_eligible and attack.hit_index==0 and attacker.id in s.allies and not (attack.skill_effect and attack.skill_effect.mode=='NEXT_ATTACK'):
        attacker.skill.gain(attacker.data.skill,'ATTACK_RECOVERY',sim.current_time)
    damage=DamageCalculator.calculate(kind,atk,defense,resistance)
    damage=modified(damage,'DAMAGE_SCALE',attacker.modifiers,sim.current_time)*talents.outgoing_scale(attacker,target,attack)
    amount=deal_damage(sim,target,damage,kind,attacker,defense_sp=not attack.is_dot)
    if attacker.id in s.enemies and attacker.data.lifesteal and talents.effect_enabled(attacker,'lifesteal',sim.current_time):
        heal_self(sim,attacker,amount*attacker.data.lifesteal,as_healing=True)
    hit_effects(sim,attacker,target,attack,amount)
    talents.after_hit(sim,attacker,target,attack,amount)
    if attacker.id in s.enemies and attacker.data.splash:
        for other in [*s.allies.values(),*(d for d in s.devices.values() if d.data.kind=='crate')]:
            if other is target or not other.alive:continue
            if abs(other.position[0]-target.position[0])<=1 and abs(other.position[1]-target.position[1])<=1:
                df,res=defenses(sim,other)
                deal_damage(sim,other,DamageCalculator.calculate('PHYSICAL',atk,df,res),
                            'PHYSICAL',attacker,event_kind='SPLASH_DAMAGE')


def area_hit(sim, payload):
    source_id,position,atk,radius=payload
    s=sim.state
    source=s.enemies.get(source_id)
    targets=[*s.allies.values(),*(e for e in s.enemies.values() if e.data.faction=='ESCORT'),
             *(d for d in s.devices.values() if d.data.kind=='crate')]
    talents.refresh(sim)
    for target in sorted(targets,key=lambda u:u.id):
        if not target.alive or getattr(target,'hidden',False) or math.dist(target.position,position)>radius:continue
        defense,resistance=defenses(sim,target)
        damage=DamageCalculator.calculate('PHYSICAL',atk,defense,resistance)
        deal_damage(sim,target,damage,'PHYSICAL',source,event_kind='AREA_DAMAGE')
