"""Talent hooks with state-local counters, timers and RNG.

Source: pinned GameData and research/mechanics/talents/*.json. Hook ordering is
explicit and deterministic; exact client frame ordering remains ASSUMED.
"""
import math
from dataclasses import replace
from arknights_sim.skills.modifier import Modifier, modified


def talents(unit, handler=None):
    return tuple(t for t in getattr(unit.data, 'talents', ())
                 if t.handler and (handler is None or t.handler == handler))


def max_hp(unit):
    return getattr(unit, 'max_hp', unit.data.hp)


def ratio(unit):
    return unit.hp / max(1, max_hp(unit))


def active(unit, time):
    return unit.skill.is_active(unit.data.skill, time)


def in_cells(source, target, cells):
    # Talent auras are centered on the operator, with authored orientation.
    for x, y in cells:
        for _ in range(source.direction):
            x, y = -y, x
        if abs(source.position[0]+x-target.position[0]) <= .5 and abs(source.position[1]+y-target.position[1]) <= .5:
            return True
    return False


NEIGHBORS = tuple((x,y) for x in (-1,0,1) for y in (-1,0,1))


def nearby_count(sim, source):
    return sum(e.alive and not e.hidden and e.data.faction == 'HOSTILE'
               and in_cells(source,e,NEIGHBORS) for e in sim.state.enemies.values())


def add_sp(sim, unit, amount):
    data = unit.data.skill
    if data and (not active(unit, sim.current_time) or data.mode == 'TOGGLE'):
        before = unit.skill.sp
        unit.skill.sp = min(data.cost*data.max_charges, before+amount)
        if unit.skill.sp != before:
            sim._log('TALENT_SP', unit=unit.id, amount=unit.skill.sp-before)


def chance(sim, unit, talent, probability, trigger):
    # Only eligible events draw. This RNG is cloned/saved with GameState.
    p = max(0., min(1., probability))
    roll = sim.state.rng.random()
    hit = roll < p
    sim._log('TALENT_ROLL', unit=unit.id, talent=talent.name, trigger=trigger,
             probability=p, roll=roll, triggered=hit)
    return hit


def set_buff(sim, unit, source, kind, value, duration):
    unit.modifiers[:] = [m for m in unit.modifiers if not (m.source==source and m.kind==kind)]
    until = None if duration is None else sim.current_time+duration
    unit.modifiers.append(Modifier(source,unit.id,kind,value,sim.current_time,until))
    if until is not None:
        sim.state.queue.push(until,'MODIFIER_BOUNDARY',unit.id)


def resize_hp(unit, value):
    # Attribute changes preserve HP ratio (not a healing event).
    old = max_hp(unit)
    value = max(1., value)
    if abs(old-value)>1e-9:
        unit.hp = min(value, unit.hp*value/max(1.,old))
        unit.maximum_hp = value


def refresh(sim):
    """Rebuild current auras/conditional stats; expiring hit buffs stay separate."""
    units = sorted((u for u in sim.state.allies.values() if u.alive),key=lambda u:u.id)
    sources = [(u,t) for u in units for t in talents(u)]
    if not sources and not any(m.source.startswith('talent-live:') for u in units for m in u.modifiers):
        return
    for target in units:
        modifiers = []
        def buff(source, t, kind, value):
            if value:
                modifiers.append(Modifier(f'talent-live:{source.id}:{t.source}:{t.prefab}',
                                          target.id,kind,float(value),0,None))
        for source,t in sources:
            h = t.handler
            own = source is target
            if own:
                if h in ('static','hp_def_penetration'):
                    for field, stat in [('atk','ATK'),('def','DEF'),('max_hp','MAX_HP')]:
                        buff(source,t,stat+'_PERCENT',t.get(field))
                elif h == 'block_stats':
                    for field,stat in [('atk','ATK'),('def','DEF')]:
                        buff(source,t,stat+'_PERCENT',t.get(field)+t.get(field+'_add')*len(source.blocked_enemies))
                elif h == 'defense_dodge':
                    buff(source,t,'DEF_PERCENT',t.get('def'))
                elif h == 'time_attack_speed':
                    stacks = min(t.get('max_stack_cnt'), math.floor((sim.current_time-source.deployed_at+1e-8)/t.get('interval')))
                    buff(source,t,'ASPD_ADD',stacks*t.get('attack_speed'))
                elif h == 'kill_attack_speed':
                    buff(source,t,'ASPD_ADD',source.talent_state.get('kills:'+t.prefab,0)*t.get('attack_speed'))
                elif h == 'berserk':
                    factor = min(1.,max(0.,(1-ratio(source))/(1-t.get('min_hp_ratio'))))
                    buff(source,t,'ASPD_ADD',t.get('min_attack_speed')*factor)
                    buff(source,t,'DEF_PERCENT',t.get('min_def')*factor)
                elif h == 'consecutive_hits':
                    buff(source,t,'ATK_PERCENT',source.talent_state.get('combo_stacks',0)*t.get('atk'))
                elif h == 'hp_steal_dot':
                    buff(source,t,'MAX_HP_ADD',source.talent_state.get('hp_stolen',0))
                elif h == 'module_healthy_speed' and ratio(source)>.5:
                    buff(source,t,'ASPD_ADD',t.get('attack_speed'))
                elif h == 'module_unblocked_speed' and not source.blocked_enemies:
                    buff(source,t,'ASPD_ADD',t.get('attack_speed'))
            if h == 'abyssal_kill_stats' and (own or 'abyssal' in target.data.factions):
                count = source.talent_state.get('kills:'+t.prefab,0)
                prefix = '' if own else 'ulpia_t_1[abyssal].'
                buff(source,t,'ATK_ADD',count*t.get(prefix+'atk'))
                buff(source,t,'MAX_HP_ADD',count*t.get(prefix+'max_hp'))
            if h == 'high_block_aura' and target.data.position_type == 'MELEE':
                capacity = modified(target.data.block_count,'BLOCK_COUNT',target.modifiers,sim.current_time)
                if capacity >= 3:
                    buff(source,t,'ASPD_ADD',t.get('attack_speed'))
                    buff(source,t,'DEF_PERCENT',t.get('def'))
            elif h == 'deploy_melee_aura' and sim.current_time < source.deployed_at+t.get('duration') and target.data.position_type=='MELEE':
                buff(source,t,'ATK_PERCENT',t.get('atk'))
                buff(source,t,'DEF_PERCENT',t.get('def'))
            elif h == 'nearby_melee_attack' and target.data.position_type=='MELEE' and in_cells(source,target,t.attack_range):
                buff(source,t,'ATK_PERCENT',t.get('atk'))
            elif h == 'conditional_defense_aura' and target.data.position_type=='MELEE':
                if source.blocked_enemies and own:
                    buff(source,t,'DEF_PERCENT',t.get('bryota_t_self.def'))
                elif not source.blocked_enemies and in_cells(source,target,NEIGHBORS):
                    buff(source,t,'DEF_PERCENT',t.get('bryota_t_ally.def'))
        target.modifiers[:] = [m for m in target.modifiers if not m.source.startswith('talent-live:')]+modifiers
        resize_hp(target,modified(target.data.hp,'MAX_HP',target.modifiers,sim.current_time))


def on_deploy(sim, unit):
    for t in talents(unit,'deploy_melee_aura'):
        sim.state.queue.push(sim.current_time+t.get('duration'),'MODIFIER_BOUNDARY',unit.id)
    refresh(sim)


def on_skill(sim, unit):
    unit.talent_state['combo_hits'] = 0
    unit.talent_state['ammo_consumed'] = 0


def natural_sp_rate(unit):
    extra = sum(t.get('min_sp_recovery_per_sec')*min(1.,max(0.,(1-ratio(unit))/(1-t.get('min_hp_ratio'))))
                for t in talents(unit,'berserk'))
    return 1+extra


def on_attack(sim, unit):
    """Roll attack-level procs once, shared across targets and multi-hits."""
    procs = []
    extra_hits = 0
    for t in talents(unit):
        if t.handler in ('attack_crit','mountain_crit'):
            if chance(sim,unit,t,t.get('prob'),'ATTACK'):
                procs.append(t.prefab)
        elif t.handler == 'attack_sp_chance':
            p=t.get('prob_2') if ratio(unit)<t.get('hp_ratio') else t.get('prob_1')
            if chance(sim,unit,t,p,'ATTACK'):
                add_sp(sim,unit,t.get('sp'))
        elif t.handler == 'extra_attack':
            p=t.get('prob')+unit.talent_state.get('ammo_consumed',0)*t.get('prob_add')
            if chance(sim,unit,t,p,'ATTACK'):
                extra_hits += 1
        elif t.handler == 'consecutive_hits' and sim.current_time > unit.last_attack_end+t.get('attack@clear_attackcount_time')+1e-8:
            unit.talent_state['combo_hits']=0
    return tuple(procs),extra_hits


def apply_silence(sim, target, duration):
    if getattr(target.data,'silence_immune',False):
        sim._log('STATUS_IMMUNE',unit=target.id,status='SILENCE')
        return
    target.silence_until=max(target.silence_until,sim.current_time+duration)
    sim.state.queue.push(target.silence_until,'MODIFIER_BOUNDARY',target.id)
    sim._log('SILENCE',unit=target.id,until=target.silence_until)


def effect_enabled(unit, effect, time):
    return not (getattr(unit,'silence_until',0)>time and effect in getattr(unit.data,'silenceable_effects',()))


def apply_tremble(sim, target, duration):
    if getattr(target.data,'tremble_immune',False):
        sim._log('STATUS_IMMUNE',unit=target.id,status='TREMBLE')
        return
    target.tremble_until=max(target.tremble_until,sim.current_time+duration)
    # Tremble suppresses melee attacks while blocked, not movement or ranged fire.
    if target.blocked_by:
        target.attack_revision+=1
    sim.state.queue.push(target.tremble_until,'MODIFIER_BOUNDARY',target.id)
    sim._log('TREMBLE',unit=target.id,until=target.tremble_until)


def before_hit(sim, attacker, target, attack):
    """Damage-addition hooks precede defense calculation and evasion."""
    scale=1.
    for t in talents(attacker):
        h=t.handler
        if h in ('attack_crit','mountain_crit') and t.prefab in attack.talent_procs:
            scale*=t.get('atk_scale')
            if h=='mountain_crit':
                set_buff(sim,target,'talent-hit:'+attacker.id+':'+t.prefab,'ATK_PERCENT',t.get('atk'),t.get('duration'))
        elif h=='damage_crit_tremble' and chance(sim,attacker,t,t.get('prob'),'DAMAGE'):
            scale*=t.get('atk_scale')
            apply_tremble(sim,target,t.get('not_combat'))
        elif h=='silence' and not attack.is_dot and not attack.is_reflection:
            apply_silence(sim,target,t.get('duration'))
            if t.get('damage_scale'):
                set_buff(sim,target,'fragile:'+attacker.id,'FRAGILE_MULTIPLY',t.get('damage_scale'),t.get('duration'))
        elif h=='hp_steal_dot' and not attack.is_dot and target.id in sim.state.enemies:
            if not target.data.special_hp:
                remaining=t.get('attack@steal_hp_max')-attacker.talent_state.get('hp_stolen',0)
                amount=min(t.get('attack@steal_hp'),remaining,max_hp(target)-1)
                if amount>0:
                    attacker.talent_state['hp_stolen']=attacker.talent_state.get('hp_stolen',0)+amount
                    resize_hp(target,max_hp(target)-amount)
                    refresh(sim)
                    sim._log('MAX_HP_STOLEN',unit=attacker.id,target=target.id,amount=amount)
            source=attacker.id+':'+t.prefab
            generation=target.talent_state.get('dot_generation:'+source,0)+1
            target.talent_state['dot_generation:'+source]=generation
            target.talent_state['dot_until:'+source]=sim.current_time+t.get('dot_duration')
            # Refresh retains the tick cadence of the existing DoT.
            if not target.talent_state.get('dot_pending:'+source):
                target.talent_state['dot_pending:'+source]=True
                sim.state.queue.push(sim.current_time+t.get('interval'),'TALENT_EVENT','DOT',
                                     attacker.id,attacker.generation,target.id,t.prefab)
    return scale


def offensive(sim, unit, target, attack, atk, defense, resistance):
    for t in talents(unit):
        h=t.handler
        if h=='arts_penetration':resistance-=t.get('magic_resist_penetrate_fixed')
        elif h=='hp_def_penetration':defense-=t.get('def_penetrate_fixed')
        elif h=='tremble_penetration' and target.tremble_until>sim.current_time:
            defense*=1-t.get('def_penetrate')
        elif h=='drone_priority' and getattr(target.data,'motion','WALK')=='FLY':atk*=t.get('atk_scale',1)
        elif h=='nearby_enemies_scale' and not attack.is_reflection:
            atk*=t.get('atk_scale_up') if nearby_count(sim,unit)>=t.get('cnt') else t.get('atk_scale_base')
        elif h=='module_blocked_attack' and target.blocked_by:atk*=t.get('atk_scale')
    return atk,max(0,defense),max(0,resistance)


def outgoing_scale(unit, target, attack):
    scale=1.
    for t in talents(unit):
        if t.handler=='unblocked_damage' and target.blocked_by!=unit.id:
            scale*=t.get('damage_scale')
        elif t.handler=='module_skill_damage' and attack.skill_effect:
            scale*=t.get('damage_scale')
    return scale


def healing_scale(unit):
    scale=1.
    for t in talents(unit):
        if t.handler=='healing_received':
            scale*=t.get('heal_scale_2') if ratio(unit)<t.get('hp_ratio') else t.get('heal_scale_1')
        elif t.handler=='module_healing':scale*=t.get('heal_scale')
    return scale


def pre_receive(sim, unit):
    from arknights_sim.skills.runtime import heal_self
    for t in talents(unit,'pre_damage_heal'):
        amount=t.get('value2') if ratio(unit)<t.get('hp_ratio') else t.get('value1')
        heal_self(sim,unit,amount,as_healing=True)


def dodge(sim, unit, kind):
    if kind!='PHYSICAL':return False
    from arknights_sim.skills.runtime import active_effect
    effect=active_effect(unit,sim.current_time)
    # Independent dodge sources combine multiplicatively in failure space.
    avoided=False
    for t in talents(unit,'defense_dodge'):
        avoided=chance(sim,unit,t,t.get('prob'),'PHYSICAL_DODGE') or avoided
    if effect and effect.physical_dodge:
        avoided=sim.state.rng.random()<effect.physical_dodge or avoided
    return avoided


def incoming_scale(sim, unit, source, kind):
    scale=1.
    shelter=0.
    for t in talents(unit):
        h=t.handler
        if h=='emergency_heal' and unit.talent_state.get('emergency_used') and kind=='PHYSICAL':
            scale*=1-t.get('damage_resistance')
        elif h=='other_enemy_protection' and kind in ('PHYSICAL','ARTS') and unit.blocked_enemies and source is not None and source.blocked_by!=unit.id:
            scale*=1-t.get('damage_resistance')
        elif h=='nearby_enemies_scale' and nearby_count(sim,unit)>=t.get('cnt'):
            scale*=1-t.get('damage_resistance')
        elif h=='low_hp_shelter' and ratio(unit)<t.get('hp_ratio') and kind in ('PHYSICAL','ARTS'):
            shelter=max(shelter,t.get('damage_resistance'))
    return scale*(1-shelter)


def hp_floor(sim, unit, source, amount):
    if source is not None and talents(source,'dying') and unit.id in sim.state.enemies:
        return 1.
    for t in talents(unit):
        if t.handler=='emergency_heal' and not unit.talent_state.get('emergency_used'):
            return 1.
        if t.handler=='lethal_survival':
            until=unit.talent_state.get('survival_until')
            if until is None and amount>=unit.hp:
                until=sim.current_time+t.get('surtr_t_2[withdraw].interval')
                unit.talent_state['survival_until']=until
                sim.state.queue.push(until,'TALENT_EVENT','SURVIVAL_END',unit.id,unit.generation)
                sim._log('LETHAL_SURVIVAL',unit=unit.id,until=until)
            if until is not None and sim.current_time<until:return 1.
    return 0.


def after_receive(sim, unit, source, damage, *, reflect=True):
    from arknights_sim.skills.runtime import heal_self
    if not unit.alive:return
    for t in talents(unit):
        if t.handler=='emergency_heal' and not unit.talent_state.get('emergency_used') and ratio(unit)<t.get('hp_ratio'):
            unit.talent_state['emergency_used']=True
            heal_self(sim,unit,max_hp(unit)*t.get('etlchi_t_2[heal].hp_ratio'),as_healing=True)
            sim._log('EMERGENCY_HEAL',unit=unit.id)
        elif t.handler=='hurt_attack_buff' and damage>0:
            set_buff(sim,unit,'talent-hit:'+unit.id+':'+t.prefab,'ATK_PERCENT',t.get('atk'),t.get('add_atk_duration'))
    if reflect and source is not None and source.id in sim.state.enemies and source.data.faction=='HOSTILE' and 'kazimierz' in getattr(unit.data,'factions',()):
        from arknights_sim.combat.runtime import deal_damage, attack_power
        for caster in sorted(sim.state.allies.values(),key=lambda u:u.id):
            if caster.alive:
                for t in talents(caster,'kazimierz_reflection'):
                    deal_damage(sim,source,attack_power(sim,caster)*t.get('atk_scale'),'TRUE',caster,
                                event_kind='REFLECT_DAMAGE',reflect=False)


def after_hit(sim, attacker, target, attack, damage):
    if damage<=0:return
    from arknights_sim.skills.runtime import heal_self
    if attacker.id in sim.state.allies:
        # Musha/reaper traits recover HP, distinct from external healing.
        branch=attacker.data.subprofession
        if branch in ('musha','reaper') and not attack.is_dot:
            if branch=='musha' or attack.target_index<attacker.data.block_count:
                heal_self(sim,attacker,(30,50,70)[attacker.data.elite])
        for t in talents(attacker):
            if t.handler=='consecutive_hits' and not active(attacker,sim.current_time):
                count=attacker.talent_state.get('combo_hits',0)+1
                if count>=t.get('hit_num'):
                    count=0
                    attacker.talent_state['combo_stacks']=min(3,attacker.talent_state.get('combo_stacks',0)+1)
                attacker.talent_state['combo_hits']=count
            if t.handler=='dying' and target.alive and target.hp<=1 and not target.dying_until:
                target.dying_until=sim.current_time+t.get('interval')
                target.talent_state['dying_sp']=t.get('sp')
                set_buff(sim,target,'dying','MOVE_SPEED_MULTIPLY',1+t.get('move_speed'),t.get('interval'))
                sim.state.queue.push(target.dying_until,'TALENT_EVENT','DYING_END',target.id,target.generation)
                sim._log('DYING',unit=target.id,until=target.dying_until)


def on_death(sim, unit, killer=None):
    if unit.data.faction!='HOSTILE':return
    from arknights_sim.skills.runtime import heal_self
    for op in sorted(sim.state.allies.values(),key=lambda u:u.id):
        if not op.alive:continue
        for t in talents(op):
            if t.handler=='nearby_death_heal' and in_cells(op,unit,t.attack_range):
                heal_self(sim,op,max_hp(op)*t.get('hp_ratio'))
            if op is killer and t.handler in ('kill_attack_speed','abyssal_kill_stats'):
                key='kills:'+t.prefab
                op.talent_state[key]=min(t.get('max_stack_cnt'),op.talent_state.get(key,0)+1)
                sim._log('TALENT_STACK',unit=op.id,talent=t.name,stacks=op.talent_state[key])
    if killer and unit.dying_until:
        add_sp(sim,killer,unit.talent_state.get('dying_sp',0))
    refresh(sim)


def event(sim, payload):
    kind,key,generation,*rest=payload
    unit=sim.state.allies.get(key) or sim.state.enemies.get(key)
    if not unit or unit.generation!=generation:return
    if kind=='SURVIVAL_END':
        if unit.alive:
            sim._remove_operator(unit)
            refresh(sim)
            sim._log('FORCED_WITHDRAW',unit=key,reason='LETHAL_SURVIVAL')
    elif kind=='DYING_END':
        if unit.alive and sim.current_time>=unit.dying_until:
            sim._remove_enemy(unit)
            sim._log('DEATH',unit=unit.id,reason='DYING')
    elif kind=='DOT':
        target_id,prefab=rest
        target=sim.state.enemies.get(target_id)
        if not target or not target.alive:return
        source=key+':'+prefab
        target.talent_state['dot_pending:'+source]=False
        if sim.current_time>target.talent_state.get('dot_until:'+source,0)+1e-9:return
        t=next(t for t in talents(unit,'hp_steal_dot') if t.prefab==prefab)
        from arknights_sim.combat.attack import AttackEvent
        from arknights_sim.combat.runtime import resolve_hit
        a=AttackEvent(key,target_id,sim.current_time,sim.current_time,sim.current_time,
                      generation,target.generation,sp_eligible=False,damage_type='ARTS',
                      fixed_atk=t.get('magic_value'),is_dot=True)
        resolve_hit(sim,a)
        if target.alive:
            target.talent_state['dot_pending:'+source]=True
            sim.state.queue.push(sim.current_time+t.get('interval'),'TALENT_EVENT',*payload)
    else:
        raise ValueError('Unknown talent event: '+kind)
