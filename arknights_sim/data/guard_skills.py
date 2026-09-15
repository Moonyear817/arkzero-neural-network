"""Explicitly reviewed skill recipes; source text is not executable behavior.

A recipe is accepted only with the expected blackboard keys. Character traits
and talents have independent coverage and are not certified by this registry.
"""
from dataclasses import replace
from .models import SkillData

NEXT_SINGLE = frozenset('huang doberm svrash savage jaksel lessng bryota'.split())
NEXT_DOUBLE = frozenset('crow highmo etlchi'.split())
RECOVERY = {'INCREASE_WITH_TIME':'AUTO_RECOVERY', 'INCREASE_WHEN_ATTACK':'ATTACK_RECOVERY',
            'INCREASE_WHEN_TAKEN_DAMAGE':'DEFENSE_RECOVERY', 8:'NONE'}


def load_guard_skill(key, row, ranges):
    bb={v['key']:v['value'] for v in row['blackboard']}
    stem=key.removeprefix('skchr_').removesuffix('_1')
    mode='TIMED'; extra={}; allowed=set()
    if key.startswith('skchr_') and key.endswith('_1') and stem in NEXT_SINGLE | NEXT_DOUBLE | {'varkis','flameb','hodrer','surtr','humus','frostl','frncat','f12yin','ayer'}:
        mode='NEXT_ATTACK'; allowed={'atk_scale'}
        extra['attack_scale']=bb.get('atk_scale',1)
        if stem in NEXT_DOUBLE: extra['hits']=2
        if stem=='varkis': extra['hits']=3
        if stem in {'flameb','hodrer'}:
            allowed.add('hp_ratio');extra['heal_ratio']=bb['hp_ratio']
        if stem=='surtr': extra['kill_refill']=True
        if stem=='humus':
            allowed.add('value');extra['heal_flat']=bb['value']
        if stem=='f12yin':
            allowed.add('max_target');extra['max_targets']=int(bb['max_target'])
        if stem=='frostl':
            allowed|={'move_speed','duration'}
            extra.update(target_speed_multiplier=1+bb['move_speed'],debuff_duration=bb['duration'])
        if stem=='frncat':
            allowed={'atk','frncat_s_1[debuff].atk','frncat_s_1[debuff].duration'}
            extra.update(attack_scale=1+bb['atk'],target_atk_multiplier=1+bb['frncat_s_1[debuff].atk'],debuff_duration=bb['frncat_s_1[debuff].duration'])
        if stem=='ayer':
            allowed|={'max_target','sluggish'}
            extra.update(max_targets=int(bb['max_target']),damage_type='ARTS',target_speed_multiplier=.2,debuff_duration=bb['sluggish'])
    elif key in {'skcom_heal_self[2]','skcom_heal_self[3]'}:
        mode='INSTANT_HEAL';allowed={'heal_scale'};extra['heal_ratio']=bb['heal_scale']
    elif key in {'skchr_midn_1','skchr_broca_1','skchr_whitew_2'}:
        allowed={'atk'};extra['damage_type']='ARTS'
        if key=='skchr_whitew_2':extra.update(max_targets=2,ranged_penalty_removed=True)
    elif key=='skchr_gvial2_1':
        allowed={'atk','heal_scale'};extra['lifesteal']=bb['heal_scale']
    elif key=='skchr_estell_2':
        allowed={'atk','block_cnt'};extra['prohibit_healing']=True
        if bb['block_cnt'] != 0:raise NotImplementedError('Unreviewed Estelle block modifier')
    elif key=='skchr_astesi_2':
        allowed={'atk','def','block_cnt'};extra.update(block_delta=int(bb['block_cnt']),max_targets=-1)
    elif key=='skchr_utage_1':
        allowed={'def','hp_recovery_per_sec_by_max_hp_ratio'}
        extra.update(stop_attack=True,block_override=0,regen_ratio=bb['hp_recovery_per_sec_by_max_hp_ratio'])
    elif key=='skchr_flint_2':
        allowed={'atk','attack_speed','attack@sluggish'}
        extra.update(block_override=0,target_speed_multiplier=.2,debuff_duration=bb['attack@sluggish'])
    elif key in {'skchr_spikes_1','skchr_akafyu_1'}:
        allowed={'attack@atk_scale'} if key=='skchr_spikes_1' else {'atk'}
        extra.update(block_override=0,hits=2,attack_scale=bb.get('attack@atk_scale',1))
    elif key=='skchr_brownb_1':
        mode='PASSIVE';allowed={'prob'};extra['physical_dodge']=bb['prob']
    elif key=='skchr_f12yin_2':
        mode='TOGGLE';allowed={'atk','def','block_cnt','hp_recovery_per_sec_by_max_hp_ratio'}
        extra.update(block_delta=int(bb['block_cnt']),max_targets=-1,regen_ratio=bb['hp_recovery_per_sec_by_max_hp_ratio'])
    elif key=='skchr_lolxh_1':
        mode='TOGGLE';allowed={'prob','attack_speed'}
        extra.update(physical_dodge=bb['prob'],max_targets=-1)
    elif key=='skchr_whitew_1':
        mode='INFINITE';allowed={'atk','prob'};extra['physical_dodge']=bb['prob']
    else:
        return None
    if set(bb) != allowed:
        raise NotImplementedError(f'Unreviewed blackboard for {key}: {set(bb) ^ allowed}')
    sp=row['spData']
    recovery=RECOVERY.get(sp['spType'])
    if recovery is None or sp.get('increment',1) not in (0,1):
        raise NotImplementedError('Unreviewed skill recovery')
    if sp['maxChargeTime'] > 1:
        raise NotImplementedError('Charged skill needs a dedicated handler')
    trigger={'MANUAL':'MANUAL_TRIGGER','AUTO':'AUTO_TRIGGER','PASSIVE':'PASSIVE'}.get(row['skillType'])
    if trigger is None or (mode=='TIMED' and row['duration']<=0):
        raise NotImplementedError('Unreviewed skill duration/trigger')
    override=None
    if row.get('rangeId'):
        if row['rangeId'] not in ranges:raise NotImplementedError('Missing skill range')
        override=tuple((v['col'],v['row']) for v in ranges[row['rangeId']]['grids'])
    timed=mode in ('TIMED','TOGGLE','INFINITE')
    return SkillData(key,sp['spCost'],sp['initSp'],max(0,row['duration']),
        attack_multiplier=1+bb.get('atk',0) if timed else 1,
        defense_multiplier=1+bb.get('def',0) if timed else 1,
        attack_speed_multiplier=1+bb.get('attack_speed',0)/100 if timed else 1,
        recovery=recovery,trigger=trigger,mode=mode,range_override=override,**extra)
