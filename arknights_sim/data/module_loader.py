"""Read equipped-module stats and resolved talent candidates from pinned data.

Runtime effect support is separate: loading a module is not certification that
all its abilities are implemented. Mode-restricted modules fail closed here.
"""
from .progression import unlocked


def candidate(candidates, elite, level, potential):
    eligible=[c for c in candidates or [] if unlocked(c['unlockCondition'],elite,level)
              and c['requiredPotentialRank']<=potential]
    return max(eligible,key=lambda c:(int(c['unlockCondition']['phase'][-1]),c['unlockCondition']['level'],c['requiredPotentialRank']),default=None)


def module_data(table, key, level, elite, op_level, potential):
    if not key:
        if level:raise ValueError('Module level without module')
        return {},(),()
    if key not in table:raise ValueError('Missing module battle data')
    phase=next((p for p in table[key]['phases'] if p['equipLevel']==level),None)
    if phase is None:raise ValueError('Unknown module level')
    if phase.get('tokenAttributeBlackboard'):raise NotImplementedError('Module token stat overrides')
    attributes={v['key']:v['value'] for v in phase['attributeBlackboard']}
    if set(attributes)-{'max_hp','atk','def','magic_resistance','attack_speed','cost','respawn_time'}:
        raise NotImplementedError('Unsupported module attribute')
    traits=[];talents=[]
    for part in phase['parts']:
        if part.get('isToken') or part.get('validInGameTag') or part.get('validInMapTag'):
            raise NotImplementedError('Mode-restricted or token module effect')
        trait=candidate(part['overrideTraitDataBundle']['candidates'],elite,op_level,potential)
        talent=candidate(part['addOrOverrideTalentDataBundle']['candidates'],elite,op_level,potential)
        if trait:traits.append(trait)
        if talent:talents.append(talent)
    return attributes,tuple(traits),tuple(talents)
