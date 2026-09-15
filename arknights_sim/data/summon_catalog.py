"""Summon source relationships; catalog data never implies battle support.

Token records are separate from roster operators. Null skillId slots often mean
owner-driven effects, not a missing ordinary operator skill to invent.
"""
from copy import deepcopy
from .summon_loader import SUPPORTED_SUMMON_PAIRS


def summon_records(loader):
    rows=[]
    for key in loader.available_ids():
        owner=loader.chars[key]
        candidates=[c for talent in owner.get('talents') or [] for c in talent.get('candidates') or []]
        tokens=set(owner.get('displayTokenDict') or {})
        tokens.update(c['tokenKey'] for c in candidates if c.get('tokenKey'))
        tokens.update(s['overrideTokenKey'] for s in owner['skills'] if s.get('overrideTokenKey'))
        for token_key in sorted(tokens):
            raw=loader.chars.get(token_key)
            rows.append({
                'owner_id':key,'owner_name':loader.catalog.get(key,{}).get('name',owner['name']),
                'token_id':token_key,'token_name':raw['name'] if raw else 'UNKNOWN',
                'status':'PARTIALLY_IMPLEMENTED' if (key,token_key) in SUPPORTED_SUMMON_PAIRS else 'UNSUPPORTED_RUNTIME',
                'runtime_scope': 'No-module tentacle placement/combat and owner S1; lifecycle ASSUMED; S2 unsupported' if (key,token_key) in SUPPORTED_SUMMON_PAIRS else 'No battle handler',
                'owner_trait':owner.get('description'),
                'owner_talents':deepcopy(candidates),
                'owner_skill_bindings':[{'skill_index':i,'skill_id':s['skillId'],'override_token':s.get('overrideTokenKey'),'unlock':deepcopy(s['unlockCond'])} for i,s in enumerate(owner['skills'])],
                'position_type':raw.get('position') if raw else 'UNKNOWN',
                'phases':deepcopy(raw.get('phases')) if raw else None,
                'token_skills':deepcopy(raw.get('skills')) if raw else None,
                'token_talents':deepcopy(raw.get('talents')) if raw else None,
                'runtime_requirements':[
                    'Owner identity and independent token instance IDs',
                    'Inventory, simultaneous limits, deployment-slot weight, per-token cooldown',
                    'Legal tile, owner-range restrictions and owner-alive conditions',
                    'Token attacks, blocking, targeting, healing eligibility and death',
                    'Owner skill propagation, token skills, recall/refund and replacement',
                    'Special fusion/copy/respawn rules and module changes where applicable',
                    'Clone/replay/RNG state and neural observation/action representation',
                ],
            })
    return tuple(rows)


def resolve_summon_sources(loader, owner_id, progression):
    """Select source variants at effective owner elite/level/potential/skill.

    Returns source facts only. Deployment weight, refunds and restrictions must
    come from a verified runtime handler, never inferred from the display table.
    """
    from .progression import resolve_progression, unlocked
    owner_id=loader.resolve(owner_id)
    owner=loader.chars[owner_id]
    effective, adjustments=resolve_progression(owner,progression)
    selected=[]
    for talent in owner.get('talents') or []:
        eligible=[c for c in talent.get('candidates') or []
                  if unlocked(c['unlockCondition'],effective.elite,effective.level)
                  and c.get('requiredPotentialRank',0)<=effective.potential]
        if eligible:
            selected.append(max(eligible,key=lambda c:(int(c['unlockCondition']['phase'].split('_')[-1]),c['unlockCondition']['level'],c.get('requiredPotentialRank',0))))
    skill=owner['skills'][effective.skill_index] if owner['skills'] else {}
    override=skill.get('overrideTokenKey')
    token_keys={override} if override else {c['tokenKey'] for c in selected if c.get('tokenKey')}
    records={r['token_id']:r for r in summon_records(loader) if r['owner_id']==owner_id}
    return {
        'owner_id':owner_id,'progression':effective.to_dict(),'adjustments':adjustments,
        'selected_skill_id':skill.get('skillId'),'selected_talents':deepcopy(selected),
        'selected_tokens':[deepcopy(records[key]) for key in sorted(token_keys) if key in records],
        'source_inventory_counts':[entry['value'] for c in selected for entry in c.get('blackboard') or [] if entry['key']=='cnt'],
        'runtime_status':'PARTIALLY_IMPLEMENTED' if token_keys and all((owner_id,k) in SUPPORTED_SUMMON_PAIRS for k in token_keys) else 'UNSUPPORTED_RUNTIME',
    }
