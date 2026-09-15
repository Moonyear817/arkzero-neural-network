"""Auditable v1/v2 warm-start migration, including meta/assign UI loading."""
import torch


def migrate_weights(module,state):
    target=module.state_dict()
    result=dict(state)
    legacy=not any(k.startswith('future_events.') for k in state)
    loaded=[];partial=[];new=[];skipped=[]
    template=None
    for key,current in target.items():
        if key in result and result[key].shape==current.shape:
            loaded.append(key);continue
        old=result.get(key)
        if old is not None:
            expandable=(key=='enemy_encoder.0.weight' and old.shape[1]==32 and current.shape[1]==40) or (key=='action_encoder.0.weight' and old.shape[1]==80 and current.shape[1]==96)
            if expandable and legacy and old.shape[0]==current.shape[0]:
                value=torch.zeros(current.shape,dtype=old.dtype,device=old.device)
                value[:,:old.shape[1]]=old;result[key]=value;partial.append(key);continue
            raise ValueError(f'Checkpoint tensor shape mismatch: {key}: {tuple(old.shape)} -> {tuple(current.shape)}')
        if not legacy:
            # Current-version missing tensors remain errors even with strict=False.
            raise ValueError(f'Current checkpoint is incomplete: missing {key}')
        allowed=key.startswith(('mechanics_','enemy_event_encoder.','future_events.','progression_encoder.','summon_encoder.','context_projection.','context_gate.','context_norm.','hierarchical_policy.','event_context.','residual_blocks.2.','residual_blocks.3.'))
        if not allowed:raise ValueError(f'Legacy checkpoint is missing required tensor {key}')
        if current.is_meta:
            if template is None:
                # The desktop inspects under meta and assign=True. Construct only
                # missing weights locally, restoring the caller's RNG exactly.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(20260915)
                    with torch.device('cpu'):
                        template=type(module)().state_dict()
            value=template[key].clone()
        else:value=current.detach().clone()
        if key.startswith('mechanics_'):value.zero_()
        result[key]=value;new.append(key)
    unexpected=set(result)-set(target)
    if unexpected:raise ValueError(f'Unknown checkpoint tensors: {sorted(unexpected)}')
    partial_loaded=sum(state[k].numel() for k in partial)
    partial_new=sum(target[k].numel()-state[k].numel() for k in partial)
    module.migration_report={
        'legacy_migration':legacy,'loaded_tensors':loaded,'partial_tensors':partial,
        'skipped_tensors':skipped,'new_tensors':new,
        'loaded_parameters':sum(target[k].numel() for k in loaded)+partial_loaded,
        'skipped_parameters':0,'new_parameters':sum(target[k].numel() for k in new)+partial_new,
    }
    report=module.migration_report
    print(f'Checkpoint {type(module).__name__}: Loaded parameters={report["loaded_parameters"]:,}; Skipped parameters=0; New parameters={report["new_parameters"]:,}; partial tensors={len(partial)}')
    return result
