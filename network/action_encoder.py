"""Dynamic action features preserve the exact legal-action ordering supplied."""

import torch

from arknights_sim.environment.action import ActionType

from .state_encoder import OPERATOR_FEATURES, map_features, operator_features

ACTION_FEATURES = 103
LEGACY_ACTION_FEATURES = 96
HIERARCHY_START = 96


class ActionEncoder:
    FEATURE_DIM = ACTION_FEATURES

    def encode(self, state, ordered_actions):
        game, stage = state.game, state.game.stage
        actions = tuple(ordered_actions)
        result = torch.zeros(len(actions), ACTION_FEATURES)
        keys = tuple(sorted(set(game.squad) | set(game.summon_cards) | set(game.summons)))
        op_vectors = {key: operator_features(game, key) for key in keys}
        image = map_features(game) if any(a.operator_id for a in actions) else None
        kinds = tuple(ActionType)
        from .mechanics_encoder import mechanics_features
        mechanic_image=mechanics_features(game)
        entity_keys=tuple(sorted(set(keys)|set(game.devices)|{d.id for d in stage.devices}))
        for i, action in enumerate(actions):
            kind=kinds.index(action.type)
            wait=getattr(action,'wait_seconds',None)
            wait_index=(None,.5,1.,2.,5.).index(wait)
            result[i,96:103]=torch.tensor([kind,entity_keys.index(action.operator_id)+1 if action.operator_id in entity_keys else 0,
                *(action.tile or (-1,-1)),action.direction.index if action.direction is not None else -1,wait_index,(wait or 0)/5])
            if action.type in (ActionType.PLACE_DEVICE,ActionType.ACTIVATE_DEVICE,ActionType.REMOVE_DEVICE):
                result[i,80+(kinds.index(action.type)-4)]=1
                data=next((d for d in stage.devices if d.id==action.operator_id),None)
                device=game.devices.get(action.operator_id)
                if device:data=device.data
                tile=action.tile or (device.position if device else None)
                if tile:
                    x,y=tile;result[i,52:54]=torch.tensor((x/max(1,stage.map.width-1),y/max(1,stage.map.height-1)))
                    result[i,58:78]=image[:,y,x]
                if data:
                    result[i,83]=data.cost/100;result[i,84]=game.device_inventory.get(data.id,0)/10
                    result[i,85]=device.sp/max(1,data.charge) if device else 0
                    result[i,86]=max(0,device.active_until-game.current_time)/30 if device else 0
                continue
            base_kind=0 if action.type==ActionType.DEPLOY_SUMMON else (2 if action.type==ActionType.RETREAT_SUMMON else kinds.index(action.type))
            result[i, base_kind] = 1
            if action.operator_id is None:
                continue
            key = action.operator_id
            if key not in op_vectors:
                raise ValueError(f"Action operator not in squad: {key}")
            result[i, 4 : 4 + OPERATOR_FEATURES] = torch.tensor(op_vectors[key])
            unit = game.allies.get(key)
            tile = (
                action.tile
                if action.tile is not None
                else (unit.position if unit and unit.alive else None)
            )
            direction = (
                action.direction.index
                if action.direction is not None
                else (unit.direction if unit and unit.alive else None)
            )
            if tile is not None:
                x, y = tile
                if (
                    x != int(x)
                    or y != int(y)
                    or not (0 <= x < stage.map.width and 0 <= y < stage.map.height)
                ):
                    raise ValueError("Action tile is outside the map")
                x, y = int(x), int(y)
                result[i, 52:54] = torch.tensor(
                    (x / max(1, stage.map.width - 1), y / max(1, stage.map.height - 1))
                )
                result[i, 58:78] = image[:, y, x]
                result[i,87:96]=mechanic_image[[0,1,3,6,7,8,9,10,15],y,x]
                data=game.squad[key] if key in game.squad else (game.summon_cards[key].unit if key in game.summon_cards else game.summons[key].data)
                offsets = data.attack_range
                rotated = []
                for dx, dy in offsets:
                    for _ in range(direction or 0):
                        dx, dy = -dy, dx
                    rotated.append((x + dx, y + dy))
                result[i, 79] = sum(
                    0 <= px < stage.map.width and 0 <= py < stage.map.height
                    for px, py in rotated
                ) / max(1, len(offsets))
            if direction is not None:
                result[i, 54 + direction] = 1
            result[i, 78] = (keys.index(key) + 1) / max(1, len(keys))
        return result

    __call__ = encode
