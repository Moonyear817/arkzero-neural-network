"""Small neural policy/value model with dynamic action scoring and strict masks."""

import torch
from torch import nn

from .action_encoder import ACTION_FEATURES
from .future_event_encoder import FutureEventEncoder, EVENT_FEATURES, PROGRESSION_FEATURES
from .hierarchical_policy import HierarchicalPolicy
from .state_encoder import (
    ENEMY_FEATURES,
    FUTURE_FEATURES,
    FUTURE_WINDOWS,
    GLOBAL_FEATURES,
    MAP_CHANNELS,
    OPERATOR_FEATURES,
)


def _masked_pool(values, mask, dim):
    mask = mask.bool()
    expanded = mask.unsqueeze(1) if dim == (2, 3) else mask.unsqueeze(-1)
    values = values.masked_fill(~expanded, 0)
    count = expanded.sum(dim=dim).clamp_min(1)
    mean = values.sum(dim=dim) / count
    maximum = values.masked_fill(~expanded, -torch.inf).amax(dim=dim)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=-1)


class _ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x, mask):
        # Mask at every convolution so activations in the padded region cannot
        # leak back into the real map on the next convolution. No spatial norm.
        y = self.activation(self.conv1(x)).masked_fill(~mask, 0)
        y = self.conv2(y).masked_fill(~mask, 0)
        return self.activation(x + y).masked_fill(~mask, 0)


class PolicyValueNetwork(nn.Module):
    """Spatial and dependency-aware event policy with conditional legal actions.

    Single input: logits[A], scalar value. Batched input: logits[B,A], value[B].
    Padded actions receive -inf logits; valid rows must contain at least one
    action. Consumers computing cross entropy must mask 0 * -inf terms.
    """

    def __init__(self, future_events_enabled=True):
        super().__init__()
        self.future_events_enabled=bool(future_events_enabled)
        channels, hidden_dim = 96, 256
        self.architecture_config = {
            "version": 3,
            "observation_version": 3,
            "event_graph_version": 1,
            "future_events_enabled": self.future_events_enabled,
            "residual_blocks": 4,
            "event_features": EVENT_FEATURES,
            "progression_features": PROGRESSION_FEATURES,
            "channels": channels,
            "hidden_dim": hidden_dim,
            "map_channels": MAP_CHANNELS,
            "operator_features": OPERATOR_FEATURES,
            "enemy_features": ENEMY_FEATURES,
            "global_features": GLOBAL_FEATURES,
            "future_features": FUTURE_FEATURES,
            "future_windows": list(FUTURE_WINDOWS),
            "action_features": ACTION_FEATURES,
        }
        self.map_input = nn.Conv2d(MAP_CHANNELS, channels, 3, padding=1)
        self.mechanics_input = nn.Conv2d(16, channels, 3, padding=1,bias=False)
        self.residual_blocks = nn.ModuleList(
            [_ResidualBlock(channels) for _ in range(4)]
        )
        self.operator_encoder = nn.Sequential(
            nn.Linear(OPERATOR_FEATURES, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.enemy_encoder = nn.Sequential(
            nn.Linear(ENEMY_FEATURES, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.global_encoder = nn.Sequential(nn.Linear(GLOBAL_FEATURES, 64), nn.ReLU())
        self.future_encoder = nn.Sequential(
            nn.Linear(FUTURE_FEATURES * len(FUTURE_WINDOWS), 64), nn.ReLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(channels * 2 + 256 * 2 + 64 * 2, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(96, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
        )
        self.policy_query = nn.Linear(hidden_dim, hidden_dim)
        self.action_bias = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh()
        )
        self.enemy_event_encoder=nn.Linear(5,128,bias=False)
        self.future_events=FutureEventEncoder()
        self.progression_encoder=nn.Sequential(nn.Linear(PROGRESSION_FEATURES,128),nn.ReLU())
        self.summon_encoder=nn.Sequential(nn.Linear(56,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU())
        self.context_projection=nn.Sequential(nn.Linear(256+128+256,256),nn.ReLU(),nn.Linear(256,256))
        self.context_gate=nn.Linear(512,256)
        self.context_norm=nn.LayerNorm(256)
        self.hierarchical_policy=HierarchicalPolicy()
        for block in self.residual_blocks[2:]:
            nn.init.zeros_(block.conv2.weight);nn.init.zeros_(block.conv2.bias)
        # Unknown outcomes start at neutral value; avoid saturating tanh before
        # any examples have been observed. Checkpoint loading replaces this.
        nn.init.zeros_(self.value_head[-2].weight)
        nn.init.zeros_(self.value_head[-2].bias)

    def load_state_dict(self,state_dict,strict=True,assign=False):
        from .checkpoint_compat import migrate_weights
        return super().load_state_dict(migrate_weights(self,state_dict),strict=strict,assign=assign)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(self, state, action_features, action_mask=None):
        logits,value,_=self.forward_with_embedding(state,action_features,action_mask)
        return logits,value

    def forward_with_embedding(self,state,action_features,action_mask=None):
        single = state["map"].ndim == 3
        if single:
            state = {k: v.unsqueeze(0) for k, v in state.items()}
            action_features = action_features.unsqueeze(0)
            if action_mask is not None:
                action_mask = action_mask.unsqueeze(0)
        if state["map"].ndim != 4 or action_features.ndim != 3:
            raise ValueError("Expected single or batched map/action tensors")
        if (
            action_features.shape[0] != state["map"].shape[0]
            or action_features.shape[-1] != ACTION_FEATURES
        ):
            raise ValueError("Action/state batch or feature dimensions do not match")
        if action_features.shape[1] == 0:
            raise ValueError("Network requires at least one legal action per state")
        if action_mask is None:
            action_mask = torch.ones(
                action_features.shape[:2],
                dtype=torch.bool,
                device=action_features.device,
            )
        if action_mask.shape != action_features.shape[:2]:
            raise ValueError("Action mask shape does not match actions")
        action_mask = action_mask.bool()
        if not bool(action_mask.any(dim=-1).all()):
            raise ValueError("Each state requires at least one valid action")
        fused = self.encode_state(state)
        action_input = action_features.masked_fill(~action_mask.unsqueeze(-1), 0)
        action_input[...,78]=0  # Group identity is metadata, not a learned unit vocabulary.
        actions = self.action_encoder(action_input[...,:96])
        logits = (actions * self.policy_query(fused).unsqueeze(1)).sum(-1) / 16
        logits = logits + self.action_bias(actions).squeeze(-1)
        logits = self.hierarchical_policy(fused,action_features,action_mask,logits)
        value = self.value_head(fused).squeeze(-1)
        return (logits[0], value[0], fused[0]) if single else (logits, value, fused)

    def encode_state(self,state):
        single=state['map'].ndim==3
        if single:
            return self.encode_state({k:v.unsqueeze(0) for k,v in state.items()})[0]
        map_mask = state["map_mask"].bool().unsqueeze(1)
        image = state["map"].masked_fill(~map_mask, 0)
        spatial = self.map_input(image)
        if 'mechanics' in state:spatial=spatial+self.mechanics_input(state['mechanics'].masked_fill(~map_mask,0))
        spatial = torch.relu(spatial).masked_fill(~map_mask, 0)
        for block in self.residual_blocks:
            spatial = block(spatial, map_mask)
        op_input = state["operators"].masked_fill(
            ~state["operator_mask"].bool().unsqueeze(-1), 0
        )
        enemy_input = state["enemies"].masked_fill(
            ~state["enemy_mask"].bool().unsqueeze(-1), 0
        )
        features = torch.cat(
            (
                _masked_pool(spatial, state["map_mask"], (2, 3)),
                _masked_pool(
                    self.operator_encoder(op_input), state["operator_mask"], 1
                ),
                _masked_pool(self.enemy_encoder(enemy_input)+(self.enemy_event_encoder(state["enemy_events"].masked_fill(~state["enemy_mask"][...,None],0)) if "enemy_events" in state and self.future_events_enabled else 0), state["enemy_mask"], 1),
                self.global_encoder(state["global"]),
                self.future_encoder(state["future"].flatten(start_dim=1)),
            ),
            dim=-1,
        )
        fused = self.fusion(features)
        batch=state['map'].shape[0]
        if 'summons' in state:
            clean=state['summons'].masked_fill(~state['summon_mask'][...,None],0)
            summons=_masked_pool(self.summon_encoder(clean),state['summon_mask'],1)
        else:summons=fused.new_zeros(batch,256)
        if self.future_events_enabled and 'events' in state:
            events=self.future_events(state['events'],state['event_mask'],state['event_relations'])
            progression=self.progression_encoder(state['progression'])
        else:
            events=fused.new_zeros(batch,256);progression=fused.new_zeros(batch,128)
        context=self.context_projection(torch.cat((events,progression,summons),-1))
        gate=torch.sigmoid(self.context_gate(torch.cat((fused,context),-1)))
        return self.context_norm(fused+gate*context)

    state_embedding = encode_state
