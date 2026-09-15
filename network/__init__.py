"""Neural representation and inference. The simulator never imports this module."""

from .action_encoder import ACTION_FEATURES, ActionEncoder
from .inference import NeuralEvaluator
from .future_event_encoder import FutureEventEncoder, EVENT_FEATURES, PROGRESSION_FEATURES, EVENT_GRAPH_VERSION

OBSERVATION_VERSION = 3
from .policy_value_net import PolicyValueNetwork
from .state_encoder import (
    ENEMY_FEATURES,
    FUTURE_FEATURES,
    GLOBAL_FEATURES,
    MAP_CHANNELS,
    OPERATOR_FEATURES,
    StateEncoder,
    collate_states,
)

__all__ = [
    "ACTION_FEATURES",
    "OBSERVATION_VERSION",
    "EVENT_GRAPH_VERSION",
    "EVENT_FEATURES",
    "PROGRESSION_FEATURES",
    "FutureEventEncoder",
    "ENEMY_FEATURES",
    "FUTURE_FEATURES",
    "GLOBAL_FEATURES",
    "MAP_CHANNELS",
    "OPERATOR_FEATURES",
    "ActionEncoder",
    "NeuralEvaluator",
    "PolicyValueNetwork",
    "StateEncoder",
    "collate_states",
]
