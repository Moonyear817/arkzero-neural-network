"""Automatic deployment policy and optional learned squad selection."""
from pathlib import Path
import torch
from network import NeuralEvaluator,PolicyValueNetwork
from mcts import PUCTConfig,PUCTSearch
from training.squad_selection import SquadCoordinator
from training.decision_logging import decision_explanation


class NeuralBattleAgent:
    def __init__(self,checkpoint,data_dir,seed=12345,simulations=16):
        path=Path(checkpoint)
        if not path.is_file():raise ValueError('请先选择可用的模型文件，或等待训练生成模型')
        self.payload=torch.load(path,map_location='cpu',weights_only=True)
        self.model=PolicyValueNetwork(future_events_enabled=self.payload.get('network_metadata',{}).get('future_events_enabled',True))
        self.model.load_state_dict(self.payload['model_state_dict']);self.model.eval()
        self.model.reward_version = self.payload.get('reward_version', 1)
        self.evaluator=NeuralEvaluator(self.model,device='cpu')
        stored_config=self.payload.get('config',{})
        self.search=PUCTSearch(self.evaluator,PUCTConfig(mcts_simulations=simulations,seed=seed,
            temperature=0,prior_uniform_mix=0,dirichlet_epsilon=0,max_depth=stored_config.get('max_depth',64),
            enabled=stored_config.get('planner_enabled',True),top_k_actions=stored_config.get('top_k_actions',8),
            progressive_widening=stored_config.get('progressive_widening',True),
            widening_coefficient=stored_config.get('widening_coefficient',2.0),
            widening_exponent=stored_config.get('widening_exponent',.5)))
        self.coordinator=None
        self.last_decision_explanation=None
        state=self.payload.get('squad_selector')
        if state:
            self.coordinator=SquadCoordinator(data_dir,state['pool'],state['size'],trainable=False,
                progression=stored_config.get('operator_progression'),skill_overrides=stored_config.get('skill_overrides'))
            self.coordinator.load_state_dict(state)

    def choose_squad(self,env,stage_id,seed):
        if self.coordinator is None:raise ValueError('此模型尚无自主编队网络。请选择联合训练模型，或在人机协同模式下手动选队。')
        return self.coordinator.choose(env,stage_id,seed,False)

    def select_action(self,env,state):
        result=self.search.search(env,state,training=False)
        self.last_decision_explanation=decision_explanation(env,state,result)
        return result.selected_action

    def predict(self,env,state):
        return self.evaluator.predict(env,state)

    def top_k_actions(self,env,state,k=8):
        return self.evaluator.top_k_actions(env,state,k)
