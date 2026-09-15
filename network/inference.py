"""Qt-free neural inference and planner adapter, preserving legal action order."""
import torch
from arknights_sim.environment.rewards import REWARD_VERSION
from .action_encoder import ActionEncoder
from .state_encoder import StateEncoder


class NeuralEvaluator:
    def __init__(self, model, device='cpu', state_encoder=None, action_encoder=None):
        self.model=model
        self.device=torch.device(device)
        self.state_encoder=state_encoder or StateEncoder()
        self.action_encoder=action_encoder or ActionEncoder()
        self.model.to(self.device)
        self.evaluations=0

    def _predict(self,env,state,actions,include_embedding):
        if env.is_terminal(state):
            raise ValueError('Terminal states require exact environment values, not neural evaluation')
        actions=tuple(actions)
        if not actions:raise ValueError('Cannot evaluate an empty legal-action set')
        encoded={key:tensor.to(self.device) for key,tensor in self.state_encoder.encode(state).items()}
        action_features=self.action_encoder.encode(state,actions).to(self.device)
        was_training=self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                if include_embedding:
                    logits,value,embedding=self.model.forward_with_embedding(encoded,action_features)
                else:
                    logits,value=self.model(encoded,action_features)
                    embedding=None
                probabilities=torch.softmax(logits,dim=-1).cpu()
                scalar=float(value.cpu()) if getattr(self.model,'reward_version',REWARD_VERSION)==REWARD_VERSION else 0.0
                if not bool(torch.isfinite(probabilities).all()) or not -1<=scalar<=1:
                    raise ValueError('Network produced invalid priors or value')
                if embedding is not None and not bool(torch.isfinite(embedding).all()):
                    raise ValueError('Network produced invalid state embedding')
                self.evaluations+=1
                result=dict(policy=tuple(float(p) for p in probabilities),value=scalar,legal_actions=actions)
                if include_embedding:result['state_embedding']=embedding.detach().cpu()
                return result
        finally:self.model.train(was_training)

    def evaluate(self,env,state,actions):
        result=self._predict(env,state,actions,False)
        return result['policy'],result['value']

    __call__=evaluate

    def predict(self,env,state,actions=None):
        return self._predict(env,state,env.legal_actions(state) if actions is None else actions,True)

    def top_k_actions(self,env,state,k):
        if type(k) is not int or k<1:raise ValueError('k must be a positive integer')
        result=self.predict(env,state)
        order=sorted(range(len(result['policy'])),key=lambda i:(-result['policy'][i],i))[:k]
        return tuple((result['legal_actions'][i],result['policy'][i]) for i in order)
