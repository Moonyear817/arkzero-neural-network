"""Unlock configured stages only after repeated fixed-seed clean victories."""


class StageSampler:
    def __init__(self, stage="0-1", stages=None, *, curriculum=False, threshold=0.8, evaluations=3):
        self.stage = stage
        self.stages = tuple(stages or (stage,))
        self.curriculum = curriculum
        self.threshold = threshold
        self.evaluations = evaluations
        self.unlocked = 1 if curriculum else len(self.stages)
        self.streak = 0

    @property
    def active_stages(self):
        return self.stages[:self.unlocked]

    def sample(self, rng):
        return rng.choice(self.active_stages) if len(self.active_stages) > 1 else self.active_stages[0]

    def observe(self, rows):
        if not self.curriculum or self.unlocked == len(self.stages):
            return False
        passed = True
        for stage in self.active_stages:
            results = [r for r in rows if r['stage'] == stage]
            passed &= bool(results) and sum(r['success'] for r in results) / len(results) >= self.threshold
        self.streak = self.streak + 1 if passed else 0
        if self.streak >= self.evaluations:
            self.unlocked += 1
            self.streak = 0
            return True
        return False

    def state_dict(self):
        return dict(unlocked=self.unlocked, streak=self.streak)

    def load_state_dict(self, state):
        if not 1 <= state['unlocked'] <= len(self.stages) or not 0 <= state['streak'] < self.evaluations:
            raise ValueError('Invalid curriculum checkpoint')
        self.unlocked, self.streak = state['unlocked'], state['streak']
