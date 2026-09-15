from dataclasses import dataclass


@dataclass
class SkillState:
    sp: float = 0
    active_until: float = 0
    active: bool = False

    def is_active(self, data, time):
        return bool(data and (data.mode=='PASSIVE' or (self.active and data.mode in ('TOGGLE','INFINITE')) or time<self.active_until))

    def recover(self, data, dt, time):
        if data and data.recovery == 'AUTO_RECOVERY' and (not self.is_active(data,time) or data.mode=='TOGGLE'):
            self.sp = min(data.cost * data.max_charges, self.sp + dt)

    def gain(self, data, reason, time):
        if data and data.recovery == reason and (not self.is_active(data,time) or data.mode=='TOGGLE'):
            self.sp = min(data.cost * data.max_charges, self.sp + 1)
