class DamageCalculator:
    @staticmethod
    def physical_damage(atk, defense):
        return max(0, atk - defense, 0.05 * atk)

    @staticmethod
    def arts_damage(atk, resistance):
        return max(0, atk * (1 - resistance / 100), 0.05 * atk)

    @staticmethod
    def true_damage(atk):
        return max(0, atk)

    @classmethod
    def calculate(cls, kind, atk, defense=0, resistance=0):
        if kind == "PHYSICAL":
            return cls.physical_damage(atk, defense)
        if kind == "ARTS":
            return cls.arts_damage(atk, resistance)
        if kind == "TRUE":
            return cls.true_damage(atk)
        raise ValueError(kind)
