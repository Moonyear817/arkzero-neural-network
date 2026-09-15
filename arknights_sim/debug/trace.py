class BattleTrace:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.events = []

    def log(self, time, kind, **fields):
        if self.enabled:
            self.events.append({"time": round(time, 9), "event": kind, **fields})

    def text(self):
        return "\n".join(
            f"{e['time']:09.3f} {e['event']} "
            + " ".join(f"{k}={v}" for k, v in e.items() if k not in ["time", "event"])
            for e in sorted(self.events, key=lambda e: e["time"])
        )
