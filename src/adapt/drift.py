"""Concept-drift detection on the stream of labelled prediction errors (Page-Hinkley test)."""


class PageHinkley:
    """Detects an INCREASE in the mean of a 0/1 error stream. alarm() is True when the cumulative deviation
    from the running mean (minus tolerance delta) rises more than `lam` above its running minimum."""

    def __init__(self, delta=0.01, lam=6.0, min_n=40):
        self.delta, self.lam, self.min_n = delta, lam, min_n
        self.reset()

    def reset(self):
        self.n, self.mean, self.cum, self.cum_min = 0, 0.0, 0.0, 0.0

    def update(self, err: float) -> bool:
        self.n += 1
        self.mean += (err - self.mean) / self.n
        self.cum += err - self.mean - self.delta
        self.cum_min = min(self.cum_min, self.cum)
        if self.n >= self.min_n and self.cum - self.cum_min > self.lam:
            self.reset()
            return True
        return False
