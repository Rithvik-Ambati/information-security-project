"""Adaptive policy engine (v0): fuse Trust + Risk, map to class and action, adapt via feedback."""
from dataclasses import dataclass


@dataclass
class Policy:
    t_suspicious: float = 0.10
    t_malicious: float = 0.50
    lr: float = 0.01

    def fuse(self, trust: float, risk: float) -> float:
        return max(risk, 1 - trust)  # either signal alone can raise an alert

    def decide(self, trust: float, risk: float, fused: float | None = None):
        """fused: pass a learned fused score; otherwise max(risk, 1 - trust)."""
        s = self.fuse(trust, risk) if fused is None else fused
        if s >= self.t_malicious:
            return s, "Malicious", "BLOCK / QUARANTINE"
        if s >= self.t_suspicious:
            return s, "Suspicious", "DELIVER WITH WARNING"
        return s, "Trusted", "DELIVER"

    def feedback(self, fused_score: float, was_phishing: bool):
        """Dynamic thresholds: false positive -> loosen; missed phish -> tighten."""
        if was_phishing and fused_score < self.t_suspicious:
            self.t_suspicious -= self.lr
        elif not was_phishing and fused_score >= self.t_suspicious:
            self.t_suspicious += self.lr
        self.t_suspicious = min(max(self.t_suspicious, 0.02), self.t_malicious - 0.1)
