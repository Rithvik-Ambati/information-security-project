"""Behavioral Risk module (v0 heuristic placeholder).

Replace with: sender-profile + metadata + NLP features -> XGBoost/transformer,
plus IsolationForest on the sender profile. Interface stays: risk(email, profile) -> RiskResult.
"""
import re
from dataclasses import dataclass, field

URGENT = re.compile(r"\b(urgent|immediately|asap|within \d+ (hours?|minutes?)|act now|suspend|verify your)\b", re.I)
PAYMENT = re.compile(r"\b(wire|transfer|gift card|invoice|bank account|payment|bitcoin|password)\b", re.I)
URL = re.compile(r"https?://[^\s]+", re.I)


@dataclass
class SenderProfile:
    """Built incrementally from a sender's past (legitimate) mail."""
    hours: list = field(default_factory=list)       # hour-of-day of past mails
    avg_len: float = 0.0
    n: int = 0
    urgent_rate: float = 0.0

    def update(self, hour, body):
        self.hours.append(hour)
        self.avg_len = (self.avg_len * self.n + len(body)) / (self.n + 1)
        self.urgent_rate = (self.urgent_rate * self.n + bool(URGENT.search(body))) / (self.n + 1)
        self.n += 1


@dataclass
class RiskResult:
    score: float
    contributions: dict


def risk(email: dict, profile: SenderProfile | None) -> RiskResult:
    body = email["body"] + " " + email["headers"].get("subject", "")
    hour = email.get("hour", 12)
    c = {}
    c["urgency language"] = 0.25 if URGENT.search(body) else 0.0
    c["payment/credential request"] = 0.30 if PAYMENT.search(body) else 0.0
    c["links in body"] = min(0.15, 0.05 * len(URL.findall(body)))
    if profile and profile.n >= 5:
        usual = sum(1 for h in profile.hours if abs(h - hour) <= 2) / len(profile.hours)
        c["unusual sending hour"] = round(0.15 * (1 - usual), 3)
        if not profile.urgent_rate and URGENT.search(body):
            c["out-of-character tone"] = 0.15
    return RiskResult(round(min(1.0, sum(c.values())), 3), {k: v for k, v in c.items() if v})
