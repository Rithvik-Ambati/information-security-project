import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from simulate import SCENARIOS, World
from crypto.trust import verify
from behavior.risk import risk
from policy.engine import Policy

w, p = World(), Policy()
for s in SCENARIOS:
    e, snd = w.scenario(s)
    t, r = verify(e, w.registry), risk(e, w.profiles.get(snd))
    print(f"{s:40s} trust={t.score:.2f} risk={r.score:.2f} -> {p.decide(t.score, r.score)[1:]}")
