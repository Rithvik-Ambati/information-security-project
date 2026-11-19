"""Score real .eml files end to end: real DKIM/SPF verification + behavioral risk + fusion.

  python scripts/analyze_eml.py demo/samples/*.eml           # offline DKIM keys for the bundled samples
  python scripts/analyze_eml.py inbox/msg.eml --live-dns     # real DNS lookups (needs network)
"""
import argparse
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from crypto.real_auth import verify_eml
from pipeline import Pipeline
from policy.engine import Policy


def offline_dns():
    recs = json.load(open(os.path.join(ROOT, "demo", "samples", "dns_records.json")))

    def f(name, timeout=5):
        n = name.decode() if isinstance(name, bytes) else name
        v = recs.get(n)
        return v.encode() if v else None
    return f


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--live-dns", action="store_true")
    ap.add_argument("--known-sender", default="alice@corp.example", help="give this sender a demo history")
    a = ap.parse_args()
    pipe = Pipeline()
    pipe.seed_demo_sender(a.known_sender)
    thr = pipe.fusion.thr
    pol = Policy(t_suspicious=thr, t_malicious=min(0.95, thr + (1 - thr) / 2))
    dnsf = None if a.live_dns else offline_dns()
    for f in a.files:
        r = verify_eml(open(f, "rb").read(), dnsfunc=dnsf, check_spf=a.live_dns)
        s = pipe.score_external(r["email"], r["trust"])
        _, label, action = pol.decide(s["trust"].score, s["risk"], s["fused"])
        print(f"{os.path.basename(f):38s} trust={s['trust'].score:.2f} risk={s['risk']:.2f} -> {label:10s} {action}")
        for why in s["trust"].reasons:
            print("    -", why)
