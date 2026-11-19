"""Adversarial robustness: an adaptive attacker rewrites TEST-window attacks to evade the content/behavior model.

The attacker cannot change the cryptographic outcome of an attack type (a takeover keeps its valid signature,
a spoof stays unsigned...), so Trust inputs are left untouched; only text, send hour and recipient change.
Levels (cumulative strength shown in the report):
  0 none            original attacks
  1 keyword evasion replace payment/urgency keywords with synonyms
  2 benign padding  append generic benign sentences (good-word attack)
  3 style mimic     copy the victim's real greeting/sentences/closing from their past mail
  4 relationship    send at the victim's usual hour to a recipient the victim already writes to
  5 full mimic      levels 1 + 3 + 4 together
Run: python src/eval/adversarial.py  -> docs/adversarial_results.json"""
import copy
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import f1_score

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from behavior.features import build_matrix  # noqa: E402
from context import ATTACKS, build_context  # noqa: E402
from policy.fusion import best_threshold    # noqa: E402

SYN = {r"\bwire\b": "send", r"\btransfer\b": "move", r"\bgift cards?\b": "store vouchers", r"\bpassword\b": "access phrase",
       r"\bcredentials?\b": "access details", r"\binvoice\b": "bill", r"\bbank account\b": "account", r"\bpayment\b": "settlement",
       r"\burgent(ly)?\b": "soon", r"\bimmediately\b": "when you can", r"\bconfidential\b": "private", r"\bverify\b": "confirm",
       r"\bsuspend\w*\b": "pause", r"\bexpires?\b": "ends", r"\basap\b": "soon"}
PAD = ["Hope the project is moving along nicely.", "Let me know if the schedule still works for you.",
       "Thanks again for all the help last week.", "The agenda for the next meeting looks fine to me.",
       "I will catch up with the rest of the team later today."]


def sents(t):
    return [s for s in re.split(r"(?<=[.!?])\s+", t) if s]


def transform(e, level, hist):
    """hist: latest legitimate mail of the same sender before this one, or None."""
    e = copy.deepcopy(e)
    body = e["body"]
    if level in (1, 5):
        for pat, rep in SYN.items():
            body = re.sub(pat, rep, body, flags=re.I)
    if level == 2:
        body = body + " " + " ".join(PAD[:3])
    if level in (3, 5) and hist is not None:
        ss = sents(hist["body"])
        body = " ".join(ss[:2]) + " " + body + (" " + ss[-1] if len(ss) > 2 else "")
    if level in (4, 5) and hist is not None:
        hr = Counter(hist["hours"]).most_common(1)[0][0]
        e["date"] = e["headers"]["date"] = e["date"][:11] + f"{hr:02d}" + e["date"][13:]
        e["headers"]["to"] = hist["to"]
    e["body"] = body
    return e


if __name__ == "__main__":
    c = build_context()
    emails = c.emails
    # per-sender legit history (latest legit mail before index i)
    last = defaultdict(list)   # sender -> list of (idx, body, to, hour)
    for i, e in enumerate(emails):
        if e["label"] == 0:
            last[e["from"]].append((i, e["body"], e["headers"]["to"], c.hour[i]))

    def hist_for(i):
        lst = [t for t in last.get(emails[i]["true_sender"], []) if t[0] < i]
        if not lst:
            return None
        hours = [t[3] for t in lst[-20:]]
        return {"body": lst[-1][1], "to": lst[-1][2], "hours": hours}

    # thresholds from validation, same as the main evaluation
    sv = {"content": c.c.predict_proba(c.text[c.VAL.start:c.VAL.stop])[:, 1],
          "behavior": c.b.predict_proba(np.column_stack([c.X[c.VAL], c.m._anom(c.X[c.VAL])]))[:, 1],
          "risk": c.m.risk(c.X[c.VAL], c.text[c.VAL.start:c.VAL.stop])}
    thr = {k: best_threshold(c.y[c.VAL], v)[0] for k, v in sv.items()}
    te = np.arange(c.TE[0], c.TE[1])
    atk_idx = [i for i in te if c.y[i] == 1]
    out = {}
    for level in range(6):
        mod = list(emails)
        for i in atk_idx:
            mod[i] = transform(emails[i], level, hist_for(i)) if level else emails[i]
        X, _ = build_matrix(mod, fit_until=c.H[1]) if level else (c.X, None)
        text = [m["headers"]["subject"] + " " + m["body"] for m in mod]
        sl = c.TEST
        r = c.m.risk(X[sl], text[sl.start:sl.stop])
        scores = {"content-only": c.c.predict_proba(text[sl.start:sl.stop])[:, 1],
                  "behavior-only": c.b.predict_proba(np.column_stack([X[sl], c.m._anom(X[sl])]))[:, 1],
                  "risk (behavior+content)": r,
                  "FUSION (risk+trust)": c.fusion.score(r, c.chk[sl], c.auth[sl], c.integ[sl])}
        thrs = {"content-only": thr["content"], "behavior-only": thr["behavior"], "risk (behavior+content)": thr["risk"],
                "FUSION (risk+trust)": c.fusion.thr}
        y, at = c.y[sl], c.atk[sl]
        row = {}
        for name, s in scores.items():
            p = s >= thrs[name]
            row[name] = {"recall_all": float(p[y == 1].mean()), "recall_takeover": float(p[at == "takeover"].mean()),
                         "FPR": float(p[y == 0].mean()), "F1": float(f1_score(y, p, zero_division=0)),
                         **{f"recall_{a}": float(p[at == a].mean()) for a in ATTACKS}}
        out[level] = row
        print(f"level {level}: " + " | ".join(f"{n[:12]} takeover={v['recall_takeover']:.2f} all={v['recall_all']:.2f}"
                                               for n, v in row.items()), flush=True)
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "docs", "adversarial_results.json"), "w"), indent=1)
