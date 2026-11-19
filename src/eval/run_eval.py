"""Full comparison: auth-only vs risk-only vs fused Trust+Risk, incl. account-takeover headline."""
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from baselines import load, trust_scores, report
from behavior.features import build_matrix
from behavior.model import RiskModel

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="tfidf", choices=["tfidf", "transformer"])
ap.add_argument("--model", default=None, help="HF model id (default distilbert-base-uncased)")
ap.add_argument("--local-only", action="store_true", help="use only cached HF weights, no download")
args = ap.parse_args()

emails, sp, dns = load()
y = np.array([e["label"] for e in emails])
atk = np.array([e["attack_type"] for e in emails])
text = [e["headers"]["subject"] + " " + e["body"] for e in emails]
X, names = build_matrix(emails)
trust = trust_scores(emails, sp, dns)
h, tr, te = (slice(*sp[k]) for k in ("history", "train", "test"))

# train on history+train windows, test on the future window
fit = slice(h.start, tr.stop)
idx = np.arange(fit.start, fit.stop)
hist_mask = (y[fit] == 0) & (X[fit][:, 1] >= 3) & (idx < h.stop)
m = RiskModel(args.backend, args.model, args.local_only).fit(X[fit], text[fit.start:fit.stop], y[fit], hist_mask)
risk = m.risk(X[te], text[te.start:te.stop])

# learned fusion: meta-classifier on [risk, 1-trust]. Fit on train window (risk there is in-sample,
# so weights are optimistic -- replace with a dedicated validation window on real data).
risk_tr = m.risk(X[tr], text[tr.start:tr.stop])
meta = LogisticRegression(class_weight="balanced").fit(np.column_stack([risk_tr, 1 - trust[tr]]), y[tr])
fused = meta.predict_proba(np.column_stack([risk, 1 - trust[te]]))[:, 1]
print("meta weights [risk, 1-trust]:", np.round(meta.coef_[0], 2))

auth = 1 - trust[te]
t = {"auth-only": (auth, 0.2), "risk-only": (risk, 0.5), "FUSION": (fused, 0.5)}
print(f"test emails={te.stop - te.start}  attack rate={y[te].mean():.2f}")
for k, (s, thr) in t.items():
    report(k, y[te], s, thr, atk[te])

sel = np.isin(atk[te], ["takeover", "legit"])
print("\nHeadline: account takeover (valid signature) vs legit only")
for k, (s, thr) in t.items():
    report(k, y[te][sel], s[sel], thr)

gb = m.gb.feature_importances_
cols = names + ["text_score", "anomaly"]
print("\ntop features:", [(cols[i], round(float(gb[i]), 2)) for i in np.argsort(gb)[::-1][:6]])
