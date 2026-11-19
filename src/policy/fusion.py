"""Fusion of the behavioral Risk score and the cryptographic Trust result.

Candidates are fitted/selected on a held-out VALIDATION window only. Selection rule (fixed in advance):
highest validation F1 at FPR <= 1%; any candidate within 0.005 F1 of the best loses to the simpler one
(candidates are listed simplest first).
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

CHK = ["signature", "body_hash", "alignment", "key_continuity", "header_consistency", "domain_legitimacy"]


def trust_inputs(trust_results):
    """-> (checks matrix, auth = 1-trust, integrity_fail flag) for a list of TrustResult."""
    chk = np.array([[t.checks[c] for c in CHK] for t in trust_results])
    auth = np.array([1 - t.score for t in trust_results])
    integ = ((chk[:, 0] == 0) | (chk[:, 1] == 0)).astype(float)
    return chk, auth, integ


class Fusion:
    def __init__(self, name, fn, thr=0.5):
        self.name, self.fn, self.thr = name, fn, thr

    def score(self, risk, chk, auth, integ):
        return self.fn(np.asarray(risk), chk, auth, integ)

    def flag(self, risk, chk, auth, integ):
        return self.score(risk, chk, auth, integ) >= self.thr


def _candidates(rv, cv, av, iv, yv):
    c = [("max(risk, 1-trust)", lambda r, k, a, i: np.maximum(r, a)),
         ("max(risk, integrity-rule)", lambda r, k, a, i: np.maximum(r, i))]
    lr1 = LogisticRegression(class_weight="balanced").fit(np.column_stack([rv, av]), yv)
    c.append(("LR[risk, 1-trust]", lambda r, k, a, i: lr1.predict_proba(np.column_stack([r, a]))[:, 1]))
    lr2 = LogisticRegression(class_weight="balanced", max_iter=2000).fit(np.column_stack([rv, cv]), yv)
    c.append(("LR[risk, trust checks]", lambda r, k, a, i: lr2.predict_proba(np.column_stack([r, k]))[:, 1]))
    gb = GradientBoostingClassifier(random_state=0, n_estimators=100, max_depth=2).fit(np.column_stack([rv, cv]), yv)
    c.append(("GB[risk, trust checks]", lambda r, k, a, i: gb.predict_proba(np.column_stack([r, k]))[:, 1]))
    return c


def best_threshold(y, s, max_fpr=0.01):
    """Max-F1 threshold with FPR <= max_fpr; if none qualifies, the lowest-FPR threshold."""
    grid = np.unique(np.concatenate([np.linspace(0.02, 0.98, 49), np.quantile(s, [0.5, 0.7, 0.8, 0.9, 0.95])]))
    best, fallback = None, None
    for t in grid:
        p = s >= t
        fpr = p[y == 0].mean()
        f = f1_score(y, p, zero_division=0)
        if fpr <= max_fpr and (best is None or f > best[1]):
            best = (t, f, fpr)
        if fallback is None or (fpr, -f) < (fallback[2], -fallback[1]):
            fallback = (t, f, fpr)
    return best or fallback


def select_fusion(rv, cv, av, iv, yv, tol=0.005):
    rows, fusions = [], []
    for name, fn in _candidates(rv, cv, av, iv, yv):
        s = fn(np.asarray(rv), cv, av, iv)
        thr, f1, fpr = best_threshold(yv, s)
        fusions.append(Fusion(name, fn, float(thr)))
        rows.append({"name": name, "thr": float(thr), "val_f1": float(f1), "val_fpr": float(fpr)})
    top = max(r["val_f1"] for r in rows)
    chosen = next(i for i, r in enumerate(rows) if r["val_f1"] >= top - tol)  # simplest within tolerance
    return fusions[chosen], fusions, rows
