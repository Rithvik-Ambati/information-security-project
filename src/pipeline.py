"""End-to-end pipeline used by the demo: ingest -> trust + risk -> fusion -> explanation."""
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src", "eval"))
from baselines import load, trust_timeline       # noqa: E402
from policy.fusion import select_fusion, trust_inputs  # noqa: E402
from behavior.features import NAMES, build_matrix   # noqa: E402
from behavior.model import RiskModel            # noqa: E402


class Pipeline:
    def __init__(self, backend="tfidf", model_name=None, local_only=False):
        self.emails, self.sp, self.dns = load()
        e, sp = self.emails, self.sp
        self.y = np.array([x["label"] for x in e])
        self.atk = np.array([x["attack_type"] for x in e])
        self.text = [x["headers"]["subject"] + " " + x["body"] for x in e]

        # walk time order once: sender profiles + feature rows (causal); trust verdicts with an adaptive TOFU registry
        self.X, _, self.profiles, self.style_vec = build_matrix(e, fit_until=sp["history"][1], return_state=True)
        self.trust_res = trust_timeline(e)
        self.trust = np.array([t.score for t in self.trust_res])
        self.chk, self.auth, self.integ = trust_inputs(self.trust_res)

        h, tr = slice(*sp["history"]), slice(*sp["train"])
        # same protocol as eval/full_eval.py: risk model on [0, mid); fusion fitted on held-out [mid, train end)
        mid = self.mid = (tr.start + tr.stop) // 2
        fit = slice(0, mid)
        idx = np.arange(fit.start, fit.stop)
        mask = (self.y[fit] == 0) & (self.X[fit][:, 1] >= 3) & (idx < h.stop)
        self.model = RiskModel(backend, model_name, local_only).fit(
            self.X[fit], self.text[fit.start:fit.stop], self.y[fit], mask)
        # fusion strategy chosen on the held-out validation window [mid, train end) (see policy/fusion.py)
        val = slice(mid, tr.stop)
        r_val = self.model.risk(self.X[val], self.text[val.start:val.stop])
        self.fusion, self.fusions, self.fusion_table = select_fusion(
            r_val, self.chk[val], self.auth[val], self.integ[val], self.y[val])
        self.cols = NAMES + ["text_score", "anomaly"]
        t_tr = self.model.text.predict_proba(self.text[tr.start:tr.stop])[:, 1]
        self.mean_aug = self.model._aug(self.X[tr], t_tr).mean(axis=0)

    def batch_scores(self, ids):
        """Fast (no SHAP) risk and fused scores for many dataset emails."""
        ids = np.asarray(ids)
        risk = self.model.risk(self.X[ids], [self.text[i] for i in ids])
        return risk, self.fusion.score(risk, self.chk[ids], self.auth[ids], self.integ[ids])

    def test_indices(self, attack_type):
        a, b = self.sp["test"]
        return [i for i in range(a, b) if self.atk[i] == attack_type]

    def score(self, i):
        t = self.model.text.predict_proba([self.text[i]])[:, 1]
        xa = self.model._aug(self.X[i:i + 1], t)
        risk = float(self.model.gb.predict_proba(xa)[0, 1])
        fused = float(self.fusion.score([risk], self.chk[i:i + 1], self.auth[i:i + 1], self.integ[i:i + 1])[0])
        contrib = self._explain(xa, risk)
        return {"trust": self.trust_res[i], "risk": risk, "fused": fused,
                "why": contrib}

    def _explain(self, xa, risk):
        """SHAP (TreeExplainer on the boosted model, log-odds contributions) when available,
        otherwise occlusion: risk drop when a feature is set to its typical value."""
        contrib = {}
        try:
            import shap
            if not hasattr(self, "_explainer"):
                self._explainer = shap.TreeExplainer(self.model.gb)
            vals = np.asarray(self._explainer.shap_values(xa)).reshape(-1)
            contrib = {c: round(float(v), 3) for c, v in zip(self.cols, vals) if v > 0.05}
        except Exception:
            for j, c in enumerate(self.cols):
                x2 = xa.copy()
                x2[0, j] = self.mean_aug[j]
                d = risk - float(self.model.gb.predict_proba(x2)[0, 1])
                if d > 0.01:
                    contrib[c] = round(d, 3)
        return dict(sorted(contrib.items(), key=lambda kv: -kv[1])[:6])

    def seed_demo_sender(self, addr, to="me@corp.example", n=40, seed=3):
        """Give a demo sender a realistic legitimate history (business hours, calm tone) so that
        'out of character' can be judged for emails uploaded in the demo."""
        import random
        rng = random.Random(seed)
        bodies = ["Hi, attaching the weekly report. Let me know if you have questions. Thanks, {n}",
                  "Meeting moved to 3pm tomorrow. The agenda is in the shared doc. Thanks, {n}",
                  "Thanks for the update, I will review the draft by Friday. Regards, {n}",
                  "Lunch on Thursday? The new place near the office looks good. Cheers, {n}",
                  "Reminder: sprint demo is next week. Please update your tickets. Thanks, {n}",
                  "Sharing the notes from today's call. Action items are at the bottom. Best, {n}"]
        for k in range(n):
            ev = {"from": addr, "to": to, "body": rng.choice(bodies).format(n=addr.split("@")[0].title()),
                  "headers": {"subject": "Update", "to": to, "return-path": addr, "from": addr},
                  "date": f"2026-10-{1 + k % 28:02d}T{rng.randint(9, 17):02d}:00:00", "sig": {"d": "x"}, "label": 0}
            v = self.style_vec.transform([ev["body"]]).tocsr()[0]
            self.profiles.update(ev, v)

    def score_external(self, e, trust):
        """Score an email that is NOT in the dataset (e.g. an uploaded .eml) with a precomputed TrustResult."""
        v = self.style_vec.transform([e["body"]]).tocsr()[0]
        x = np.array([self.profiles.features(e, v)], dtype=float)
        text = e["headers"]["subject"] + " " + e["body"]
        t = self.model.text.predict_proba([text])[:, 1]
        xa = self.model._aug(x, t)
        risk = float(self.model.gb.predict_proba(xa)[0, 1])
        chk, auth, integ = trust_inputs([trust])
        fused = float(self.fusion.score([risk], chk, auth, integ)[0])
        return {"trust": trust, "risk": risk, "fused": fused, "why": self._explain(xa, risk)}
