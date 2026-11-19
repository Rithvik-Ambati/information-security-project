"""Shared experiment context: load data, causal features, trust verdicts, models fitted on [0, mid), validation-chosen
thresholds and fusion. Used by the stream/adversarial/figure scripts so they all use the SAME protocol as
full_eval.py.  Cached per data dir (EMAIL_DATA_DIR) as _ctx.pkl."""
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
import baselines                                                         # noqa: E402
from baselines import load, trust_timeline                               # noqa: E402
from behavior.features import build_matrix                               # noqa: E402
from behavior.model import RiskModel                                     # noqa: E402
from policy.fusion import best_threshold, select_fusion, trust_inputs    # noqa: E402

ATTACKS = ["spoof", "takeover", "lookalike", "tamper"]


def risk_model(ctx_like, end, backend="tfidf", model_name=None, local_only=False, extra_idx=None):
    """Fit a RiskModel on rows [0, end) (+ optional extra revealed-label rows)."""
    idx = np.arange(end) if extra_idx is None else np.concatenate([np.arange(end), np.asarray(extra_idx, int)])
    mask = (ctx_like.y[idx] == 0) & (ctx_like.X[idx][:, 1] >= 3) & (idx < ctx_like.H[1])
    return RiskModel(backend, model_name, local_only).fit(
        ctx_like.X[idx], [ctx_like.text[i] for i in idx], ctx_like.y[idx], mask)


def tfidf_lr(ctx_like, end):
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=1000, class_weight="balanced")).fit(
        ctx_like.text[:end], ctx_like.y[:end])


def build_context(backend="tfidf", model_name=None, local_only=False, use_cache=True):
    pdir = baselines.P
    cache = os.path.join(pdir, f"_ctx_{backend}.pkl")
    em = os.path.join(pdir, "emails.jsonl")
    c = None
    if use_cache and os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(em):
        c = pickle.load(open(cache, "rb"))
    if c is None:
        emails, sp, _ = load()
        c = SimpleNamespace(emails=emails, sp=sp, backend=backend)
        c.y = np.array([e["label"] for e in emails])
        c.atk = np.array([e["attack_type"] for e in emails])
        c.text = [e["headers"]["subject"] + " " + e["body"] for e in emails]
        c.H, c.TR, c.TE = sp["history"], sp["train"], sp["test"]
        c.X, c.names = build_matrix(emails, fit_until=c.H[1])
        c.trust_res = trust_timeline(emails)
        c.chk, c.auth, c.integ = trust_inputs(c.trust_res)
        c.mid = (c.TR[0] + c.TR[1]) // 2
        c.m = risk_model(c, c.mid, backend, model_name, local_only)
        c.c = tfidf_lr(c, c.mid)
        mid = c.mid
        c.b = GradientBoostingClassifier(random_state=0).fit(np.column_stack([c.X[:mid], c.m._anom(c.X[:mid])]),
                                                             c.y[:mid])
        pickle.dump(c, open(cache, "wb"))
    c.VAL, c.TEST = slice(c.mid, c.TR[1]), slice(*c.TE)
    c.trust = np.array([t.score for t in c.trust_res])
    c.hour = np.array([int(e["date"][11:13]) for e in c.emails])
    # validation-chosen fusion + thresholds (cheap, so recomputed rather than pickled)
    sv = c.m.risk(c.X[c.VAL], c.text[c.VAL.start:c.VAL.stop])
    c.fusion, c.fusions, c.fusion_rows = select_fusion(sv, c.chk[c.VAL], c.auth[c.VAL], c.integ[c.VAL], c.y[c.VAL])
    return c


def fused_scores(c, risk, sl):
    return c.fusion.score(risk, c.chk[sl], c.auth[sl], c.integ[sl])
