"""Streaming simulation: process mail in time order with limited user feedback.

Part A (adaptation strategies, models can change):
  static            model fitted once on the training era
  scheduled         retrain every K emails on all labels revealed so far
  drift-triggered   retrain only when the Page-Hinkley detector fires on the labelled-error stream
Part B (alert thresholds, model fixed): fixed vs global-adaptive vs per-account-adaptive thresholds.

The stream starts at the validation window, so two real distribution shifts occur (train->val and val->test
use disjoint phishing pools and BEC 'ask' phrases). Feedback model: each email's label is revealed with
probability FEEDBACK_P (user 'report phishing' / 'not phishing' clicks).
Run:  python src/adapt/stream.py   -> docs/stream_results.json, docs/figures data via stream_data.npz"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import f1_score

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "eval"))
from adapt.drift import PageHinkley            # noqa: E402
from context import build_context, risk_model  # noqa: E402

FEEDBACK_P = 0.3
CHUNK = 100
SCHEDULE_K = 2000


def stream_idx(c):
    return np.arange(c.mid, len(c.emails))


def score_chunk(c, model, ids):
    r = model.risk(c.X[ids], [c.text[i] for i in ids])
    sl_chk, sl_auth, sl_int = c.chk[ids], c.auth[ids], c.integ[ids]
    return c.fusion.score(r, sl_chk, sl_auth, sl_int)


def run_models(c, strategy, seed=0, lam=2.0):
    rng = np.random.default_rng(seed)
    ids_all = stream_idx(c)
    model, ph = c.m, PageHinkley(delta=0.005, lam=lam, min_n=30)
    revealed, preds, scores, retrains, alarms = [], [], [], [], []
    since = 0
    for s in range(0, len(ids_all), CHUNK):
        ids = ids_all[s:s + CHUNK]
        sc = score_chunk(c, model, ids)
        pr = sc >= c.fusion.thr
        scores.append(sc)
        preds.append(pr)
        lab = ids[rng.random(len(ids)) < FEEDBACK_P]
        revealed += list(lab)
        fire = False
        for i in lab:
            p_i = pr[np.where(ids == i)[0][0]]
            fire |= ph.update(float(p_i != bool(c.y[i])))
        since += len(ids)
        if fire:
            alarms.append(int(ids[0]))
        retrain = (strategy == "scheduled" and since >= SCHEDULE_K) or (strategy == "drift" and fire)
        if retrain and len(revealed) > 100:
            model = risk_model(c, c.mid, extra_idx=revealed)
            retrains.append(int(ids[-1]))
            since = 0
    return np.concatenate(scores), np.concatenate(preds), retrains, alarms


def roll_f1(y, pred, w=400, step=100):
    xs, fs = [], []
    for e in range(w, len(y) + 1, step):
        xs.append(e)
        fs.append(f1_score(y[e - w:e], pred[e - w:e], zero_division=0))
    return xs, fs


def run_thresholds(c, scores, seed=1, lr=0.02, shrink=5):
    """Model fixed (scores given). Feedback tightens/loosens alert thresholds."""
    rng = np.random.default_rng(seed)
    ids = stream_idx(c)
    acct = np.array([c.emails[i].get("true_sender", c.emails[i]["from"]) for i in ids])
    y = c.y[ids]
    out = {}
    for mode in ("fixed", "global-adaptive", "per-account-adaptive", "per-account FP-only"):
        g, off, cnt = c.fusion.thr, {}, {}
        pred = np.zeros(len(ids), bool)
        for k in range(len(ids)):
            a = acct[k]
            thr = g if not mode.startswith("per-account") else g + off.get(a, 0.0) * cnt.get(a, 0) / (cnt.get(a, 0) + shrink)
            pred[k] = scores[k] >= thr
            if mode != "fixed" and rng.random() < FEEDBACK_P:
                step = 0.0
                if y[k] == 1 and not pred[k]:
                    step = -lr          # missed attack -> more sensitive
                elif y[k] == 0 and pred[k]:
                    step = +lr          # false alarm -> less sensitive
                if mode == "per-account FP-only" and step < 0:
                    step = 0.0          # conservative: never tighten, only relax on confirmed false alarms
                if mode == "global-adaptive":
                    g = float(np.clip(g + step, 0.02, 0.95))
                else:
                    off[a] = float(np.clip(off.get(a, 0.0) + step, -0.3, 0.6))
                    cnt[a] = cnt.get(a, 0) + 1
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
        out[mode] = {"F1": f1_score(y, pred, zero_division=0), "FP": fp, "FN": fn, "TP": tp,
                     "FPR": fp / max(int((y == 0).sum()), 1), "recall": tp / max(tp + fn, 1)}
    return out


if __name__ == "__main__":
    c = build_context()
    y = c.y[stream_idx(c)]
    res, series = {}, {}
    RUNS = [("static", "static", None), ("scheduled", "scheduled", None),
            ("drift-triggered", "drift", 2.0),            # main setting (lambda=2, fixed before seeing results)
            ("drift-triggered lam=1", "drift", 1.0), ("drift-triggered lam=4", "drift", 4.0)]
    for name, strat, lam in RUNS:
        print("strategy:", name, flush=True)
        sc, pr, rt, al = run_models(c, strat, lam=lam or 2.0)
        xs, fs = roll_f1(y, pr)
        res[name] = {"F1": f1_score(y, pr, zero_division=0), "recall": float((pr & (y == 1)).sum() / max((y == 1).sum(), 1)),
                     "FPR": float((pr & (y == 0)).sum() / max((y == 0).sum(), 1)), "retrains": len(rt),
                     "retrain_at": rt, "alarms_at": al, "n_stream": int(len(y))}
        series[name] = {"x": xs, "f1": fs, "retrain_at": rt, "alarms_at": al, "scores": sc.tolist()}
        print(" ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[name].items()
                     if k not in ("retrain_at", "alarms_at")}, flush=True)
    sc_static = np.array(series["static"]["scores"])
    thr_res = run_thresholds(c, sc_static)
    print("thresholds:", {k: {a: round(b, 3) for a, b in v.items()} for k, v in thr_res.items()})
    ids = stream_idx(c)
    series["meta"] = {"boundary_val_to_test": int(c.TE[0] - c.mid), "n": int(len(y)),
                      "attack_flag": c.y[ids].tolist(), "hour": c.hour[ids].tolist()}
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    json.dump({"strategies": res, "thresholds": thr_res}, open(os.path.join(ROOT, "docs", "stream_results.json"), "w"), indent=1)
    json.dump(series, open(os.path.join(ROOT, "docs", "stream_series.json"), "w"))
