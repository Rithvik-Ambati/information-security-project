"""Full evaluation: baselines, ablation, account-takeover scenario, temporal drift, latency.
Protocol (no tuning on test): models fit on [0, mid); every method's threshold and the fusion strategy are
chosen on the validation window [mid, train end) (max F1 at FPR <= 1%); everything is reported on the
chronologically later test window.

Run:  python src/eval/full_eval.py [--backend transformer] [--model ID] [--local-only]
Env:  EMAIL_DATA_DIR=<dir> evaluates a dataset other than data/processed (used by multiseed.py)
Writes <data dir>/results.json and (for the default dir) docs/results.md."""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
import baselines                                                         # noqa: E402
from baselines import load, trust_timeline                               # noqa: E402
from behavior.features import build_matrix                               # noqa: E402
from behavior.model import RiskModel                                     # noqa: E402
from crypto.trust import KeyRegistry, verify                             # noqa: E402
from policy.fusion import best_threshold, select_fusion, trust_inputs    # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="tfidf", choices=["tfidf", "transformer"])
ap.add_argument("--model", default=None)
ap.add_argument("--local-only", action="store_true")
args = ap.parse_args()

emails, sp, _ = load()
y = np.array([e["label"] for e in emails])
atk = np.array([e["attack_type"] for e in emails])
text = [e["headers"]["subject"] + " " + e["body"] for e in emails]
trust_res = trust_timeline(emails)
chk_all, auth_all, integ_all = trust_inputs(trust_res)
H, TR, TE = sp["history"], sp["train"], sp["test"]
X, names = build_matrix(emails, fit_until=H[1])
mid = (TR[0] + TR[1]) // 2           # fit models on [0, mid); validate on [mid, TR end); test on TE
VAL = slice(mid, TR[1])
TEST = slice(*TE)
ATTACKS = ["spoof", "takeover", "lookalike", "tamper"]
PREV = 0.01                           # realistic attack prevalence for the precision-at-prevalence figure


def risk_model(end):
    idx = np.arange(0, end)
    mask = (y[:end] == 0) & (X[:end][:, 1] >= 3) & (idx < H[1])
    return RiskModel(args.backend, args.model, args.local_only).fit(X[:end], text[:end], y[:end], mask)


def tfidf_lr(end):
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                         LogisticRegression(max_iter=1000, class_weight="balanced")).fit(text[:end], y[:end])


def scores(m, c, b, sl):
    return {"content": c.predict_proba(text[sl.start:sl.stop])[:, 1],
            "auth": auth_all[sl],
            "behavior": b.predict_proba(np.column_stack([X[sl], m._anom(X[sl])]))[:, 1],
            "risk": m.risk(X[sl], text[sl.start:sl.stop])}


m, c = risk_model(mid), tfidf_lr(mid)
b = GradientBoostingClassifier(random_state=0).fit(np.column_stack([X[:mid], m._anom(X[:mid])]), y[:mid])
SV, ST = scores(m, c, b, VAL), scores(m, c, b, TEST)
yv, yt, at = y[VAL], y[TEST], atk[TEST]

fusion, fusions, fusion_rows = select_fusion(SV["risk"], chk_all[VAL], auth_all[VAL], integ_all[VAL], yv)
fv = fusion.score(SV["risk"], chk_all[VAL], auth_all[VAL], integ_all[VAL])
ft = fusion.score(ST["risk"], chk_all[TEST], auth_all[TEST], integ_all[TEST])

# method -> (validation scores, test scores); thresholds are all chosen on validation
methods = {
    "A. Content-only (TF-IDF+LR)": (SV["content"], ST["content"]),
    "B. Auth-only (crypto Trust)": (SV["auth"], ST["auth"]),
    "C. Simple combo max(A, B)": (np.maximum(SV["content"], SV["auth"]), np.maximum(ST["content"], ST["auth"])),
    "Ablation: behavior-only (no text)": (SV["behavior"], ST["behavior"]),
    "Ablation: risk (behavior+content)": (SV["risk"], ST["risk"]),
    f"FUSION (selected: {fusion.name})": (fv, ft),
}
thr = {k: best_threshold(yv, v[0])[0] for k, v in methods.items()}


def metrics(yy, s, t):
    p = (s >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yy, p, labels=[0, 1]).ravel()
    tpr, fpr = tp / max(tp + fn, 1), fp / max(fp + tn, 1)
    ppv = tpr * PREV / max(tpr * PREV + fpr * (1 - PREV), 1e-12)
    return dict(acc=(tp + tn) / len(yy), P=precision_score(yy, p, zero_division=0), R=tpr,
                F1=f1_score(yy, p, zero_division=0), FPR=fpr,
                AUC=roc_auc_score(yy, s) if len(set(yy)) > 1 else float("nan"), PPV1=ppv)


main_rows = {k: metrics(yt, v[1], thr[k]) for k, v in methods.items()}


def bootstrap_ci(yy, s, t, n=1000, seed=0):
    """95% bootstrap CI (resampling test emails) for F1, recall and FPR."""
    rng = np.random.default_rng(seed)
    p = s >= t
    f1s, rs, fprs = [], [], []
    for _ in range(n):
        i = rng.integers(0, len(yy), len(yy))
        yb, pb = yy[i], p[i]
        tp, fp, fn = (pb & (yb == 1)).sum(), (pb & (yb == 0)).sum(), (~pb & (yb == 1)).sum()
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1)); rs.append(tp / max(tp + fn, 1)); fprs.append(fp / max((yb == 0).sum(), 1))
    q = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {"F1": q(f1s), "R": q(rs), "FPR": q(fprs)}


boot = {k: bootstrap_ci(yt, v[1], thr[k]) for k, v in methods.items()}
per_class = {k: {a: float((v[1] >= thr[k])[at == a].mean()) for a in ATTACKS if (at == a).any()}
             for k, v in methods.items()}
sel = np.isin(at, ["takeover", "legit"])
tk_rows = {k: metrics(yt[sel], v[1][sel], thr[k]) for k, v in methods.items()}

# FPR broken down by kind of legitimate mail (where false alarms come from)
leg = yt == 0
src = np.array([e.get("true_sender") and ("unsigned" if e["sig"] is None else "signed") for e in emails[TEST]])
fpr_split = {}
fkey = f"FUSION (selected: {fusion.name})"
flag = ft >= thr[fkey]
for kind in ("signed", "unsigned"):
    mk = leg & (src == kind)
    fpr_split[kind] = (float(flag[mk].mean()) if mk.any() else float("nan"), int(mk.sum()))

# ---- temporal drift: split test window in 3 chunks; static model vs retrain-before-each-chunk
edges = np.linspace(TE[0], TE[1], 4).astype(int)
drift = {"static risk": [], "retrained risk": [], "content-only static": []}
tr_r, tr_c = thr["Ablation: risk (behavior+content)"], thr["A. Content-only (TF-IDF+LR)"]
for k in range(3):
    lo, hi = edges[k], edges[k + 1]
    yy = y[lo:hi]
    drift["static risk"].append(f1_score(yy, m.risk(X[lo:hi], text[lo:hi]) >= tr_r, zero_division=0))
    drift["content-only static"].append(f1_score(yy, c.predict_proba(text[lo:hi])[:, 1] >= tr_c, zero_division=0))
    rm = risk_model(lo)
    drift["retrained risk"].append(f1_score(yy, rm.risk(X[lo:hi], text[lo:hi]) >= tr_r, zero_division=0))

# ---- latency per email
reg = KeyRegistry()
n_lat = min(100, TE[1] - TE[0])
t0 = time.perf_counter()
for i in range(TE[0], TE[0] + n_lat):
    e = emails[i]
    pub = Ed25519PublicKey.from_public_bytes(bytes.fromhex(e["pub_hex"])) if e.get("pub_hex") else None
    verify(dict(e, pubkey=pub), reg)
t_trust = (time.perf_counter() - t0) / n_lat * 1000
t0 = time.perf_counter()
for i in range(TE[0], TE[0] + n_lat):
    m.risk(X[i:i + 1], [text[i]])
t_risk = (time.perf_counter() - t0) / n_lat * 1000


def table(rows):
    hdr = "| Method | thr | Acc | Prec | Recall | F1 | FPR | AUC | Prec@1% prev. |\n|---|---|---|---|---|---|---|---|---|\n"
    return hdr + "\n".join(f"| {k} | {thr.get(k, float('nan')):.2f} | " + " | ".join(
        f"{v[c]:.3f}" for c in ("acc", "P", "R", "F1", "FPR", "AUC", "PPV1")) + " |" for k, v in rows.items())


out = [f"# Evaluation results\n\nBackend: `{args.backend}`  |  test emails: {len(yt)}  |  attack rate: {yt.mean():.2f}\n"
       "Chronological split: history -> train -> test (future). Thresholds and the fusion strategy are chosen on a "
       "validation window (max F1 at FPR<=1%), never on test. `Prec@1% prev.` = precision if attacks were only 1% "
       "of mail, computed from the measured TPR/FPR.\n",
       "## Overall (test window)\n", table(main_rows),
       "\n## Recall per attack type\n",
       "| Method | " + " | ".join(ATTACKS) + " |\n|---|" + "---|" * len(ATTACKS),
       *[f"| {k} | " + " | ".join(f"{v.get(a, float('nan')):.2f}" for a in ATTACKS) + " |" for k, v in per_class.items()],
       "\n## Headline scenario: account takeover (valid signature) vs legitimate\n", table(tk_rows),
       "\n## False-alarm rate by kind of legitimate mail (selected fusion)\n",
       f"- signed legitimate: {fpr_split['signed'][0]:.3f} (n={fpr_split['signed'][1]})\n"
       f"- unsigned legitimate (lists/forwarders): {fpr_split['unsigned'][0]:.3f} (n={fpr_split['unsigned'][1]})",
       "\n## Fusion candidates (validation window)\n",
       "| Candidate | thr | val F1 | val FPR |\n|---|---|---|---|\n"
       + "\n".join(f"| {r['name']} | {r['thr']:.2f} | {r['val_f1']:.3f} | {r['val_fpr']:.3f} |" for r in fusion_rows)
       + f"\n\nChosen: **{fusion.name}** (simplest candidate within 0.005 val-F1 of the best).",
       "\n## Temporal drift (F1 over 3 consecutive future chunks)\n\n| Model | chunk 1 | chunk 2 | chunk 3 |\n|---|---|---|---|\n"
       + "\n".join(f"| {k} | " + " | ".join(f"{x:.3f}" for x in v) + " |" for k, v in drift.items()),
       f"\n## Latency per email\n\nTrust verification: {t_trust:.2f} ms | Risk scoring: {t_risk:.2f} ms",
       "\n> Caveats: legitimate mail = Enron sent folders + SpamAssassin ham (dates spread over the Enron span); "
       "phishing = Nazario 2015 mass-phishing text (near-duplicates removed, train/test pools disjoint) plus synthetic "
       "BEC text with template-level holdout. The authentication layer and attack injection are synthetic by design; "
       "Enron (2001) vs Nazario (2015) era/style differences still make mass phishing easier to separate than in "
       "the wild. Single run unless aggregated by multiseed.py."]
report = "\n".join(out)
print(report)

pdir = baselines.P
json.dump({"n_test": int(len(yt)), "attack_rate": float(yt.mean()),
           "methods": {k: dict(v, thr=float(thr[k])) for k, v in main_rows.items()}, "bootstrap_ci": boot,
           "per_class": per_class, "takeover": tk_rows, "fusion": fusion.name,
           "fusion_candidates": fusion_rows, "fpr_split": fpr_split, "drift": drift,
           "latency_ms": {"trust": t_trust, "risk": t_risk}},
          open(os.path.join(pdir, "results.json"), "w"), indent=1)
if os.path.abspath(pdir) == os.path.abspath(os.path.join(ROOT, "data", "processed")):
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    open(os.path.join(ROOT, "docs", "results.md"), "w", encoding="utf-8").write(report)
