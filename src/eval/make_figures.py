"""Generate every report figure + table from the current dataset/results.
Run:  python src/eval/make_figures.py        -> docs/figures/*.png (+ captions.json, live_simulation.gif), docs/tables/*.csv|.tex
Inputs used when present: <data>/results.json, data/processed_seed*/results.json, docs/stream_*.json, docs/adversarial_results.json"""
import glob
import json
import os
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from context import ATTACKS, build_context      # noqa: E402
from policy.fusion import best_threshold         # noqa: E402

FIG, TAB, DOCS = (os.path.join(ROOT, "docs", d) for d in ("figures", "tables", ""))
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk", font_scale=0.8)
PAL = {"content-only": "#8c8c8c", "auth-only": "#d95f02", "combo": "#7570b3", "behavior-only": "#1b9e77",
       "risk": "#66a61e", "FUSION": "#0b5fa5"}
ATK_COL = {"legit": "#9aa5b1", "spoof": "#d95f02", "takeover": "#c0172b", "lookalike": "#7570b3", "tamper": "#e6ab02"}
captions = {}
try:
    captions = json.load(open(os.path.join(FIG, 'captions.json')))
except Exception:
    pass
c = build_context()
emails, y, atk, TE, VAL = c.emails, c.y, c.atk, c.TE, c.VAL
yt, at = y[c.TEST], atk[c.TEST]

# ---- scores on val/test (same protocol as full_eval)
def _scores(sl):
    return {"content-only": c.c.predict_proba(c.text[sl.start:sl.stop])[:, 1],
            "auth-only": c.auth[sl],
            "behavior-only": c.b.predict_proba(np.column_stack([c.X[sl], c.m._anom(c.X[sl])]))[:, 1],
            "risk": c.m.risk(c.X[sl], c.text[sl.start:sl.stop])}


SV, ST = _scores(VAL), _scores(c.TEST)
SV["combo"], ST["combo"] = np.maximum(SV["content-only"], SV["auth-only"]), np.maximum(ST["content-only"], ST["auth-only"])
SV["FUSION"] = c.fusion.score(SV["risk"], c.chk[VAL], c.auth[VAL], c.integ[VAL])
ST["FUSION"] = c.fusion.score(ST["risk"], c.chk[c.TEST], c.auth[c.TEST], c.integ[c.TEST])
THR = {k: best_threshold(y[VAL], v)[0] for k, v in SV.items()}
NAMES = {"content-only": "Content-only", "auth-only": "Auth-only (crypto)", "combo": "Simple combo",
         "behavior-only": "Behavior-only", "risk": "Risk (behavior+content)", "FUSION": "FUSION (ours)"}
ORDER = ["content-only", "auth-only", "combo", "behavior-only", "risk", "FUSION"]


def save(name, cap):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight")
    plt.close()
    captions[name] = cap
    print("  ok", name, flush=True)


def c_dir():
    import baselines
    return baselines.P


ONLY = [x for x in os.environ.get("FIGS", "").split(",") if x]


def figure(fn):
    if ONLY and not any(fn.__name__.startswith(o) for o in ONLY):
        return fn
    try:
        fn()
    except Exception:
        print("  FAILED", fn.__name__)
        traceback.print_exc()
    return fn


@figure
def fig01_architecture():
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.axis("off")
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.2)

    def box(x, y_, w, h, t, col, fs=10):
        ax.add_patch(FancyBboxPatch((x, y_), w, h, boxstyle="round,pad=0.05,rounding_size=0.12", fc=col, ec="#333", lw=1.2))
        ax.text(x + w / 2, y_ + h / 2, t, ha="center", va="center", fontsize=fs)

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.6, color="#333"))
    box(0.1, 2.0, 1.7, 1.2, "Incoming\nemail", "#eef2f7")
    box(2.4, 2.0, 1.9, 1.2, "Parse &\nextract\nfeatures", "#eef2f7")
    box(5.0, 3.5, 3.3, 1.3, "Cryptographic Trust module\nDKIM/SPF/DMARC alignment\nkey continuity (TOFU), integrity", "#fde5d1")
    box(5.0, 0.3, 3.3, 1.7, "Behavioral Risk module\nsender profile + metadata + content\nwriting-style similarity\nIsolationForest + TF-IDF/DistilBERT", "#d6efe2")
    box(9.0, 2.0, 1.9, 1.2, "Adaptive policy\nengine\n(fusion)", "#d5e5f7")
    box(11.3, 3.2, 1.6, 0.9, "DELIVER", "#e6f4ea")
    box(11.3, 2.15, 1.6, 0.9, "WARN", "#fef7e0")
    box(11.3, 1.1, 1.6, 0.9, "BLOCK", "#fce8e6")
    arr(1.8, 2.6, 2.4, 2.6); arr(4.3, 2.8, 5.0, 4.0); arr(4.3, 2.4, 5.0, 1.4)
    arr(8.3, 4.0, 9.0, 2.9); arr(8.3, 1.5, 9.0, 2.3)
    arr(10.9, 2.8, 11.3, 3.5); arr(10.9, 2.6, 11.3, 2.6); arr(10.9, 2.3, 11.3, 1.6)
    ax.annotate("", xy=(9.95, 2.0), xytext=(9.95, 0.35), arrowprops=dict(arrowstyle="->", ls="--", color="#555"))
    ax.text(9.95, 0.1, "user / admin feedback -> adaptive thresholds,\ndrift detection, retraining", ha="center", va="top", fontsize=9, color="#555")
    ax.text(6.5, 5.05, "SHAP explanation in the warning banner", ha="center", fontsize=10, color="#555")
    save("fig01_architecture.png", "System architecture: parallel Trust and Risk modules feed an adaptive policy engine with a feedback loop.")


@figure
def fig02_dataset_composition():
    sp = c.sp
    spans = {"history": sp["history"], "train": sp["train"], "test": sp["test"]}
    rows = []
    for nm, (a, b) in spans.items():
        for t in ["legit"] + ATTACKS:
            rows.append({"split": nm, "type": t, "n": int((atk[a:b] == t).sum())})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    piv = df.pivot(index="split", columns="type", values="n").loc[["history", "train", "test"]][["legit"] + ATTACKS]
    piv.plot(kind="bar", stacked=True, ax=ax[0], color=[ATK_COL[k] for k in piv.columns], rot=0)
    ax[0].set_ylabel("emails"); ax[0].set_title("Dataset composition per chronological split")
    legit_src = pd.Series([e.get("source", "enron") if "source" in e else ("signed" if e["sig"] else "unsigned")
                           for e in emails if e["label"] == 0]).value_counts()
    sig = pd.Series(["signed" if e["sig"] else "unsigned" for e in emails if e["label"] == 0]).value_counts()
    ax[1].pie(sig.values, labels=[f"{k}\n({v})" for k, v in sig.items()], colors=["#4c78a8", "#f58518"], autopct="%1.1f%%",
              startangle=90)
    ax[1].set_title("Legitimate mail: signed vs unsigned")
    df.to_csv(os.path.join(TAB, "dataset_composition.csv"), index=False)
    save("fig02_dataset_composition.png", "Dataset composition by split and attack type; share of legitimate mail that is unsigned.")


@figure
def fig03_timeline():
    dates = pd.to_datetime([e["date"] for e in emails], format="ISO8601")
    df = pd.DataFrame({"m": dates.to_period("M").astype(str), "t": ["attack" if v else "legit" for v in y]})
    g = df.groupby(["m", "t"]).size().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(13, 4.2))
    g.plot(kind="bar", stacked=True, ax=ax, color=["#c0172b", "#9aa5b1"] if list(g.columns)[0] == "attack" else ["#9aa5b1", "#c0172b"], width=0.9)
    ax.set_xticks(ax.get_xticks()[::max(1, len(g) // 14)])
    ax.set_xlabel("month"); ax.set_ylabel("emails"); ax.set_title("Mail volume over time (legitimate vs attack)")
    ymax = ax.get_ylim()[1]
    for nm, idx in (("history|train", c.sp["train"][0]), ("fit|validation", c.mid), ("validation|test", c.TE[0])):
        m = pd.Period(dates[idx], "M").strftime("%Y-%m")
        if m in list(g.index):
            ax.axvline(list(g.index).index(m), color="k", ls="--", lw=1)
            ax.text(list(g.index).index(m), ymax * 0.97, nm, rotation=90, va="top", ha="right", fontsize=9)
    save("fig03_timeline.png", "Chronological layout: models are fitted on early mail, thresholds chosen on a validation slice, results reported on the latest mail.")


@figure
def fig04_send_hours():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    hrs = c.hour
    ax[0].hist([hrs[y == 0], hrs[y == 1]], bins=np.arange(25) - 0.5, color=["#9aa5b1", "#c0172b"], label=["legit", "attack"], density=True)
    ax[0].set_xlabel("hour of day"); ax[0].set_ylabel("density"); ax[0].legend(); ax[0].set_title("Send hour: legitimate vs attack")
    tk = hrs[atk == "takeover"]
    own = []
    for i in np.where(atk == "takeover")[0]:
        own.append(c.X[i, 15])
    ax[1].bar(["legit", "takeover"], [c.X[y == 0][:, 15].mean(), np.mean(own)], color=["#9aa5b1", "#c0172b"])
    ax[1].set_ylabel("share sent at night (21:00-06:00)"); ax[1].set_title("'Odd hour' feature rate")
    save("fig04_send_hours.png", "Half of simulated attacks use unusual hours, so timing alone is only a partial signal.")


@figure
def fig05_roc():
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    for k in ORDER:
        f, t, _ = roc_curve(yt, ST[k])
        ax.plot(f, t, color=PAL[k], lw=2.4 if k == "FUSION" else 1.6, label=NAMES[k])
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlim(-0.005, 0.2); ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("false positive rate (zoomed to 0-0.2)"); ax.set_ylabel("true positive rate"); ax.legend(fontsize=9, loc="lower right")
    ax.set_title("ROC curves (test window)")
    save("fig05_roc.png", "ROC curves on the latest-time test window (x axis zoomed to low false-positive rates).")


@figure
def fig06_pr():
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    pi_test, pi_real = yt.mean(), 0.01
    for k in ORDER:
        p, r, _ = precision_recall_curve(yt, ST[k])
        ax[0].plot(r, p, color=PAL[k], lw=2.4 if k == "FUSION" else 1.6, label=NAMES[k])
        # reweight precision to 1% prevalence: precision' = r*pi'/(r*pi' + fpr'*(1-pi')), fpr from precision
        fpr = r * pi_test * (1 - p) / np.maximum(p * (1 - pi_test), 1e-12)
        p2 = r * pi_real / np.maximum(r * pi_real + fpr * (1 - pi_real), 1e-12)
        ax[1].plot(r, p2, color=PAL[k], lw=2.4 if k == "FUSION" else 1.6)
    ax[0].set_title(f"Precision-recall (test prevalence {pi_test:.0%})"); ax[1].set_title("Re-weighted to 1% attack prevalence")
    for a in ax:
        a.set_xlabel("recall"); a.set_ylabel("precision"); a.set_ylim(0, 1.02)
    ax[0].legend(fontsize=8, loc="lower left")
    save("fig06_pr.png", "Precision-recall curves; right panel re-weights precision to a realistic 1% attack prevalence.")


def _multiseed_rows():
    files = sorted(glob.glob(os.path.join(ROOT, "data", "processed_seed*", "results.json")))
    runs = [json.load(open(f)) for f in files]
    return runs


@figure
def fig07_method_comparison():
    runs = _multiseed_rows()
    key = lambda k: "FUSION (selected)" if k.startswith("FUSION") else k
    names = [key(k) for k in runs[0]["methods"]] if runs else None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    data = {}
    for r in runs:
        for k, v in r["methods"].items():
            data.setdefault(key(k), []).append(v)
    labels = list(data)
    for ax, (m, ttl) in zip(axes, (("F1", "F1"), ("R", "Recall"), ("FPR", "False-positive rate"))):
        mu = [np.mean([v[m] for v in data[l]]) for l in labels]
        sd = [np.std([v[m] for v in data[l]], ddof=1) if len(data[l]) > 1 else 0 for l in labels]
        cols = [PAL["FUSION"] if l.startswith("FUSION") else "#9bb0c4" for l in labels]
        ax.barh(range(len(labels)), mu, xerr=sd, color=cols, capsize=3)
        ax.set_yticks(range(len(labels))); ax.set_yticklabels([l.replace("Ablation: ", "") for l in labels] if ax is axes[0] else [""] * len(labels), fontsize=9)
        ax.invert_yaxis(); ax.set_title(f"{ttl} (mean ± std, {len(runs)} seeds)")
    pd.DataFrame({l: {m: np.mean([v[m] for v in data[l]]) for m in ("F1", "R", "FPR", "AUC", "PPV1")} for l in labels}).T.round(4).to_csv(
        os.path.join(TAB, "method_comparison_mean.csv"))
    save("fig07_method_comparison.png", "Method comparison across seeds (thresholds and fusion chosen on validation only).")


@figure
def fig08_per_attack_heatmap():
    M = pd.DataFrame({NAMES[k]: {a: float((ST[k] >= THR[k])[at == a].mean()) for a in ATTACKS} for k in ORDER}).T
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.heatmap(M, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "recall"})
    ax.set_title("Recall per attack type (test window)")
    M.round(3).to_csv(os.path.join(TAB, "per_attack_recall.csv"))
    save("fig08_per_attack_heatmap.png", "Per-attack recall: crypto catches spoof/tamper/lookalike, behavior is needed for valid-signature takeover.")


@figure
def fig09_confusion():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, k in zip(axes, ["content-only", "auth-only", "FUSION"]):
        cm = confusion_matrix(yt, ST[k] >= THR[k], labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["ok", "flagged"], yticklabels=["legit", "attack"])
        ax.set_title(NAMES[k]); ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    save("fig09_confusion_matrices.png", "Confusion matrices on the test window for content-only, auth-only and the fusion.")


@figure
def fig10_score_distributions():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    s = ST["FUSION"]
    for lab, col, nm in ((0, "#9aa5b1", "legitimate"), (1, "#c0172b", "attack")):
        ax[0].hist(s[yt == lab], bins=40, alpha=0.75, color=col, label=nm, density=True)
    ax[0].axvline(c.fusion.thr, color="k", ls="--", label="alert threshold"); ax[0].set_yscale("log")
    ax[0].set_xlabel("fused threat score"); ax[0].set_title("Fusion score distribution"); ax[0].legend()
    for t in ["legit"] + ATTACKS:
        m = at == t
        ax[1].scatter(1 - c.trust[c.TEST][m] + np.random.default_rng(0).normal(0, 0.006, m.sum()), ST["risk"][m], s=8, alpha=0.45,
                      color=ATK_COL[t], label=t)
    ax[1].set_xlabel("1 - Trust (cryptographic suspicion)"); ax[1].set_ylabel("Risk (behavioral)"); ax[1].legend(markerscale=3, fontsize=9)
    ax[1].set_title("Trust vs Risk plane")
    save("fig10_score_distributions.png", "Left: fused-score distributions; right: the Trust/Risk plane showing that takeover lives at high risk but high trust.")


@figure
def fig11_shap_summary():
    import shap
    cols = c.names + ["text_score", "anomaly"]
    sl = c.TEST
    t = c.m.text.predict_proba(c.text[sl.start:sl.stop])[:, 1]
    Xa = c.m._aug(c.X[sl], t)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(Xa), min(1500, len(Xa)), replace=False)
    ex = shap.TreeExplainer(c.m.gb)
    sv = ex.shap_values(Xa[idx])
    plt.figure(figsize=(9, 7))
    shap.summary_plot(sv, Xa[idx], feature_names=cols, show=False, max_display=15)
    plt.title("SHAP summary: what drives the risk score (test sample)")
    save("fig11_shap_summary.png", "SHAP summary (beeswarm) of the behavioral risk model.")
    i = np.where(at == "takeover")[0][0]
    xa = Xa[i:i + 1]
    e = shap.Explanation(values=ex.shap_values(xa)[0], base_values=float(np.ravel(ex.expected_value)[0]), data=xa[0], feature_names=cols)
    plt.figure(figsize=(9, 6))
    shap.plots.waterfall(e, show=False, max_display=12)
    save("fig12_shap_waterfall_takeover.png", "SHAP waterfall for one account-takeover email: why it was scored high-risk despite a valid signature.")
    imp = pd.Series(c.m.gb.feature_importances_, index=cols).sort_values(ascending=False)[:15][::-1]
    plt.figure(figsize=(8, 6))
    imp.plot(kind="barh", color="#0b5fa5")
    plt.title("Risk model feature importance (gradient boosting)")
    imp.to_csv(os.path.join(TAB, "feature_importance.csv"))
    save("fig13_feature_importance.png", "Top-15 features of the behavioral risk model.")


@figure
def fig14_ablation():
    res = json.load(open(os.path.join(c_dir(), "results.json")))
    names = list(res["methods"])
    f1 = [res["methods"][k]["F1"] for k in names]
    tk = [res["per_class"][k]["takeover"] for k in names]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, f1, w, label="overall F1", color="#0b5fa5")
    ax.bar(x + w / 2, tk, w, label="takeover recall", color="#c0172b")
    ax.set_xticks(x); ax.set_xticklabels([n.replace("Ablation: ", "").replace(" (", "\n(") for n in names], fontsize=8)
    ax.set_ylim(0, 1.05); ax.legend(); ax.set_title("Ablation: contribution of each module")
    save("fig14_ablation.png", "Ablation: behavior-only, content-only, crypto-only and fused configurations (overall F1 and takeover recall).")


@figure
def fig15_fusion_candidates():
    rows = c.fusion_rows
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    cols = ["#0b5fa5" if r == c.fusion.name else "#9bb0c4" for r in df["name"]]
    ax.barh(df["name"], df["val_f1"], color=cols); ax.invert_yaxis(); ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("validation F1 (FPR <= 1%)"); ax.set_title("Fusion strategies compared on the validation window")
    df.round(4).to_csv(os.path.join(TAB, "fusion_candidates.csv"), index=False)
    save("fig15_fusion_candidates.png", "Candidate fusion strategies; the selected one (blue) is the simplest within 0.005 F1 of the best.")


@figure
def fig16_calibration():
    fig, ax = plt.subplots(figsize=(6, 5.4))
    for k in ["risk", "FUSION", "content-only"]:
        s = np.clip(ST[k], 0, 1)
        fr, mp = calibration_curve(yt, s, n_bins=10, strategy="quantile")
        ax.plot(mp, fr, "o-", color=PAL[k], label=NAMES[k])
    ax.plot([0, 1], [0, 1], "k:"); ax.set_xlabel("mean predicted score"); ax.set_ylabel("fraction of attacks"); ax.legend(fontsize=9)
    ax.set_title("Reliability diagram (test)")
    save("fig16_calibration.png", "Calibration of the scores on the test window.")


@figure
def fig17_fpr_breakdown_latency():
    res = json.load(open(os.path.join(c_dir(), "results.json")))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    fs = res["fpr_split"]
    ax[0].bar(["signed legit", "unsigned legit"], [fs["signed"][0], fs["unsigned"][0]], color=["#4c78a8", "#f58518"])
    for i, k in enumerate(["signed", "unsigned"]):
        ax[0].text(i, fs[k][0], f"n={fs[k][1]}", ha="center", va="bottom")
    ax[0].set_ylabel("false-alarm rate"); ax[0].set_title("False alarms by kind of legitimate mail")
    lat = res["latency_ms"]
    ax[1].bar(["Trust (crypto)", "Risk (model)"], [lat["trust"], lat["risk"]], color=["#d95f02", "#1b9e77"])
    ax[1].set_ylabel("ms per email"); ax[1].set_title("Latency per email")
    save("fig17_fpr_and_latency.png", "False-alarm rate by legitimate-mail type and per-email latency.")


@figure
def fig18_drift():
    p = os.path.join(DOCS, "stream_series.json")
    s = json.load(open(p))
    fig, ax = plt.subplots(figsize=(12, 4.8))
    cols = {"static": "#8c8c8c", "scheduled": "#7570b3", "drift-triggered": "#0b5fa5"}
    for k in ("static", "scheduled", "drift-triggered"):
        ax.plot(s[k]["x"], s[k]["f1"], color=cols[k], lw=2.2, label=k)
    b = s["meta"]["boundary_val_to_test"]
    ax.axvline(b, color="#c0172b", ls="--"); ax.text(b, 0.02, " new attack wording", color="#c0172b", rotation=90, va="bottom")
    for a in s["drift-triggered"]["alarms_at"]:
        ax.axvline(a, color="#0b5fa5", ls=":", alpha=0.7)
    for a in s["drift-triggered"]["retrain_at"]:
        ax.scatter([a], [0.5], marker="v", color="#0b5fa5", zorder=5)
    ax.set_xlabel("emails processed in the stream"); ax.set_ylabel("rolling F1 (window 400)"); ax.set_ylim(0, 1.02); ax.legend()
    ax.set_title("Concept drift: static vs scheduled vs drift-triggered retraining")
    save("fig18_drift_rolling_f1.png", "Rolling F1 on a stream with two distribution shifts; dotted lines = drift alarms, triangles = retraining.")


@figure
def fig19_thresholds():
    r = json.load(open(os.path.join(DOCS, "stream_results.json")))
    t, st_ = r["thresholds"], r["strategies"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    modes = list(t)
    ax[0].bar(modes, [t[m]["FN"] for m in modes], color="#c0172b"); ax[0].set_title("Missed attacks (FN)")
    ax[1].bar(modes, [t[m]["FP"] for m in modes], color="#f58518"); ax[1].set_title("False alarms (FP)")
    ax[2].bar(modes, [t[m]["F1"] for m in modes], color="#0b5fa5"); ax[2].set_title("F1"); ax[2].set_ylim(0, 1)
    for a in ax:
        a.tick_params(axis="x", rotation=15)
    pd.DataFrame(t).T.round(4).to_csv(os.path.join(TAB, "threshold_adaptation.csv"))
    pd.DataFrame({k: {m: v[m] for m in ("F1", "recall", "FPR", "retrains")} for k, v in st_.items()}).T.round(4).to_csv(
        os.path.join(TAB, "adaptation_strategies.csv"))
    save("fig19_threshold_adaptation.png", "Alert-threshold adaptation from user feedback: fixed vs global vs per-account thresholds.")


@figure
def fig20_adversarial():
    r = json.load(open(os.path.join(DOCS, "adversarial_results.json")))
    lv = ["none", "keywords", "padding", "style mimic", "relationship", "full mimic"]
    names = list(r["0"])
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for n in names:
        col = PAL["FUSION"] if n.startswith("FUSION") else {"content-only": PAL["content-only"], "behavior-only": PAL["behavior-only"]}.get(n, PAL["risk"])
        ax[0].plot(lv, [r[str(i)][n]["recall_takeover"] for i in range(6)], "o-", color=col, label=n, lw=2.4 if n.startswith("FUSION") else 1.6)
        ax[1].plot(lv, [r[str(i)][n]["recall_all"] for i in range(6)], "o-", color=col, label=n, lw=2.4 if n.startswith("FUSION") else 1.6)
    ax[0].set_title("Account-takeover recall under evasion"); ax[1].set_title("Recall over all attack types")
    for a in ax:
        a.set_ylim(0, 1.02); a.set_ylabel("recall"); a.tick_params(axis="x", rotation=25)
    ax[0].legend(fontsize=9)
    pd.DataFrame({n: {lv[i]: r[str(i)][n]["recall_takeover"] for i in range(6)} for n in names}).round(3).to_csv(
        os.path.join(TAB, "adversarial_takeover_recall.csv"))
    save("fig20_adversarial_robustness.png", "Robustness to an adaptive attacker (keyword evasion, padding, style/timing/recipient mimicry).")


@figure
def fig21_tfidf_vs_distilbert():
    d = os.path.join(ROOT, "data", "processed_distilbert", "results.json")
    a = json.load(open(os.path.join(c_dir(), "results.json")))
    b = json.load(open(d))
    pick = lambda r, pre: next(k for k in r["methods"] if k.startswith(pre))
    rows = []
    for lab, pre in (("Risk (behavior+content)", "Ablation: risk"), ("FUSION", "FUSION")):
        for nm, r in (("TF-IDF", a), ("DistilBERT", b)):
            k = pick(r, pre)
            rows.append({"model": lab, "text backend": nm, "F1": r["methods"][k]["F1"], "takeover recall": r["per_class"][k]["takeover"]})
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for a_, m in zip(ax, ("F1", "takeover recall")):
        sns.barplot(data=df, x="model", y=m, hue="text backend", ax=a_, palette=["#9bb0c4", "#0b5fa5"])
        a_.set_ylim(0, 1.05); a_.set_title(m)
        for c_ in a_.containers:
            a_.bar_label(c_, fmt="%.2f", fontsize=9)
    df.round(4).to_csv(os.path.join(TAB, "tfidf_vs_distilbert.csv"), index=False)
    save("fig21_tfidf_vs_distilbert.png", "Effect of replacing the TF-IDF text score with DistilBERT embeddings (single run, same split).")


@figure
def live_gif():
    from matplotlib.animation import FuncAnimation, PillowWriter
    s = json.load(open(os.path.join(DOCS, "stream_series.json")))
    ids = np.arange(c.mid, len(emails))
    yy = y[ids]
    sc = np.array(s["drift-triggered"]["scores"])
    pred = sc >= c.fusion.thr
    n = len(ids)
    frames = np.linspace(400, n, 45).astype(int)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [2.2, 1]})

    def upd(f):
        ax[0].clear(); ax[1].clear()
        xs = list(range(400, f + 1, 100))
        fs = []
        for e in xs:
            tp = (pred[e - 400:e] & (yy[e - 400:e] == 1)).sum(); fp = (pred[e - 400:e] & (yy[e - 400:e] == 0)).sum()
            fn = (~pred[e - 400:e] & (yy[e - 400:e] == 1)).sum(); fs.append(2 * tp / max(2 * tp + fp + fn, 1))
        ax[0].plot(xs, fs, color=PAL["FUSION"], lw=2.4); ax[0].set_xlim(0, n); ax[0].set_ylim(0, 1.02)
        b = s["meta"]["boundary_val_to_test"]
        ax[0].axvline(b, color="#c0172b", ls="--")
        for a in s["drift-triggered"]["retrain_at"]:
            if a <= f:
                ax[0].axvline(a, color="#0b5fa5", ls=":")
        ax[0].set_title(f"Live stream: {f} emails processed"); ax[0].set_xlabel("emails"); ax[0].set_ylabel("rolling F1")
        p, yp = pred[:f], yy[:f]
        vals = [int((p & (yp == 1)).sum()), int((~p & (yp == 1)).sum()), int((p & (yp == 0)).sum())]
        ax[1].bar(["caught", "missed", "false\nalarms"], vals, color=["#137333", "#c0172b", "#f58518"])
        ax[1].set_title("Outcomes so far")
    ani = FuncAnimation(fig, upd, frames=frames, interval=250)
    ani.save(os.path.join(FIG, "live_simulation.gif"), writer=PillowWriter(fps=4))
    plt.close()
    captions["live_simulation.gif"] = "Animated replay of the live stream with drift-triggered retraining."
    print("  ok live_simulation.gif")


# ---- LaTeX/CSV copies of the main results table
try:
    res = json.load(open(os.path.join(c_dir(), "results.json")))
    df = pd.DataFrame(res["methods"]).T[["thr", "acc", "P", "R", "F1", "FPR", "AUC", "PPV1"]].round(3)
    df.index = [k.replace("FUSION (selected: ", "FUSION (").replace("Ablation: ", "") for k in df.index]
    df.to_csv(os.path.join(TAB, "main_results.csv"))
    try:
        open(os.path.join(TAB, "main_results.tex"), "w").write(df.to_latex())
    except Exception:
        pass
except Exception:
    traceback.print_exc()
json.dump(captions, open(os.path.join(FIG, "captions.json"), "w"), indent=1)
print("figures:", len(captions))
