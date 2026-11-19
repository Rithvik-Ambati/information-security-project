"""Streamlit demo.  Run:  streamlit run demo/app.py
Tabs: live inbox simulation | scenario explorer | analyze a real .eml (real DKIM/SPF) | results & figures.
Needs data/processed (python scripts/build_dataset.py --demo, or the real corpora)."""
import glob
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from crypto.real_auth import verify_eml
from explain import banner_html
from pipeline import Pipeline
from policy.engine import Policy


def show_html(html, height=380, scrolling=True):
    """st.iframe on new Streamlit (components.html is deprecated), components.html on older ones."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        components.html(html, height=height, scrolling=scrolling)

st.set_page_config(page_title="Adaptive Email Security", layout="wide")
st.title("AI-Driven Adaptive Email Security")
st.caption("Behavioral risk analysis + cryptographic trust verification (research prototype)")

if not os.path.exists(os.path.join(ROOT, "data", "processed", "emails.jsonl")):
    st.error("No dataset found. Run `python scripts/build_dataset.py --demo` first.")
    st.stop()


@st.cache_resource(show_spinner="Training models on the history/train windows (about a minute)...")
def get_pipeline():
    p = Pipeline(os.environ.get("EMAIL_BACKEND", "tfidf"), os.environ.get("EMAIL_MODEL"),
                 os.environ.get("EMAIL_LOCAL_ONLY") == "1")
    p.seed_demo_sender("alice@corp.example")   # demo history for the bundled .eml samples
    return p


@st.cache_resource
def static_scores(_pipe):
    ids = np.arange(_pipe.mid, len(_pipe.emails))
    r, f = [], []
    for s in range(0, len(ids), 2000):
        a, b = _pipe.batch_scores(ids[s:s + 2000])
        r.append(a); f.append(b)
    return np.concatenate(r), np.concatenate(f)


pipe = get_pipeline()
THR = pipe.fusion.thr
MAL = min(0.95, THR + (1 - THR) / 2)
if "policy" not in st.session_state:
    st.session_state.policy = Policy(t_suspicious=THR, t_malicious=MAL)
policy = st.session_state.policy
LABELS = {"legit": "Legitimate", "spoof": "Spoofed sender (no signature)",
          "takeover": "Account takeover (valid signature!)", "lookalike": "Lookalike domain",
          "tamper": "Header/body tampering"}
VERDICT = {0: "DELIVER", 1: "WARN", 2: "BLOCK"}
LABEL_OF = {"DELIVER": "Trusted", "WARN": "Suspicious", "BLOCK": "Malicious"}
st.sidebar.markdown(f"**Fusion strategy (picked on validation):** `{pipe.fusion.name}`")
st.sidebar.write(f"Alert threshold `{THR:.2f}` | block threshold `{MAL:.2f}`")

tab_live, tab_scn, tab_eml, tab_res = st.tabs(
    ["Live simulation", "Scenario explorer", "Analyze a real .eml", "Results & figures"])

# ---------------------------------------------------------------- live simulation
with tab_live:
    series_path = os.path.join(ROOT, "docs", "stream_series.json")
    series = json.load(open(series_path)) if os.path.exists(series_path) else None
    ids = np.arange(pipe.mid, len(pipe.emails))
    n = len(ids)
    y = pipe.y[ids]
    c1, c2, c3, c4 = st.columns(4)
    strategies = ["static"] + ([s for s in ("scheduled", "drift-triggered") if series and s in series] if series else [])
    strategy = c1.selectbox("Model adaptation", strategies,
                            help="static = never retrained; drift-triggered = retrains when drift is detected")
    speed = c2.slider("Speed (emails / second)", 5, 120, 30)
    reveal = c3.checkbox("Reveal ground truth", True)
    if c4.button("Reset stream"):
        st.session_state.pos = 0
    pos = st.session_state.get("pos", 0)
    b1, b2 = st.columns(2)
    run = b1.button("Start / resume")
    b2.button("Pause")      # any interaction reruns the script, which halts the loop

    if strategy == "static" or not series:
        fused = static_scores(pipe)[1]
    else:
        fused = np.array(series[strategy]["scores"])
    verdict = np.where(fused >= MAL, 2, np.where(fused >= THR, 1, 0))
    boundary = series["meta"]["boundary_val_to_test"] if series else int(pipe.sp["test"][0] - pipe.mid)
    events = []
    if series and strategy in series:
        events += [(a, "Drift detector fired") for a in series[strategy]["alarms_at"]]
        events += [(a, "Model retrained on revealed labels") for a in series[strategy]["retrain_at"]]
    events.append((boundary, "New attack wording appears (distribution shift)"))
    events.sort()

    kpi, chart_ph, feed_ph, log_ph, prog = st.empty(), st.empty(), st.empty(), st.empty(), st.empty()

    def render(p):
        v, yy = verdict[:p], y[:p]
        flagged = v > 0
        caught, missed = int((flagged & (yy == 1)).sum()), int((~flagged & (yy == 1)).sum())
        false_alarm = int((flagged & (yy == 0)).sum())
        with kpi.container():
            k = st.columns(7)
            k[0].metric("Processed", p)
            k[1].metric("Delivered", int((v == 0).sum()))
            k[2].metric("Warned", int((v == 1).sum()))
            k[3].metric("Blocked", int((v == 2).sum()))
            k[4].metric("Attacks caught", caught)
            k[5].metric("Attacks missed", missed)
            k[6].metric("False alarms", false_alarm)
        xs, fs = [], []
        for e in range(300, p + 1, 50):
            tp = int(((verdict[e - 300:e] > 0) & (y[e - 300:e] == 1)).sum())
            fp = int(((verdict[e - 300:e] > 0) & (y[e - 300:e] == 0)).sum())
            fn = int(((verdict[e - 300:e] == 0) & (y[e - 300:e] == 1)).sum())
            xs.append(e)
            fs.append(2 * tp / max(2 * tp + fp + fn, 1))
        if fs:
            chart_ph.line_chart(pd.DataFrame({"rolling F1 (last 300 emails)": fs}, index=xs), height=180)
        lo = max(0, p - 10)
        rows = []
        for i in range(p - 1, lo - 1, -1):
            e = pipe.emails[ids[i]]
            rows.append({"#": i, "from": e["from"], "snippet": e["body"][:70], "trust": round(float(pipe.trust[ids[i]]), 2),
                         "verdict": VERDICT[int(verdict[i])],
                         "truth": (LABELS.get(e["attack_type"], e["attack_type"]) if reveal else "hidden")})
        if rows:
            df = pd.DataFrame(rows)
            color = {"DELIVER": "#e6f4ea", "WARN": "#fef7e0", "BLOCK": "#fce8e6"}
            feed_ph.dataframe(df.style.apply(lambda r: [f"background-color:{color[r['verdict']]};color:#202124"] * len(r), axis=1),
                              hide_index=True)
        shown = [f"#{a}: {t}" for a, t in events if a <= p][-6:]
        if shown:
            log_ph.info("\n\n".join(shown))
        prog.progress(min(p / n, 1.0))

    if run:
        step = max(1, int(speed * 0.25))
        while pos < n:
            pos = min(n, pos + step)
            st.session_state.pos = pos
            render(pos)
            time.sleep(0.25)
        st.success("Stream complete.")
    else:
        render(pos)

    st.subheader("Inspect a flagged email")
    flagged_ids = [i for i in range(min(pos, n)) if verdict[i] > 0][-15:][::-1]
    if flagged_ids:
        pick = st.selectbox("Recent flagged mail", flagged_ids,
                            format_func=lambda i: f"#{i}  {pipe.emails[ids[i]]['from']}  [{VERDICT[int(verdict[i])]}]")
        gi = int(ids[pick])
        res = pipe.score(gi)
        e = pipe.emails[gi]
        lab = LABEL_OF[VERDICT[int(verdict[pick])]]
        show_html(banner_html(lab, e["from"], e["headers"]["subject"], e["body"], res["trust"].reasons, res["why"],
                                    res["trust"].score, res["risk"], VERDICT[int(verdict[pick])]), height=380, scrolling=True)
    else:
        st.caption("Press Start to stream mail; flagged messages appear here with the reasons.")

# ---------------------------------------------------------------- scenario explorer
with tab_scn:
    kind = st.selectbox("Scenario", list(LABELS), format_func=LABELS.get)
    cands = pipe.test_indices(kind)
    if not cands:
        st.warning("No test emails of this type.")
    else:
        k = st.number_input(f"Example (1-{len(cands)}, unseen future mail)", 1, len(cands), 1)
        i = cands[k - 1]
        e = pipe.emails[i]
        res = pipe.score(i)
        tr = res["trust"]
        fused_v, label, action = policy.decide(tr.score, res["risk"], res["fused"])
        left, right = st.columns(2)
        with left:
            st.subheader("Incoming email")
            st.code(f"From: {e['from']}\nReturn-Path: {e['headers']['return-path']}\nDate: {e['date']}\n"
                    f"Signed: {'yes (d=' + e['sig']['d'] + ')' if e['sig'] else 'NO'}\n\n{e['body'][:600]}")
        with right:
            m1, m2, m3 = st.columns(3)
            m1.metric("Trust score", round(tr.score, 2))
            m2.metric("Risk score", round(res["risk"], 2))
            m3.metric("Fused score", round(fused_v, 2))
            color = {"Trusted": "green", "Suspicious": "orange", "Malicious": "red"}[label]
            st.markdown(f"### :{color}[{label}] → {action}")
            st.caption(f"Ground truth: {'attack' if e['label'] else 'legitimate'} ({e['attack_type']})")
        a, b = st.columns(2)
        with a:
            st.subheader("Why: trust checks")
            st.bar_chart(tr.checks)
            for r in tr.reasons:
                st.warning(r)
        with b:
            st.subheader("Why: top risk drivers (SHAP)")
            if res["why"]:
                st.bar_chart(res["why"])
            else:
                st.success("No risky behavior detected")
        if kind == "takeover":
            st.info("Headline: SPF/DKIM/DMARC pass this mail (Trust is high). Only the behavioral baseline "
                    "(off-profile style, unusual hour, payment request, new recipient) exposes the compromised account.")
        f1, f2 = st.columns(2)
        if f1.button("Report phishing"):
            policy.feedback(fused_v, True); st.rerun()
        if f2.button("Mark as not phishing"):
            policy.feedback(fused_v, False); st.rerun()
        st.caption(f"Adaptive alert threshold now {policy.t_suspicious:.2f}")

# ---------------------------------------------------------------- real .eml
with tab_eml:
    st.write("Real DKIM verification (dkimpy) and SPF (pyspf) on an actual RFC-5322 message, then the same "
             "behavioral model and fusion. Bundled samples are signed with real RSA DKIM keys served from an "
             "offline DNS file; uploaded mail uses live DNS if enabled.")
    src = st.radio("Message", ["Bundled sample", "Upload .eml"], horizontal=True)
    live_dns = st.checkbox("Use live DNS (DKIM key + SPF lookups; sends DNS queries)", False)
    raw, name = None, None
    if src == "Bundled sample":
        files = sorted(glob.glob(os.path.join(ROOT, "demo", "samples", "*.eml")))
        if not files:
            st.warning("Run `python scripts/make_sample_emls.py` to create the samples.")
        else:
            name = st.selectbox("Sample", files, format_func=os.path.basename)
            raw = open(name, "rb").read()
    else:
        up = st.file_uploader("Upload a .eml file", type=["eml"])
        if up:
            raw, name = up.read(), up.name
    if raw:
        recs_path = os.path.join(ROOT, "demo", "samples", "dns_records.json")
        dnsfunc = None
        if not live_dns and os.path.exists(recs_path):
            recs = json.load(open(recs_path))

            def dnsfunc(nm, timeout=5):
                s = nm.decode() if isinstance(nm, bytes) else nm
                v = recs.get(s)
                return v.encode() if v else None
        vr = verify_eml(raw, dnsfunc=dnsfunc, check_spf=live_dns)
        res = pipe.score_external(vr["email"], vr["trust"])
        _, label, action = policy.decide(res["trust"].score, res["risk"], res["fused"])
        m1, m2, m3 = st.columns(3)
        m1.metric("Trust", round(res["trust"].score, 2))
        m2.metric("Risk", round(res["risk"], 2))
        m3.metric("Verdict", f"{label}")
        e = vr["email"]
        show_html(banner_html(label, e["from"], e["headers"]["subject"], e["body"], res["trust"].reasons,
                                    res["why"], res["trust"].score, res["risk"], action), height=380, scrolling=True)
        st.json(vr["details"])
        st.bar_chart(res["trust"].checks)

# ---------------------------------------------------------------- results
with tab_res:
    figs = sorted(glob.glob(os.path.join(ROOT, "docs", "figures", "*.png")))
    caps_p = os.path.join(ROOT, "docs", "figures", "captions.json")
    caps = json.load(open(caps_p)) if os.path.exists(caps_p) else {}
    if not figs:
        st.info("Run `python src/eval/make_figures.py` to generate the report figures.")
    cols = st.columns(2)
    for k, f in enumerate(figs):
        cols[k % 2].image(f, caption=caps.get(os.path.basename(f), os.path.basename(f)))
    for md in ("results.md", "results_multiseed.md"):
        p = os.path.join(ROOT, "docs", md)
        if os.path.exists(p):
            with st.expander(md):
                st.markdown(open(p, encoding="utf-8").read())
