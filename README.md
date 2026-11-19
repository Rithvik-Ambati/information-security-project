# AI-Driven Adaptive Email Security Framework
Using Behavioral Risk Analysis and Cryptographic Trust Verification — research prototype / gateway-level filter.

Two modules run in parallel on every message and a policy engine fuses them:
- **Cryptographic Trust** (graded 0–1): DKIM signature + body hash, From/signing-domain alignment, trust-on-first-use key continuity (flags key/selector changes), header consistency, lookalike-domain check. Real DKIM/SPF verification on `.eml` files via `dkimpy`/`pyspf`.
- **Behavioral Risk** (0–1): sender profile (hours, recipients, history), metadata, content, and **writing-style similarity to the sender's past mail**, plus an IsolationForest anomaly score; optional DistilBERT text features. SHAP explains each verdict.
- **Adaptive policy engine**: fusion strategy and thresholds selected on a validation window, feedback-driven drift detection and retraining, banner explanations. Headline target: **authenticated-but-out-of-character** mail (a compromised account with a valid signature).

## Quick start
```
pip install -r requirements.txt
python scripts/build_dataset.py --demo        # tiny synthetic corpus, enough to run the demo
streamlit run demo/app.py
```
For the real evaluation (Enron + SpamAssassin + Nazario):
```
python scripts/download_data.py enron nazario spamassassin   # Enron ~420 MB; antivirus may flag Nazario files
python scripts/build_dataset.py --max-ham 15000
python src/eval/full_eval.py                  # docs/results.md (incl. bootstrap CIs)
python src/eval/multiseed.py --seeds 1 2 3 4 5 6 7 8 9 10    # docs/results_multiseed.md
python src/adapt/stream.py                    # drift / retraining / threshold experiments
python src/eval/adversarial.py                # evasion robustness
python src/eval/make_figures.py               # docs/figures, docs/tables, live_simulation.gif
python scripts/build_report_index.py          # docs/REPORT_ASSETS.md
python scripts/make_sample_emls.py            # real DKIM-signed demo .eml files
python scripts/analyze_eml.py demo/samples/*.eml
```
DistilBERT: `EMAIL_DATA_DIR=<copy of data/processed> python src/eval/full_eval.py --backend transformer` (downloads ~270 MB on first use).

## Demo (what to show)
`streamlit run demo/app.py` has four tabs:
1. **Live simulation**: replays mail in time order with counters (delivered / warned / blocked, attacks caught / missed, false alarms), rolling F1, an event log (distribution shift, drift alarm, model retrained), a colour-coded inbox, and a Gmail-style warning banner with reasons for any flagged message. Choose static, scheduled or drift-triggered adaptation.
2. **Scenario explorer**: spoof, takeover (valid signature), lookalike, tampering, with trust checks, SHAP drivers and feedback buttons.
3. **Analyze a real .eml**: real DKIM verification on bundled samples signed with real RSA keys (or upload your own; optional live DNS/SPF).
4. **Results & figures**: all report figures and tables.

Suggested 4-minute flow: legitimate mail delivered → spoof blocked → **account takeover** (all SPF/DKIM checks pass, Trust high, behavior exposes it) → start the live stream and point to the drift event and retraining → open a flagged email's banner → show the adversarial-robustness figure as the honest limitation.

## Results (test window, 10 seeds, thresholds/fusion chosen on validation only)
| Method | F1 | Recall | FPR | Takeover recall |
|---|---|---|---|---|
| Content-only (TF-IDF+LR) | 0.959 ± 0.010 | 0.943 | 0.006 | 0.952 |
| Auth-only (crypto Trust) | 0.380 ± 0.026 | 0.236 | 0.002 | 0.000 |
| Simple combo | 0.835 ± 0.053 | 0.725 | 0.002 | 0.464 |
| Behavior-only (no text) | 0.876 ± 0.011 | 0.813 | 0.011 | 0.590 |
| Risk (behavior + content) | 0.903 ± 0.039 | 0.829 | 0.001 | 0.769 |
| **Fusion (selected)** | **0.963 ± 0.013** | 0.942 | 0.004 | 0.782 |

Full tables with 95% confidence intervals: `docs/results_multiseed.md`, `docs/results.md`.

- **Crypto alone misses the key threat**: valid-signature takeover recall is 0.00; spoof/tamper/lookalike are where trust helps (fusion reaches ~1.00 on them).
- **Adaptation** (stream with two distribution shifts, 30% feedback): static F1 0.957; scheduled retraining 0.976; drift-triggered 0.977 (same number of retrains). Detector sensitivity sweep (λ=1/2/4) trades retrains (6/4/1) against F1 (0.981/0.977/0.968). Feedback-driven *threshold* adaptation did **not** help (missed attacks score near zero, so thresholds cannot recover them; global adaptation caused more false alarms); labels help by retraining the model.
- **DistilBERT** (single run): replacing the TF-IDF text score lifts risk F1 0.924→0.974 and takeover recall 0.76→0.97 (`fig21`).
- **Adversarial test**: an attacker who pads with benign text and copies the victim's style cuts fusion's takeover recall from 0.76 to 0.30 (behavior-only falls to 0.05; content-only 0.98→0.75). Timing/recipient mimicry alone has no effect.

## Honest limitations
- The authentication layer and attack injection are **synthetic by design** (public corpora have no real DKIM or account takeovers). Real DKIM is demonstrated on generated `.eml` files, not on real-world compromised accounts.
- Legit mail is Enron (2001) + SpamAssassin (2002–03, dates spread over Enron's span); phishing is Nazario (2015) + a hand-written combinatorial BEC grammar (not LLM-generated). Era/style gaps make attacks easier to separate than in the wild, which is why content-only is strong; the adversarial test is the harder check.
- Absolute numbers are optimistic. Seeds vary attack placement/text/keys, not the underlying corpora. Precision at a realistic 1% attack prevalence is only ~0.6–0.8 for all methods.
- Behavior features alone are brittle to style mimicry; DistilBERT and richer sender modelling are the clear next steps.
- Key-store learning and sender profiles update from confirmed-legitimate mail (ground truth in the simulation).

## Layout
- `src/crypto/` `trust.py` (graded Trust), `real_auth.py` (real DKIM/SPF/alignment on `.eml`)
- `src/behavior/` `features.py` (profile/metadata/content/style), `model.py`, `embed.py` (DistilBERT)
- `src/policy/` `engine.py` (thresholds, feedback), `fusion.py` (validation-selected fusion)
- `src/adapt/` `drift.py` (Page-Hinkley), `stream.py` (retraining strategies, per-account thresholds)
- `src/eval/` `full_eval.py`, `multiseed.py`, `adversarial.py`, `context.py`, `make_figures.py`
- `src/data/` parsing, signing/attack generation, `bec.py`; `scripts/` data download/build, sample `.eml`, `.eml` analyzer
- `demo/` Streamlit app and sample emails; `docs/` results, figures, tables, `REPORT_ASSETS.md`
