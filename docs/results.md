# Evaluation results

Backend: `tfidf`  |  test emails: 5787  |  attack rate: 0.20
Chronological split: history -> train -> test (future). Thresholds and the fusion strategy are chosen on a validation window (max F1 at FPR<=1%), never on test. `Prec@1% prev.` = precision if attacks were only 1% of mail, computed from the measured TPR/FPR.

## Overall (test window)

| Method | thr | Acc | Prec | Recall | F1 | FPR | AUC | Prec@1% prev. |
|---|---|---|---|---|---|---|---|---|
| A. Content-only (TF-IDF+LR) | 0.24 | 0.992 | 0.966 | 0.996 | 0.981 | 0.009 | 0.999 | 0.531 |
| B. Auth-only (crypto Trust) | 0.84 | 0.842 | 1.000 | 0.222 | 0.363 | 0.000 | 0.821 | 1.000 |
| C. Simple combo max(A, B) | 0.83 | 0.941 | 1.000 | 0.711 | 0.831 | 0.000 | 0.978 | 1.000 |
| Ablation: behavior-only (no text) | 0.46 | 0.957 | 0.952 | 0.834 | 0.889 | 0.011 | 0.990 | 0.437 |
| Ablation: risk (behavior+content) | 0.02 | 0.971 | 0.992 | 0.865 | 0.924 | 0.002 | 0.932 | 0.834 |
| FUSION (selected: GB[risk, trust checks]) | 0.28 | 0.985 | 0.986 | 0.941 | 0.963 | 0.003 | 0.974 | 0.732 |

## Recall per attack type

| Method | spoof | takeover | lookalike | tamper |
|---|---|---|---|---|
| A. Content-only (TF-IDF+LR) | 1.00 | 0.98 | 1.00 | 1.00 |
| B. Auth-only (crypto Trust) | 0.89 | 0.00 | 0.00 | 0.00 |
| C. Simple combo max(A, B) | 0.99 | 0.30 | 0.79 | 0.76 |
| Ablation: behavior-only (no text) | 1.00 | 0.63 | 0.98 | 0.71 |
| Ablation: risk (behavior+content) | 0.89 | 0.76 | 0.91 | 0.89 |
| FUSION (selected: GB[risk, trust checks]) | 1.00 | 0.76 | 1.00 | 1.00 |

## Headline scenario: account takeover (valid signature) vs legitimate

| Method | thr | Acc | Prec | Recall | F1 | FPR | AUC | Prec@1% prev. |
|---|---|---|---|---|---|---|---|---|
| A. Content-only (TF-IDF+LR) | 0.24 | 0.991 | 0.874 | 0.983 | 0.925 | 0.009 | 0.999 | 0.527 |
| B. Auth-only (crypto Trust) | 0.84 | 0.941 | 0.000 | 0.000 | 0.000 | 0.000 | 0.463 | 0.000 |
| C. Simple combo max(A, B) | 0.83 | 0.959 | 1.000 | 0.303 | 0.466 | 0.000 | 0.948 | 1.000 |
| Ablation: behavior-only (no text) | 0.46 | 0.968 | 0.786 | 0.634 | 0.702 | 0.011 | 0.979 | 0.371 |
| Ablation: risk (behavior+content) | 0.02 | 0.984 | 0.965 | 0.762 | 0.852 | 0.002 | 0.880 | 0.816 |
| FUSION (selected: GB[risk, trust checks]) | 0.28 | 0.983 | 0.932 | 0.762 | 0.839 | 0.003 | 0.893 | 0.689 |

## False-alarm rate by kind of legitimate mail (selected fusion)

- signed legitimate: 0.004 (n=4251)
- unsigned legitimate (lists/forwarders): 0.000 (n=358)

## Fusion candidates (validation window)

| Candidate | thr | val F1 | val FPR |
|---|---|---|---|
| max(risk, 1-trust) | 0.84 | 0.842 | 0.000 |
| max(risk, integrity-rule) | 0.02 | 0.760 | 0.077 |
| LR[risk, 1-trust] | 0.68 | 0.842 | 0.000 |
| LR[risk, trust checks] | 0.62 | 0.938 | 0.003 |
| GB[risk, trust checks] | 0.28 | 0.944 | 0.000 |

Chosen: **GB[risk, trust checks]** (simplest candidate within 0.005 val-F1 of the best).

## Temporal drift (F1 over 3 consecutive future chunks)

| Model | chunk 1 | chunk 2 | chunk 3 |
|---|---|---|---|
| static risk | 0.933 | 0.915 | 0.924 |
| retrained risk | 0.971 | 0.986 | 0.972 |
| content-only static | 0.975 | 0.979 | 0.989 |

## Latency per email

Trust verification: 0.11 ms | Risk scoring: 21.70 ms

> Caveats: legitimate mail = Enron sent folders + SpamAssassin ham (dates spread over the Enron span); phishing = Nazario 2015 mass-phishing text (near-duplicates removed, train/test pools disjoint) plus synthetic BEC text with template-level holdout. The authentication layer and attack injection are synthetic by design; Enron (2001) vs Nazario (2015) era/style differences still make mass phishing easier to separate than in the wild. Single run unless aggregated by multiseed.py.