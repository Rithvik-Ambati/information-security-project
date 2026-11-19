# Report assets

Regenerate everything with `python src/eval/make_figures.py` (after the experiments in the README).

## Figures (`docs/figures/`)

| File | Suggested section | Caption |
|---|---|---|
| `fig01_architecture.png` | Methodology / system design | System architecture: parallel Trust and Risk modules feed an adaptive policy engine with a feedback loop. |
| `fig02_dataset_composition.png` | Dataset | Dataset composition by split and attack type; share of legitimate mail that is unsigned. |
| `fig03_timeline.png` | Dataset / evaluation protocol | Chronological layout: models are fitted on early mail, thresholds chosen on a validation slice, results reported on the latest mail. |
| `fig04_send_hours.png` | Dataset / threat model | Half of simulated attacks use unusual hours, so timing alone is only a partial signal. |
| `fig05_roc.png` | Results: detection performance | ROC curves on the latest-time test window (x axis zoomed to low false-positive rates). |
| `fig06_pr.png` | Results: detection performance | Precision-recall curves; right panel re-weights precision to a realistic 1% attack prevalence. |
| `fig07_method_comparison.png` | Results: method comparison (mean ± std over seeds) | Method comparison across seeds (thresholds and fusion chosen on validation only). |
| `fig08_per_attack_heatmap.png` | Results: per-attack analysis | Per-attack recall: crypto catches spoof/tamper/lookalike, behavior is needed for valid-signature takeover. |
| `fig09_confusion_matrices.png` | Results: error analysis | Confusion matrices on the test window for content-only, auth-only and the fusion. |
| `fig10_score_distributions.png` | Results: Trust vs Risk analysis | Left: fused-score distributions; right: the Trust/Risk plane showing that takeover lives at high risk but high trust. |
| `fig11_shap_summary.png` | Explainability | SHAP summary (beeswarm) of the behavioral risk model. |
| `fig12_shap_waterfall_takeover.png` | Explainability (worked example) | SHAP waterfall for one account-takeover email: why it was scored high-risk despite a valid signature. |
| `fig13_feature_importance.png` | Explainability | Top-15 features of the behavioral risk model. |
| `fig14_ablation.png` | Ablation study | Ablation: behavior-only, content-only, crypto-only and fused configurations (overall F1 and takeover recall). |
| `fig15_fusion_candidates.png` | Fusion / policy engine | Candidate fusion strategies; the selected one (blue) is the simplest within 0.005 F1 of the best. |
| `fig16_calibration.png` | Calibration (appendix) | Calibration of the scores on the test window. |
| `fig17_fpr_and_latency.png` | Practicality: false alarms & latency | False-alarm rate by legitimate-mail type and per-email latency. |
| `fig18_drift_rolling_f1.png` | Adaptation: concept drift | Rolling F1 on a stream with two distribution shifts; dotted lines = drift alarms, triangles = retraining. |
| `fig19_threshold_adaptation.png` | Adaptation: feedback thresholds | Alert-threshold adaptation from user feedback: fixed vs global vs per-account thresholds. |
| `fig20_adversarial_robustness.png` | Robustness / limitations | Robustness to an adaptive attacker (keyword evasion, padding, style/timing/recipient mimicry). |
| `fig21_tfidf_vs_distilbert.png` | Text-model upgrade (TF-IDF vs DistilBERT) | Effect of replacing the TF-IDF text score with DistilBERT embeddings (single run, same split). |
| `live_simulation.gif` | Demo / presentation | Animated replay of the live stream with drift-triggered retraining. |

## Tables (`docs/tables/`, CSV; `main_results.tex` is LaTeX)

- `adaptation_strategies.csv`
- `adversarial_takeover_recall.csv`
- `dataset_composition.csv`
- `feature_importance.csv`
- `fusion_candidates.csv`
- `main_results.csv`
- `main_results.tex`
- `method_comparison_mean.csv`
- `per_attack_recall.csv`
- `tfidf_vs_distilbert.csv`
- `threshold_adaptation.csv`

## Result write-ups

- `docs/results.md`: single-run evaluation with bootstrap CIs
- `docs/results_multiseed.md`: mean ± std [95% CI] over seeds
- `docs/results_distilbert.json`: DistilBERT run
- `docs/stream_results.json`, `docs/adversarial_results.json`