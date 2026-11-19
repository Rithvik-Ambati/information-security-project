"""Write docs/REPORT_ASSETS.md: an index of every figure and table with a caption and the report section it suits."""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")
D = os.path.join(ROOT, "docs")
caps = json.load(open(os.path.join(D, "figures", "captions.json")))
SECTION = {
    "fig01": "Methodology / system design", "fig02": "Dataset", "fig03": "Dataset / evaluation protocol",
    "fig04": "Dataset / threat model", "fig05": "Results: detection performance", "fig06": "Results: detection performance",
    "fig07": "Results: method comparison (mean ± std over seeds)", "fig08": "Results: per-attack analysis",
    "fig09": "Results: error analysis", "fig10": "Results: Trust vs Risk analysis", "fig11": "Explainability",
    "fig12": "Explainability (worked example)", "fig13": "Explainability", "fig14": "Ablation study",
    "fig15": "Fusion / policy engine", "fig16": "Calibration (appendix)", "fig17": "Practicality: false alarms & latency",
    "fig18": "Adaptation: concept drift", "fig19": "Adaptation: feedback thresholds", "fig20": "Robustness / limitations",
    "fig21": "Text-model upgrade (TF-IDF vs DistilBERT)", "live": "Demo / presentation",
}
lines = ["# Report assets\n", "Regenerate everything with `python src/eval/make_figures.py` (after the experiments in the README).\n",
         "## Figures (`docs/figures/`)\n", "| File | Suggested section | Caption |\n|---|---|---|"]
for f in sorted(caps):
    lines.append(f"| `{f}` | {SECTION.get(f.split('_')[0], '')} | {caps[f]} |")
lines += ["\n## Tables (`docs/tables/`, CSV; `main_results.tex` is LaTeX)\n"]
for f in sorted(os.listdir(os.path.join(D, "tables"))):
    lines.append(f"- `{f}`")
lines += ["\n## Result write-ups\n", "- `docs/results.md`: single-run evaluation with bootstrap CIs",
          "- `docs/results_multiseed.md`: mean ± std [95% CI] over seeds",
          "- `docs/results_distilbert.json`: DistilBERT run", "- `docs/stream_results.json`, `docs/adversarial_results.json`"]
open(os.path.join(D, "REPORT_ASSETS.md"), "w", encoding="utf-8").write("\n".join(lines))
print("wrote docs/REPORT_ASSETS.md")
