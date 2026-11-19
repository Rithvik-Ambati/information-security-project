"""Repeat build + evaluation over several seeds; report mean +/- std. Writes docs/results_multiseed.md.
Run:  python src/eval/multiseed.py [--seeds 1 2 3] [--max-ham 15000]
Each seed changes attack placement/text choice, key rotations, unsigned-legit sampling and SpamAssassin sampling
(the underlying Enron/Nazario/SpamAssassin corpora are the same), so this measures pipeline variance, not
dataset-sampling variance."""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
ap.add_argument("--max-ham", type=int, default=15000)
args = ap.parse_args()

ap2 = argparse.ArgumentParser(add_help=False)
runs = []
for s in ([] if os.environ.get("AGGREGATE_ONLY") else args.seeds):
    out = os.path.join(ROOT, "data", f"processed_seed{s}")
    env = dict(os.environ, EMAIL_DATA_DIR=out, PYTHONIOENCODING="utf-8")
    print(f"== seed {s}: build", flush=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "build_dataset.py"), "--max-ham",
                    str(args.max_ham), "--seed", str(s), "--out", out], check=True, env=env, stdout=subprocess.DEVNULL)
    print(f"== seed {s}: evaluate", flush=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "src", "eval", "full_eval.py")], check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    runs.append(json.load(open(os.path.join(out, "results.json"))))


from scipy import stats


if os.environ.get("AGGREGATE_ONLY"):   # re-aggregate existing per-seed results without rebuilding
    runs = [json.load(open(os.path.join(ROOT, "data", f"processed_seed{s}", "results.json"))) for s in args.seeds]


def ms(vals):
    """mean ± std [95% t-interval over seeds]"""
    v = np.array(vals, dtype=float)
    if len(v) < 2:
        return f"{v.mean():.3f}"
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():.3f} ± {v.std(ddof=1):.3f} [{v.mean() - h:.3f}, {v.mean() + h:.3f}]"


# method names embed the selected fusion; normalise to a stable label
def label(k):
    return "FUSION (selected)" if k.startswith("FUSION") else k


names = [label(k) for k in runs[0]["methods"]]
rows = {label(k): [] for k in runs[0]["methods"]}
for r in runs:
    for k, v in r["methods"].items():
        rows[label(k)].append(v)

lines = [f"# Multi-seed results ({len(runs)} seeds: {args.seeds})\n",
         "Mean ± sample std [95% t-interval over seeds]. Thresholds/fusion chosen on validation per seed. "
         "Prec@1% = precision at 1% attack prevalence.\n",
         "| Method | F1 | Recall | FPR | AUC | Prec@1% |\n|---|---|---|---|---|---|"]
for n in names:
    lines.append(f"| {n} | " + " | ".join(ms([v[c] for v in rows[n]]) for c in ("F1", "R", "FPR", "AUC", "PPV1")) + " |")

lines += ["\n## Takeover-only recall (valid signature)\n", "| Method | recall |\n|---|---|"]
for k in runs[0]["per_class"]:
    lines.append(f"| {label(k)} | " + ms([r["per_class"][k2]["takeover"] for r in runs for k2 in r["per_class"]
                                          if label(k2) == label(k)]) + " |")
lines += ["\n## Selected fusion per seed\n"] + [f"- seed {s}: {r['fusion']}" for s, r in zip(args.seeds, runs)]
txt = "\n".join(lines)
print(txt)
open(os.path.join(ROOT, "docs", "results_multiseed.md"), "w", encoding="utf-8").write(txt)
