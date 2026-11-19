# Multi-seed results (10 seeds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

Mean ± sample std [95% t-interval over seeds]. Thresholds/fusion chosen on validation per seed. Prec@1% = precision at 1% attack prevalence.

| Method | F1 | Recall | FPR | AUC | Prec@1% |
|---|---|---|---|---|---|
| A. Content-only (TF-IDF+LR) | 0.959 ± 0.010 [0.951, 0.966] | 0.943 ± 0.019 [0.930, 0.957] | 0.006 ± 0.002 [0.005, 0.008] | 0.997 ± 0.001 [0.997, 0.998] | 0.618 ± 0.077 [0.563, 0.673] |
| B. Auth-only (crypto Trust) | 0.380 ± 0.026 [0.361, 0.399] | 0.236 ± 0.021 [0.222, 0.251] | 0.002 ± 0.002 [0.000, 0.003] | 0.823 ± 0.007 [0.818, 0.828] | 0.710 ± 0.280 [0.510, 0.910] |
| C. Simple combo max(A, B) | 0.835 ± 0.053 [0.797, 0.873] | 0.725 ± 0.081 [0.667, 0.783] | 0.002 ± 0.002 [0.000, 0.003] | 0.977 ± 0.007 [0.973, 0.982] | 0.848 ± 0.157 [0.735, 0.960] |
| Ablation: behavior-only (no text) | 0.876 ± 0.011 [0.868, 0.884] | 0.813 ± 0.019 [0.799, 0.826] | 0.011 ± 0.002 [0.009, 0.012] | 0.989 ± 0.002 [0.988, 0.991] | 0.437 ± 0.042 [0.408, 0.467] |
| Ablation: risk (behavior+content) | 0.903 ± 0.039 [0.875, 0.931] | 0.829 ± 0.063 [0.784, 0.874] | 0.001 ± 0.001 [0.001, 0.002] | 0.920 ± 0.035 [0.895, 0.946] | 0.880 ± 0.065 [0.833, 0.927] |
| FUSION (selected) | 0.963 ± 0.013 [0.953, 0.972] | 0.942 ± 0.024 [0.925, 0.959] | 0.004 ± 0.001 [0.003, 0.004] | 0.974 ± 0.011 [0.966, 0.982] | 0.727 ± 0.064 [0.681, 0.773] |

## Takeover-only recall (valid signature)

| Method | recall |
|---|---|
| A. Content-only (TF-IDF+LR) | 0.952 ± 0.021 [0.937, 0.967] |
| B. Auth-only (crypto Trust) | 0.000 ± 0.000 [0.000, 0.000] |
| C. Simple combo max(A, B) | 0.464 ± 0.152 [0.355, 0.573] |
| Ablation: behavior-only (no text) | 0.590 ± 0.027 [0.571, 0.610] |
| Ablation: risk (behavior+content) | 0.769 ± 0.074 [0.716, 0.822] |
| FUSION (selected) | 0.782 ± 0.084 [0.722, 0.843] |

## Selected fusion per seed

- seed 1: LR[risk, trust checks]
- seed 2: LR[risk, trust checks]
- seed 3: GB[risk, trust checks]
- seed 4: LR[risk, trust checks]
- seed 5: LR[risk, trust checks]
- seed 6: LR[risk, trust checks]
- seed 7: LR[risk, trust checks]
- seed 8: LR[risk, trust checks]
- seed 9: GB[risk, trust checks]
- seed 10: LR[risk, trust checks]