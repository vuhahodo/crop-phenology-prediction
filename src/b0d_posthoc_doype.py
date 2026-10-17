"""Post-hoc, seed-matched pooled contrast: cyclical DOY-PE vs no DOY-PE."""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

from src.b0d_ablation_metrics import SEEDS

DOY_PE = ["transformer_cyclical_doy", "cba_phenonet", "cba_phenonet_residual", "cba_phenonet_disentangled"]
NO_DOY_PE = ["transformer", "transformer_calendar_gate", "transformer_scalar_gate"]
OUTCOMES = ["macro_f1", "sos_rmse_days", "pos_rmse_days", "eos_rmse_days"]


def main():
    src = Path("results/tables/b0d_ablation_metrics_per_seed.csv")
    records = list(csv.DictReader(src.open(encoding="utf-8-sig", newline="")))
    index = {(r["config"], int(r["seed"])): r for r in records}
    per_seed = []
    for seed in SEEDS:
        for label, configs in (("DOY_PE", DOY_PE), ("NO_DOY_PE", NO_DOY_PE)):
            group = [index[(cfg, seed)] for cfg in configs if (cfg, seed) in index]
            row = {"analysis": "POST_HOC", "seed": seed, "group": label,
                   "n_configs": len(group), "configs": ";".join(r["config"] for r in group)}
            for metric in OUTCOMES:
                vals = [float(r[metric]) for r in group if r[metric] not in ("", "nan", "NaN")]
                row[f"{metric}_mean"] = float(np.mean(vals)) if vals else float("nan")
            per_seed.append(row)
    out = Path("results/tables"); out.mkdir(parents=True, exist_ok=True)
    with (out / "b0d_posthoc_doype_per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_seed[0])); w.writeheader(); w.writerows(per_seed)
    by = {(r["group"], int(r["seed"])): r for r in per_seed}
    tests = []
    for metric in OUTCOMES:
        a = np.array([by[("DOY_PE", s)][f"{metric}_mean"] for s in SEEDS], dtype=float)
        b = np.array([by[("NO_DOY_PE", s)][f"{metric}_mean"] for s in SEEDS], dtype=float)
        keep = np.isfinite(a) & np.isfinite(b); a, b = a[keep], b[keep]
        result = ttest_rel(a, b) if len(a) >= 2 else None
        tests.append({"analysis": "POST_HOC", "metric": metric, "n_matched_seeds": len(a),
                      "doy_pe_configs": ";".join(DOY_PE), "no_doy_pe_configs": ";".join(NO_DOY_PE),
                      "doy_pe_mean": float(np.mean(a)) if len(a) else float("nan"),
                      "no_doy_pe_mean": float(np.mean(b)) if len(b) else float("nan"),
                      "mean_diff_doy_pe_minus_no": float(np.mean(a-b)) if len(a) else float("nan"),
                      "t_stat": float(result.statistic) if result else float("nan"),
                      "p_raw": float(result.pvalue) if result else float("nan")})
    with (out / "b0d_posthoc_doype_paired_tests.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0])); w.writeheader(); w.writerows(tests)
    print("results/tables/b0d_posthoc_doype_per_seed.csv; results/tables/b0d_posthoc_doype_paired_tests.csv")


if __name__ == "__main__":
    main()
