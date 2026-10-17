"""B0d ablation metrics and paired-seed Holm adjustment within endpoints."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel
from sklearn.metrics import f1_score

from src.make_tables import CONFIGS, SEEDS, decode_milestones, load_preds

BASE = "transformer"
VARIANTS = ["transformer_cyclical_doy", "transformer_calendar_gate", "transformer_scalar_gate",
            "cba_phenonet", "cba_phenonet_residual", "cba_phenonet_disentangled"]
MILESTONES = ("sos", "pos", "eos")


def metrics(data, config, seed):
    yt, yp, mask, doy, ndvi = (data[k] for k in ("y_true", "y_pred", "mask", "doy", "ndvi"))
    f1 = float(f1_score(yt[mask], yp[mask], average="macro", zero_division=0))
    err = defaultdict(list); true_n = defaultdict(int); detected_n = defaultdict(int); tp = defaultdict(int)
    for i in range(len(yt)):
        L = int(mask[i].sum())
        a = decode_milestones(yt[i, :L], doy[i, :L], ndvi[i, :L])
        b = decode_milestones(yp[i, :L], doy[i, :L], ndvi[i, :L])
        for j, name in enumerate(MILESTONES):
            true_n[name] += a[j] is not None
            detected_n[name] += b[j] is not None
            if a[j] is not None and b[j] is not None:
                tp[name] += 1
                err[name].append((int(a[j]) - int(b[j])) ** 2)
    row = {"config": config, "seed": seed, "macro_f1": f1}
    for m in MILESTONES:
        row[f"{m}_rmse_days"] = float(np.sqrt(np.mean(err[m]))) if err[m] else float("nan")
        row[f"{m}_recall"] = tp[m] / true_n[m] if true_n[m] else float("nan")
        row[f"{m}_detected"] = int(detected_n[m])
        row[f"{m}_true"] = int(true_n[m])
    return row


def holm(pvals):
    n = len(pvals); order = np.argsort(pvals); adj = np.empty(n); running = 0.0
    for rank, ix in enumerate(order):
        running = max(running, min(1.0, (n - rank) * pvals[ix]))
        adj[ix] = running
    return adj


def main():
    rows = []
    configs = [BASE] + VARIANTS
    for cfg in configs:
        for seed in SEEDS:
            d = load_preds(cfg, seed)
            if d is not None:
                rows.append(metrics(d, cfg, seed))
    out = Path("results/tables"); out.mkdir(parents=True, exist_ok=True)
    per = out / "b0d_ablation_metrics_per_seed.csv"
    with per.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    by = {(r["config"], int(r["seed"])): r for r in rows}
    outcomes = ["macro_f1"] + [f"{m}_{q}" for m in MILESTONES for q in ("rmse_days", "recall", "detected")]
    tests = []
    for metric in outcomes:
        family = []
        for cfg in VARIANTS:
            seeds = [s for s in SEEDS if (BASE, s) in by and (cfg, s) in by]
            a = np.array([float(by[(cfg, s)][metric]) for s in seeds]); b = np.array([float(by[(BASE, s)][metric]) for s in seeds])
            good = np.isfinite(a) & np.isfinite(b); a, b = a[good], b[good]
            p = float(ttest_rel(a, b).pvalue) if len(a) >= 2 else float("nan")
            family.append({"config": cfg, "metric": metric, "n_seeds": len(a),
                           "mean_diff_vs_transformer": float(np.mean(a-b)) if len(a) else float("nan"), "p_raw": p})
        finite_ix = [i for i, x in enumerate(family) if np.isfinite(x["p_raw"])]
        adjusted = holm(np.array([family[i]["p_raw"] for i in finite_ix])) if finite_ix else []
        for i, x in enumerate(family):
            p_adj = float(adjusted[finite_ix.index(i)]) if i in finite_ix else float("nan")
            tests.append({**x, "family": metric, "family_n": len(finite_ix), "p_adj_holm": p_adj,
                          "p_adj_lt_0_05": bool(p_adj < 0.05) if np.isfinite(p_adj) else "NA"})
    with (out / "b0d_ablation_paired_tests.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0])); w.writeheader(); w.writerows(tests)
    print(f"{per}; results/tables/b0d_ablation_paired_tests.csv")


if __name__ == "__main__":
    main()
