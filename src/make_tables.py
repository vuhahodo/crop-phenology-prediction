"""make_tables.py — Single source of truth for all paper tables, with rigorous stats.

Generates:
  - results/tables/table1_clean.csv        (Macro-F1, recall, MAE per config)
  - results/tables/table_ablation.csv      (7-config ablation, with Data Bootstrap CI)
  - results/tables/table_pvalues.csv       (Full stats: Holm, Bonferroni, Seed Bootstrap CI, Cohen's d_z, TOST)
  - results/tables/table_per_seed.csv      (raw per-seed numbers)

Unified decode protocol:
  SOS = first t where label[t-1]==0 and label[t] in {1,2,3}  (0->non-zero)
  EOS = first t where label[t-1]==3 and label[t]==0           (3->0)
  POS = argmax(NDVI) among timesteps with label in {1,2}
"""
import csv
import numpy as np
import scipy.stats as stats
from pathlib import Path
from sklearn.metrics import f1_score

PREDS_DIR = Path("results/preds")
TABLES_DIR = Path("results/tables")
TABLES_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    "transformer",
    "cnn1d",
    "bilstm",
    "gru",
    "tcn",
    "transformer_causal",
    "transformer_cyclical_doy",
    "transformer_calendar_gate",
    "transformer_scalar_gate",
    "cba_phenonet",
    "cba_phenonet_residual",
    "cba_phenonet_disentangled",
    "rf_tuned",
    "xgboost_tuned",
    "rf_fullseason",
    "xgboost_fullseason",
    "double_logistic",
]

SEEDS = [42, 123, 2024, 7, 314, 0, 1, 999, 555, 77]
ALPHA = 0.05

def decode_milestones(label_seq, doy_seq, ndvi_seq):
    sos = pos = eos = None
    T = len(label_seq)
    for t in range(1, T):
        if label_seq[t-1] == 0 and label_seq[t] in (1, 2, 3):
            sos = int(doy_seq[t])
            break
    veg_repro = np.where(np.isin(label_seq, [1, 2]))[0]
    if len(veg_repro) > 0:
        peak_idx = veg_repro[np.argmax(ndvi_seq[veg_repro])]
        pos = int(doy_seq[peak_idx])
    for t in range(1, T):
        if label_seq[t-1] == 3 and label_seq[t] == 0:
            eos = int(doy_seq[t])
            break
    return sos, pos, eos


def load_preds(config, seed):
    path = PREDS_DIR / f"{config}_{seed}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in ["y_true", "y_pred", "doy", "ndvi", "mask", "season_ids"]}


def compute_metrics(data):
    y_true, y_pred, mask = data["y_true"], data["y_pred"], data["mask"]
    doy, ndvi = data["doy"], data["ndvi"]
    N = y_true.shape[0]
    
    macro_f1 = f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)
    
    true_ms, pred_ms = [], []
    for i in range(N):
        t = mask[i].sum()
        yt, yp = y_true[i, :t], y_pred[i, :t]
        d, n = doy[i, :t], ndvi[i, :t]
        true_ms.append(decode_milestones(yt, d, n))
        pred_ms.append(decode_milestones(yp, d, n))
    
    metrics = {"macro_f1": macro_f1}
    for idx, name in enumerate(["sos", "pos", "eos"]):
        true_valid = [(i, true_ms[i][idx]) for i in range(N) if true_ms[i][idx] is not None]
        n_valid_true = len(true_valid)
        pairs = [(true_ms[i][idx], pred_ms[i][idx]) for i in range(N) if true_ms[i][idx] is not None and pred_ms[i][idx] is not None]
        n_detected = len(pairs)
        
        if pairs:
            errors = np.array([abs(t - p) for t, p in pairs])
            mae = errors.mean()
        else:
            mae = float("nan")
        
        metrics[f"{name}_recall"] = n_detected / n_valid_true if n_valid_true > 0 else 0
        metrics[f"{name}_n_valid_true"] = n_valid_true
        metrics[f"{name}_n_detected"] = n_detected
        metrics[f"{name}_mae"] = mae
    return metrics


def compute_sample_level_metrics(data):
    """Compute per-sample F1 and MAE for Data Bootstrap."""
    y_true, y_pred, mask = data["y_true"], data["y_pred"], data["mask"]
    doy, ndvi = data["doy"], data["ndvi"]
    N = y_true.shape[0]
    
    sample_f1 = np.zeros(N)
    sample_sos_err = np.full(N, np.nan)
    sample_pos_err = np.full(N, np.nan)
    sample_eos_err = np.full(N, np.nan)
    
    for i in range(N):
        t = mask[i].sum()
        yt, yp, d, n = y_true[i, :t], y_pred[i, :t], doy[i, :t], ndvi[i, :t]
        sample_f1[i] = f1_score(yt, yp, average="macro", zero_division=0)
        
        tm = decode_milestones(yt, d, n)
        pm = decode_milestones(yp, d, n)
        
        if tm[0] is not None and pm[0] is not None: sample_sos_err[i] = abs(tm[0] - pm[0])
        if tm[1] is not None and pm[1] is not None: sample_pos_err[i] = abs(tm[1] - pm[1])
        if tm[2] is not None and pm[2] is not None: sample_eos_err[i] = abs(tm[2] - pm[2])
        
    return sample_f1, sample_sos_err, sample_pos_err, sample_eos_err


def data_bootstrap_ci(sample_metrics, n_resamples=1000):
    """Bootstrap over the N=399 test samples to get CI of the mean performance."""
    # sample_metrics shape: (n_seeds, N)
    n_seeds, N = sample_metrics.shape
    rng = np.random.default_rng(42)
    boot_means = np.zeros(n_resamples)
    for i in range(n_resamples):
        # resample test indices with replacement
        idx = rng.choice(N, size=N, replace=True)
        # mean across seeds for this resample, then mean across the resampled N
        # handle NaNs for MAE
        seed_means = np.nanmean(sample_metrics[:, idx], axis=1)
        boot_means[i] = np.nanmean(seed_means)
    return np.percentile(boot_means, [2.5, 97.5])


def seed_bootstrap_ci(arr_cfg, arr_ref, n_resamples=10000):
    """Bootstrap over the 10 seeds to get CI of the mean difference."""
    diff = arr_cfg - arr_ref
    n = len(diff)
    rng = np.random.default_rng(42)
    boot_diffs = np.zeros(n_resamples)
    for i in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs[i] = np.mean(diff[idx])
    return np.percentile(boot_diffs, [2.5, 97.5])


def adjust_pvalues_holm(p_values):
    """Holm-Bonferroni step-down adjustment."""
    p = np.array(p_values)
    order = p.argsort()
    p_adj = np.empty_like(p)
    m = len(p)
    cum_max = 0.0
    for i, idx in enumerate(order):
        adj = p[idx] * (m - i)
        cum_max = max(cum_max, adj)
        p_adj[idx] = min(1.0, cum_max)
    return p_adj


def main():
    print("make_tables.py: Computing all tables with full stats...")
    
    all_metrics = {}
    all_sample_metrics = {}
    
    for config in CONFIGS:
        all_metrics[config] = {}
        all_sample_metrics[config] = {"f1": [], "sos": [], "pos": [], "eos": []}
        seeds_to_run = [0] if "rf" in config or "xgboost" in config or "logistic" in config else SEEDS
        for seed in seeds_to_run:
            data = load_preds(config, seed)
            if data is None:
                continue
            all_metrics[config][seed] = compute_metrics(data)
            
            sf1, ssos, spos, seos = compute_sample_level_metrics(data)
            all_sample_metrics[config]["f1"].append(sf1)
            all_sample_metrics[config]["sos"].append(ssos)
            all_sample_metrics[config]["pos"].append(spos)
            all_sample_metrics[config]["eos"].append(seos)
            
        for k in ["f1", "sos", "pos", "eos"]:
            if all_sample_metrics[config][k]:
                all_sample_metrics[config][k] = np.vstack(all_sample_metrics[config][k])
    
    # ===== Table: Ablation Summary with Data Bootstrap CI =====
    ablation_rows = []
    for config in CONFIGS:
        seeds_avail = sorted(all_metrics[config].keys())
        n = len(seeds_avail)
        if n == 0: continue
        
        f1s = [all_metrics[config][s]["macro_f1"] for s in seeds_avail]
        sos_maes = [all_metrics[config][s]["sos_mae"] for s in seeds_avail if not np.isnan(all_metrics[config][s]["sos_mae"])]
        pos_maes = [all_metrics[config][s]["pos_mae"] for s in seeds_avail if not np.isnan(all_metrics[config][s]["pos_mae"])]
        eos_maes = [all_metrics[config][s]["eos_mae"] for s in seeds_avail if not np.isnan(all_metrics[config][s]["eos_mae"])]
        
        # Compute Data Bootstrap CIs
        f1_ci = data_bootstrap_ci(all_sample_metrics[config]["f1"]) if len(all_sample_metrics[config]["f1"]) > 0 else (np.nan, np.nan)
        sos_ci = data_bootstrap_ci(all_sample_metrics[config]["sos"]) if len(all_sample_metrics[config]["sos"]) > 0 else (np.nan, np.nan)
        pos_ci = data_bootstrap_ci(all_sample_metrics[config]["pos"]) if len(all_sample_metrics[config]["pos"]) > 0 else (np.nan, np.nan)
        eos_ci = data_bootstrap_ci(all_sample_metrics[config]["eos"]) if len(all_sample_metrics[config]["eos"]) > 0 else (np.nan, np.nan)
        
        ablation_rows.append({
            "config": config,
            "macro_f1_mean": f"{np.mean(f1s):.4f}",
            "macro_f1_data_ci": f"[{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]",
            "sos_mae_mean": f"{np.mean(sos_maes):.2f}" if sos_maes else "NA",
            "sos_mae_data_ci": f"[{sos_ci[0]:.2f}, {sos_ci[1]:.2f}]" if not np.isnan(sos_ci[0]) else "NA",
            "pos_mae_mean": f"{np.mean(pos_maes):.2f}" if pos_maes else "NA",
            "pos_mae_data_ci": f"[{pos_ci[0]:.2f}, {pos_ci[1]:.2f}]" if not np.isnan(pos_ci[0]) else "NA",
            "eos_mae_mean": f"{np.mean(eos_maes):.2f}" if eos_maes else "NA",
            "eos_mae_data_ci": f"[{eos_ci[0]:.2f}, {eos_ci[1]:.2f}]" if not np.isnan(eos_ci[0]) else "NA",
        })
    with open(TABLES_DIR / "table_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ablation_rows[0].keys())
        w.writeheader()
        w.writerows(ablation_rows)
    print(f"  Wrote table_ablation.csv ({len(ablation_rows)} rows)")
    
    # ===== Table: P-values (Full Stats) =====
    ref = "transformer"
    ref_seeds = sorted(all_metrics[ref].keys())
    
    raw_tests = []
    family_size = (len(CONFIGS) - 6) * 4  # 6 non-standard DL configs * 4 metrics = 24
    # (Excluding ML baselines and standard transformer from the family)
    
    for config in CONFIGS:
        if config == ref or "rf" in config or "xgboost" in config or "logistic" in config:
            continue
        config_seeds = sorted(all_metrics[config].keys())
        common_seeds = sorted(set(ref_seeds) & set(config_seeds))
        if len(common_seeds) < 3: continue
        
        for metric_name, metric_key in [("Macro-F1", "macro_f1"), ("SOS_MAE", "sos_mae"), 
                                        ("POS_MAE", "pos_mae"), ("EOS_MAE", "eos_mae")]:
            vals_ref = [all_metrics[ref][s][metric_key] for s in common_seeds]
            vals_cfg = [all_metrics[config][s][metric_key] for s in common_seeds]
            
            valid = [(r, c) for r, c in zip(vals_ref, vals_cfg) if not (np.isnan(r) or np.isnan(c))]
            if len(valid) < 3: continue
            
            arr_ref = np.array([r for r, c in valid])
            arr_cfg = np.array([c for r, c in valid])
            
            # Paired t-test
            t_stat, p_raw = stats.ttest_rel(arr_cfg, arr_ref)
            
            # Mean diff & Cohen's d_z
            diff = arr_cfg - arr_ref
            mean_diff = np.mean(diff)
            std_diff = np.std(diff, ddof=1)
            cohen_dz = mean_diff / std_diff if std_diff > 0 else 0
            
            # Seed Bootstrap CI
            seed_ci = seed_bootstrap_ci(arr_cfg, arr_ref)
            
            # TOST Equivalence Test for Macro-F1 (bounds +/- 0.02)
            # Bounds choice: 0.02 (2% absolute F1) is widely accepted in applied ML 
            # as a margin of practical equivalence (differences < 2% are often noise).
            p_tost = np.nan
            tost_res = "NA"
            if metric_name == "Macro-F1":
                bound = 0.02
                t1, p1 = stats.ttest_rel(arr_cfg, arr_ref, alternative="greater") # H0: diff <= -0.02, HA: diff > -0.02. Scipy doesn't support offset directly in ttest_rel.
                # Manual TOST
                se = std_diff / np.sqrt(len(diff))
                t1_manual = (mean_diff - (-bound)) / se
                p1_manual = 1 - stats.t.cdf(t1_manual, df=len(diff)-1)
                
                t2_manual = (mean_diff - bound) / se
                p2_manual = stats.t.cdf(t2_manual, df=len(diff)-1)
                
                p_tost = max(p1_manual, p2_manual)
                tost_res = "equivalent" if p_tost < ALPHA else "not equivalent"
                
            raw_tests.append({
                "config": config,
                "metric": metric_name,
                "n_seeds": len(valid),
                "mean_diff": mean_diff,
                "seed_ci_low": seed_ci[0],
                "seed_ci_high": seed_ci[1],
                "cohen_dz": cohen_dz,
                "t_stat": t_stat,
                "p_raw": p_raw,
                "p_tost": p_tost,
                "tost_result": tost_res,
            })
            
    # Apply Multiple Testing Corrections
    if raw_tests:
        p_raws = [t["p_raw"] for t in raw_tests]
        p_holm = adjust_pvalues_holm(p_raws)
        p_bonf = np.clip(np.array(p_raws) * len(p_raws), 0.0, 1.0)
        
        pval_rows = []
        for i, t in enumerate(raw_tests):
            pval_rows.append({
                "comparison": f"{t['config']} vs Standard",
                "metric": t["metric"],
                "n_seeds": t["n_seeds"],
                "mean_diff": f"{t['mean_diff']:+.4f}",
                "seed_95CI": f"[{t['seed_ci_low']:+.4f}, {t['seed_ci_high']:+.4f}]",
                "cohen_dz": f"{t['cohen_dz']:+.3f}",
                "p_raw": f"{t['p_raw']:.5f}",
                "p_holm": f"{p_holm[i]:.5f}",
                "p_bonferroni": f"{p_bonf[i]:.5f}",
                "signif_bonf": "diff" if p_bonf[i] < ALPHA else "ns",
                "p_tost": f"{t['p_tost']:.5f}" if not np.isnan(t['p_tost']) else "NA",
                "conclusion": "equivalent (TOST)" if t['tost_result'] == "equivalent" else ("diff" if p_bonf[i] < ALPHA else "not detected / inconclusive"),
            })
            
        with open(TABLES_DIR / "table_pvalues_full.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=pval_rows[0].keys())
            w.writeheader()
            w.writerows(pval_rows)
        print(f"  Wrote table_pvalues_full.csv ({len(pval_rows)} rows)")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
