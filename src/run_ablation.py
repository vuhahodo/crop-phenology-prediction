"""Ablation Study Runner:
Evaluates all 4 configurations across the 5 canonical seeds (42, 123, 2024, 7, 314):
1. Standard Transformer (Sinusoidal Positional Encoding, no Calendar Gate)
2. Transformer + Cyclical DOY Positional Encoding (Eq. 1, no Calendar Gate)
3. Transformer + Calendar Gating Prior (Eq. 2-3, Sinusoidal PE)
4. Full CBA-PhenoNet (Both Cyclical DOY + Calendar Gating Prior)

Outputs results to results/ablation_study.csv and displays aggregated summary statistics.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.train import train_dl_model
from src.evaluation.metrics import build_eval_result

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
ABLATION_CSV = RESULTS_DIR / "ablation_study.csv"
COMPARISON_CSV = RESULTS_DIR / "comparison_table.csv"

SEEDS = [42, 123, 2024, 7, 314]


def run_model_ablation(model_name: str, num_threads: int = 5):
    torch.set_num_threads(num_threads)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / f"ablation_{model_name}.csv"
    existing_df = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()

    print(f"\n=======================================================")
    print(f"RUNNING ABLATION FOR: {model_name} (threads={num_threads})")
    print(f"=======================================================")

    for seed in SEEDS:
        if not existing_df.empty:
            already = existing_df[
                (existing_df["model_name"] == model_name) &
                (existing_df["seed"] == seed)
            ]
            if not already.empty:
                print(f"[SKIP] {model_name} seed={seed} already completed.")
                continue

        print(f"\n---> Training {model_name} [seed={seed}] ...")
        sys.stdout.flush()

        y_true, y_pred, doy, ndvi, mask, train_s, infer_ms, season_ids = train_dl_model(
            model_name, device="cpu", seed=seed
        )

        result, _ = build_eval_result(
            model_name=model_name,
            member_name=f"Ablation ({model_name}) [seed={seed}]",
            y_true=y_true, y_pred=y_pred, doy=doy, ndvi=ndvi, mask=mask,
            train_seconds=train_s, inference_ms_per_sample=infer_ms,
            season_ids=season_ids,
        )

        row = result.to_row()
        row["seed"] = seed

        file_exists = out_csv.exists()
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        existing_df = pd.read_csv(out_csv)
        print(f"[DONE] {model_name} seed={seed}: F1={row['macro_f1']:.4f}, SOS MAE={row['sos_mae_days']:.1f}d, POS MAE={row['pos_mae_days']:.1f}d, EOS MAE={row['eos_mae_days']:.1f}d, EOS RMSE={row['eos_rmse_days']:.1f}d")
        sys.stdout.flush()


def generate_summary():
    """Combines individual ablation files and baseline comparison table into results/ablation_study.csv and ablation_summary.csv."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_ablation_rows = []

    for m in ["transformer_cyclical_doy", "transformer_calendar_gate", "cba_phenonet_residual", "cba_phenonet_disentangled"]:
        m_csv = RESULTS_DIR / f"ablation_{m}.csv"
        if m_csv.exists():
            df_m = pd.read_csv(m_csv)
            all_ablation_rows.append(df_m)

    if all_ablation_rows:
        combined_ablation = pd.concat(all_ablation_rows, ignore_index=True)
        combined_ablation.to_csv(ABLATION_CSV, index=False)
    else:
        combined_ablation = pd.DataFrame()

    comp_df = pd.read_csv(COMPARISON_CSV) if COMPARISON_CSV.exists() else pd.DataFrame()

    configs = [
        ("Standard Transformer", comp_df[comp_df["model_name"] == "transformer"], 67460),
        ("Transformer + Cyclical DOY", combined_ablation[combined_ablation["model_name"] == "transformer_cyclical_doy"] if not combined_ablation.empty else pd.DataFrame(), 67652),
        ("Transformer + Calendar Gate", combined_ablation[combined_ablation["model_name"] == "transformer_calendar_gate"] if not combined_ablation.empty else pd.DataFrame(), 67590),
        ("CBA-PhenoNet (Full)", comp_df[comp_df["model_name"] == "cba_phenonet"], 67782),
        ("CBA-PhenoNet (Residual)", combined_ablation[combined_ablation["model_name"] == "cba_phenonet_residual"] if not combined_ablation.empty else pd.DataFrame(), 67782),
        ("CBA-PhenoNet (Disentangled)", combined_ablation[combined_ablation["model_name"] == "cba_phenonet_disentangled"] if not combined_ablation.empty else pd.DataFrame(), 67717),
    ]

    summary_rows = []
    for name, sub, params in configs:
        if len(sub) == 0:
            print(f"Warning: No results found for {name}")
            continue
        f1_m, f1_s = sub["macro_f1"].mean(), sub["macro_f1"].std()
        sos_m, sos_s = sub["sos_mae_days"].mean(), sub["sos_mae_days"].std()
        pos_m, pos_s = sub["pos_mae_days"].mean(), sub["pos_mae_days"].std()
        eos_m, eos_s = sub["eos_mae_days"].mean(), sub["eos_mae_days"].std()
        
        # Overall 3-milestone RMSE
        overall_rmse_series = (sub["sos_rmse_days"] + sub["pos_rmse_days"] + sub["eos_rmse_days"]) / 3.0
        rmse_m, rmse_s = overall_rmse_series.mean(), overall_rmse_series.std()

        summary_rows.append({
            "Configuration": name,
            "Parameters": params,
            "Delta_Params": f"+{params - 67460}",
            "Macro_F1": f"{f1_m:.3f} ± {f1_s:.3f}",
            "SOS_MAE_days": f"{sos_m:.1f} ± {sos_s:.1f}",
            "POS_MAE_days": f"{pos_m:.1f} ± {pos_s:.1f}",
            "EOS_MAE_days": f"{eos_m:.1f} ± {eos_s:.1f}",
            "Overall_RMSE_days": f"{rmse_m:.1f} ± {rmse_s:.1f}",
        })

    summary_table = pd.DataFrame(summary_rows)
    print("\n" + "=" * 80)
    print("ABLATION STUDY: 4-CONFIGURATION RIGOROUS BENCHMARK (5 SEEDS)")
    print("=" * 80)
    print(summary_table.to_string(index=False))
    print("=" * 80)
    summary_path = RESULTS_DIR / "ablation_summary.csv"
    summary_table.to_csv(summary_path, index=False)
    print(f"\nSaved summary table to {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transformer_cyclical_doy", "transformer_calendar_gate", "cba_phenonet_residual", "cba_phenonet_disentangled", "all", "summary"], default="all")
    parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()

    if args.model == "summary":
        generate_summary()
    elif args.model == "all":
        run_model_ablation("transformer_cyclical_doy", num_threads=args.threads)
        run_model_ablation("transformer_calendar_gate", num_threads=args.threads)
        run_model_ablation("cba_phenonet_residual", num_threads=args.threads)
        run_model_ablation("cba_phenonet_disentangled", num_threads=args.threads)
        generate_summary()
    else:
        run_model_ablation(args.model, num_threads=args.threads)


if __name__ == "__main__":
    main()
