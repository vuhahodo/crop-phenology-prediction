import pandas as pd
from scipy.stats import ttest_rel
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("D:/HocTap/N4_K1/ComputerVision/FinalExam/Final_Exam/TimeSeries/results")
ABLATION_CSV = RESULTS_DIR / "ablation_study.csv"
COMPARISON_CSV = RESULTS_DIR / "comparison_table.csv"

# Load data
combined_ablation = pd.read_csv(ABLATION_CSV) if ABLATION_CSV.exists() else pd.DataFrame()
comp_df = pd.read_csv(COMPARISON_CSV) if COMPARISON_CSV.exists() else pd.DataFrame()

# Helper to get aligned data for a model
def get_model_data(model_name, source_df):
    if source_df.empty: return None
    df = source_df[source_df["model_name"] == model_name].copy()
    if df.empty: return None
    
    # Calculate overall RMSE as done in run_ablation.py
    df["overall_rmse_days"] = (df["sos_rmse_days"] + df["pos_rmse_days"] + df["eos_rmse_days"]) / 3.0
    
    # If seed column doesn't exist, extract it from member_name
    if "seed" not in df.columns:
        def extract_seed(name):
            import re
            m = re.search(r"\[seed=(\d+)\]", name)
            if m: return int(m.group(1))
            return 42 # Default seed
        df["seed"] = df["member_name"].apply(extract_seed)
        
    # Sort by seed to ensure alignment
    df = df.sort_values("seed").reset_index(drop=True)
    return df

configs = {
    "Standard": get_model_data("transformer", comp_df),
    "Cyclical DOY": get_model_data("transformer_cyclical_doy", combined_ablation),
    "Calendar Gate": get_model_data("transformer_calendar_gate", combined_ablation),
    "Full CBA": get_model_data("cba_phenonet", comp_df),
    "Residual": get_model_data("cba_phenonet_residual", combined_ablation),
    "Disentangled": get_model_data("cba_phenonet_disentangled", combined_ablation)
}

# Metrics to test
metrics = {
    "Macro F1": "macro_f1",
    "SOS MAE": "sos_mae_days",
    "POS MAE": "pos_mae_days",
    "EOS MAE": "eos_mae_days",
    "Overall RMSE": "overall_rmse_days"
}

# 4 configs -> 6 pairs + new ones against Standard and Cyclical DOY
pairs = [
    ("Standard", "Cyclical DOY"),
    ("Standard", "Full CBA"),
    ("Cyclical DOY", "Full CBA"),
    ("Standard", "Residual"),
    ("Cyclical DOY", "Residual"),
    ("Standard", "Disentangled"),
    ("Cyclical DOY", "Disentangled")
]

alpha = 0.05
# Bonferroni correction: 6 comparisons per metric
alpha_bonf = alpha / len(pairs)

print(f"Paired t-test results across 5 seeds.")
print(f"Significance level: alpha = {alpha}")
print(f"Bonferroni-corrected alpha = {alpha_bonf:.4f} (for 6 comparisons per metric)\n")

for metric_name, col in metrics.items():
    print(f"--- Metric: {metric_name} ---")
    
    for m1, m2 in pairs:
        df1 = configs[m1]
        df2 = configs[m2]
        
        if df1 is None or df2 is None or len(df1) != 5 or len(df2) != 5:
            print(f"{m1} vs {m2}: Missing data")
            continue
            
        data1 = df1[col].values
        data2 = df2[col].values
        
        # paired t-test
        stat, p_val = ttest_rel(data1, data2)
        
        # Determine significance
        sig_standard = "*" if p_val < alpha else "ns"
        sig_bonf = "**" if p_val < alpha_bonf else ""
        
        # Mean difference
        mean_diff = data2.mean() - data1.mean()
        
        print(f"{m1:15s} vs {m2:15s} | Mean Diff: {mean_diff:>7.3f} | p-value: {p_val:>7.4f} {sig_standard}{sig_bonf}")
    print()
