import json
import numpy as np
import scipy.stats as stats

# Extracted from b0_milestones.md manually or programmatically.
# Let's write the exact arrays based on b0_milestones.md
models = ["transformer", "transformer_cyclical_doy", "transformer_calendar_gate", "cba_phenonet"]
seeds = [42, 123, 2024, 7, 314]

mae_sos_int = {
    "transformer": [6.97, 6.67, 8.62, 7.44, 9.16],
    "transformer_cyclical_doy": [11.56, 9.14, 7.90, 7.27, 22.04],
    "transformer_calendar_gate": [8.57, 9.25, 7.21, 7.12, 8.95],
    "cba_phenonet": [8.44, 6.15, 5.15, 8.86, 7.12]
}

mae_pos_int = {
    "transformer": [5.34, 3.77, 3.68, 5.22, 4.38],
    "transformer_cyclical_doy": [6.54, 5.61, 4.44, 4.79, 8.78],
    "transformer_calendar_gate": [5.14, 5.66, 4.02, 6.46, 4.38],
    "cba_phenonet": [4.42, 4.93, 3.72, 5.84, 3.58]
}

mae_eos_int = {
    "transformer": [5.77, 3.95, 3.33, 5.50, 4.26],
    "transformer_cyclical_doy": [5.38, 5.50, 3.93, 5.04, 8.77],
    "transformer_calendar_gate": [4.86, 5.04, 3.28, 5.26, 4.79],
    "cba_phenonet": [4.29, 4.50, 3.36, 6.95, 4.27]
}

def report_ttest(m1, m2, metric_dict, metric_name, family_size=4):
    val1 = np.array(metric_dict[m1])
    val2 = np.array(metric_dict[m2])
    t_stat, p_val = stats.ttest_rel(val1, val2)
    p_adj = min(1.0, p_val * family_size)
    mean_diff = np.mean(val1 - val2)
    sig = "*" if p_adj < 0.05 else "ns"
    print(f"{metric_name} ({m1} vs {m2}): Diff = {mean_diff:+.2f}, p={p_val:.4f}, p_adj={p_adj:.4f} ({sig})")

def main():
    print("--- MAE Intersection Paired T-Tests (Bonferroni n=4) ---")
    pairs = [
        ("transformer_cyclical_doy", "transformer"),
        ("cba_phenonet", "transformer"),
        ("transformer_calendar_gate", "transformer"),
        ("cba_phenonet", "transformer_cyclical_doy")
    ]
    for m1, m2 in pairs:
        report_ttest(m1, m2, mae_sos_int, "MAE SOS")
        report_ttest(m1, m2, mae_pos_int, "MAE POS")
        report_ttest(m1, m2, mae_eos_int, "MAE EOS")
        print()

if __name__ == "__main__":
    main()
