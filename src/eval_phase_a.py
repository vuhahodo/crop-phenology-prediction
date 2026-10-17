import numpy as np
import torch
import scipy.stats as stats
from torch.utils.data import DataLoader
from src.train import load_records
from src.data.dataset import NDVISequenceDataset
from src.evaluation.metrics import segmentation_metrics
from src.make_tables import decode_milestones
from pathlib import Path
import json
import csv

CONFIGS = [
    "transformer",
    "transformer_cyclical_doy",
    "transformer_calendar_gate",
    "transformer_scalar_gate",
    "cba_phenonet",
    "cba_phenonet_residual",
    "cba_phenonet_disentangled",
]
SEEDS = [42, 123, 2024, 7, 314, 0, 1, 999, 555, 77]
DROP_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.5]
MODES = ["random", "block"]

def get_model(model_name, in_channels, device):
    if model_name == "transformer":
        from src.models.transformer import NDVITransformer
        model = NDVITransformer(in_channels=in_channels).to(device)
    elif model_name == "transformer_cyclical_doy":
        from src.models.ablation_models import TransformerCyclicalDOY
        model = TransformerCyclicalDOY(in_channels=in_channels).to(device)
    elif model_name == "transformer_calendar_gate":
        from src.models.ablation_models import TransformerCalendarGate
        model = TransformerCalendarGate(in_channels=in_channels).to(device)
    elif model_name == "transformer_scalar_gate":
        from src.b0_scalar_gate import TransformerScalarGate
        model = TransformerScalarGate(in_channels=in_channels).to(device)
    elif model_name == "cba_phenonet":
        from src.models.cba_phenonet import CBAPhenoNet
        model = CBAPhenoNet(in_channels=in_channels).to(device)
    elif model_name == "cba_phenonet_residual":
        from src.models.ablation_models import CBAPhenoNetResidual
        model = CBAPhenoNetResidual(in_channels=in_channels).to(device)
    elif model_name == "cba_phenonet_disentangled":
        from src.models.ablation_models import CBAPhenoNetDisentangled
        model = CBAPhenoNetDisentangled(in_channels=in_channels).to(device)
    else:
        raise ValueError(model_name)
    return model

def apply_block_gap(x, mask, drop_ratio, rng):
    """Contiguous block gap (simulating 2-4 weeks missing data)."""
    x_dropped = x.clone()
    for i in range(x.shape[0]):
        t = mask[i].sum().item()
        if t == 0: continue
        drop_len = int(t * drop_ratio)
        if drop_len == 0: continue
        start = rng.integers(0, t - drop_len + 1)
        x_dropped[i, start:start+drop_len, 0] = 0.0  # Zero only NDVI
    return x_dropped

def apply_random_gap(x, mask, drop_ratio, rng):
    """Randomly missing days across the sequence."""
    x_dropped = x.clone()
    for i in range(x.shape[0]):
        t = int(mask[i].sum().item())
        if t == 0: continue
        drop_len = int(t * drop_ratio)
        if drop_len == 0: continue
        drop_indices = rng.choice(t, size=drop_len, replace=False)
        x_dropped[i, drop_indices, 0] = 0.0  # Zero only NDVI
    return x_dropped

def evaluate_with_dropout(model, loader, device, drop_ratio, mode="random"):
    model.eval()
    all_true, all_pred, all_mask, all_doy, all_ndvi = [], [], [], [], []
    rng = np.random.default_rng(42 + int(drop_ratio*100) + (1 if mode=="block" else 0))
    
    with torch.no_grad():
        for batch in loader:
            x, y, mask = batch["x"], batch["y"], batch["mask"]
            if drop_ratio > 0:
                if mode == "block":
                    x = apply_block_gap(x, mask, drop_ratio, rng)
                else:
                    x = apply_random_gap(x, mask, drop_ratio, rng)
            logits = model(x.to(device))
            all_true.append(y.numpy())
            all_pred.append(logits.argmax(dim=-1).cpu().numpy())
            all_mask.append(mask.numpy())
            all_doy.append(batch["doy"].numpy())
            all_ndvi.append(x[:,:,0].numpy())  # the dropped NDVI sequence
            
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    mask = np.concatenate(all_mask)
    doy = np.concatenate(all_doy)
    ndvi = np.concatenate(all_ndvi)
    
    macro_f1 = segmentation_metrics(y_true, y_pred, mask)["macro_f1"]
    
    # Calculate MAE
    N = y_true.shape[0]
    errors = []
    for i in range(N):
        t = mask[i].sum()
        yt, yp = y_true[i, :t], y_pred[i, :t]
        d, n = doy[i, :t], ndvi[i, :t]
        tm = decode_milestones(yt, d, n)
        pm = decode_milestones(yp, d, n)
        # Average MAE across SOS, POS, EOS
        for m_idx in range(3):
            if tm[m_idx] is not None and pm[m_idx] is not None:
                errors.append(abs(tm[m_idx] - pm[m_idx]))
    
    mae = np.mean(errors) if errors else np.nan
    return macro_f1, mae

def data_bootstrap_ci(values, n_resamples=1000):
    """Simple bootstrap over seed results to get 95% CI."""
    if len(values) == 0: return np.nan, np.nan
    rng = np.random.default_rng(42)
    boot_means = [np.nanmean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_resamples)]
    return np.percentile(boot_means, [2.5, 97.5])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_records = load_records("test")
    max_len = max(len(r.ndvi_smoothed) for r in test_records)
    test_ds = NDVISequenceDataset(test_records, max_len=max_len)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    in_channels = test_ds.n_channels
    
    all_res = {mode: {m: {r: {"f1": [], "mae": []} for r in DROP_RATIOS} for m in CONFIGS} for mode in MODES}
    
    for m in CONFIGS:
        for s in SEEDS:
            model = get_model(m, in_channels, device)
            pt_path = Path(f"results/models_b0c/{m}_{s}.pt")
            if not pt_path.exists():
                print(f"Warning: {pt_path} missing. Skipping seed {s}.")
                continue
            model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
            
            for mode in MODES:
                for r in DROP_RATIOS:
                    if r == 0.0 and mode == "block":
                        # reuse random mode results for 0%
                        f1, mae = all_res["random"][m][0.0]["f1"][-1], all_res["random"][m][0.0]["mae"][-1]
                    else:
                        f1, mae = evaluate_with_dropout(model, test_loader, device, r, mode=mode)
                    
                    all_res[mode][m][r]["f1"].append(f1)
                    all_res[mode][m][r]["mae"].append(mae)

    # Markdown Tables Generation
    out_lines = ["# Báo cáo Robustness (NDVI Zero-masking)", ""]
    
    for mode in MODES:
        out_lines.append(f"## {mode.capitalize()} Gap")
        
        # F1 Table
        out_lines.append(f"### Macro-F1 (Mean +/- 95% CI)")
        header = "| Config | " + " | ".join(f"{int(r*100)}%" for r in DROP_RATIOS) + " |"
        out_lines.append(header)
        out_lines.append("|---" + "|---"*(len(DROP_RATIOS)) + "|")
        
        for m in CONFIGS:
            row = [m]
            for r in DROP_RATIOS:
                vals = all_res[mode][m][r]["f1"]
                if not vals: 
                    row.append("NA")
                    continue
                mean = np.nanmean(vals)
                ci = data_bootstrap_ci(vals)
                row.append(f"{mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
            out_lines.append("| " + " | ".join(row) + " |")
        out_lines.append("")
        
        # MAE Table
        out_lines.append(f"### MAE (ngày) (Mean +/- 95% CI)")
        out_lines.append(header)
        out_lines.append("|---" + "|---"*(len(DROP_RATIOS)) + "|")
        for m in CONFIGS:
            row = [m]
            for r in DROP_RATIOS:
                vals = all_res[mode][m][r]["mae"]
                if not vals: 
                    row.append("NA")
                    continue
                mean = np.nanmean(vals)
                ci = data_bootstrap_ci(vals)
                row.append(f"{mean:.1f} [{ci[0]:.1f}, {ci[1]:.1f}]")
            out_lines.append("| " + " | ".join(row) + " |")
        out_lines.append("")

    with open("results/tables/robustness_tables.md", "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print("Exported tables to results/tables/robustness_tables.md")

    # Plot Degradation Curves
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        colors = plt.cm.tab10.colors
        
        for idx, mode in enumerate(MODES):
            ax = axes[idx]
            for i, m in enumerate(CONFIGS):
                means = []
                lows = []
                highs = []
                for r in DROP_RATIOS:
                    vals = all_res[mode][m][r]["f1"]
                    if not vals:
                        means.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                        continue
                    means.append(np.nanmean(vals))
                    ci = data_bootstrap_ci(vals)
                    lows.append(ci[0])
                    highs.append(ci[1])
                
                x = [r * 100 for r in DROP_RATIOS]
                ax.plot(x, means, label=m, marker='o', color=colors[i % len(colors)])
                ax.fill_between(x, lows, highs, alpha=0.1, color=colors[i % len(colors)])
            
            ax.set_title(f"Macro-F1 vs NDVI Gap Rate ({mode.capitalize()} Gap)")
            ax.set_xlabel("Gap Rate (%)")
            ax.set_ylabel("Macro-F1")
            if idx == 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)
        
        os.makedirs("paper/figs", exist_ok=True)
        plt.savefig("paper/figs/robustness_curve.pdf", format='pdf', bbox_inches='tight')
        print("Exported plot to paper/figs/robustness_curve.pdf")
    except ImportError:
        print("matplotlib not installed, skipping plot.")

if __name__ == "__main__":
    main()
