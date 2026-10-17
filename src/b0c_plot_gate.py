import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.ablation_models import TransformerCalendarGate
from src.data.dataset import cyclical_doy

def main():
    device = "cpu"
    seeds = [42, 123, 2024, 7, 314, 0, 1, 999, 555, 77]
    doys = np.arange(1, 366)
    
    # doy_sin_cos for DOY 1 to 365
    doy_sin, doy_cos = cyclical_doy(doys)
    doy_features = torch.tensor(np.stack([doy_sin, doy_cos], axis=-1), dtype=torch.float32).to(device)
    
    # Store gate values: shape (10 seeds, 2 layers, 365 days)
    gate_values = np.zeros((len(seeds), 2, 365))
    
    for i, s in enumerate(seeds):
        model = TransformerCalendarGate(in_channels=10).to(device)
        pt_path = f"results/models_b0c/transformer_calendar_gate_{s}.pt"
        if not Path(pt_path).exists():
            print(f"Skipping {pt_path}")
            continue
            
        model.load_state_dict(torch.load(pt_path, map_location=device, weights_only=True))
        model.eval()
        
        with torch.no_grad():
            for layer_idx in range(2):
                gate_module = model.layers[layer_idx].gate.gate
                val = gate_module(doy_features).squeeze(-1).numpy()
                gate_values[i, layer_idx, :] = val

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for layer_idx in range(2):
        ax = axes[layer_idx]
        mean_gate = np.mean(gate_values[:, layer_idx, :], axis=0)
        std_gate = np.std(gate_values[:, layer_idx, :], axis=0)
        
        ax.plot(doys, mean_gate, label=f"Layer {layer_idx+1} Mean", color="blue")
        ax.fill_between(doys, mean_gate - std_gate, mean_gate + std_gate, alpha=0.3, color="blue", label="+/- 1 Std (across seeds)")
        
        ax.set_ylim(0, 1.0)
        ax.set_title(f"Calendar Gate Value over DOY (Layer {layer_idx+1})")
        ax.set_xlabel("Day of Year (DOY)")
        ax.set_ylabel("Gate Value")
        ax.legend()
        ax.grid(True)
        
    import os
    os.makedirs("paper/figs", exist_ok=True)
    out_img = "paper/figs/calendar_gate_curve.pdf"
    plt.tight_layout()
    plt.savefig(out_img, format='pdf', bbox_inches='tight')
    print(f"Saved plot to {out_img}")
    
    # Statistics
    print("\n--- Gate Variance Statistics ---")
    for layer_idx in range(2):
        # Variance across DOY (for a single seed, how much does the gate change?)
        var_across_doy_per_seed = np.var(gate_values[:, layer_idx, :], axis=1) # shape (10,)
        mean_var_across_doy = np.mean(var_across_doy_per_seed)
        
        # Variance across seeds (for a single DOY, how much does initialization/training randomness change the gate?)
        var_across_seeds_per_doy = np.var(gate_values[:, layer_idx, :], axis=0) # shape (365,)
        mean_var_across_seeds = np.mean(var_across_seeds_per_doy)
        
        print(f"Layer {layer_idx+1}:")
        print(f"  Mean Variance across DOY (time dynamics): {mean_var_across_doy:.6f}")
        print(f"  Mean Variance across Seeds (randomness):  {mean_var_across_seeds:.6f}")
        
        if mean_var_across_doy < mean_var_across_seeds:
            print("  -> Conclusion: The gate is essentially constant over time. It fluctuates more due to random seed initialization than due to the actual calendar day!")

if __name__ == "__main__":
    main()
