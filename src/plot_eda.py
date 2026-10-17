"""Generates publication-quality Fig. 1 (EDA) for the paper:
Panel (a): Distribution of seasonal NDVI amplitude across the four crop types with 0.15 threshold.
Panel (b): Bar chart of the four-class proportions showing severe class imbalance.
Saves to docs/figures/eda_distribution.png and results/plots/eda_distribution.png.
"""
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_eda_figure():
    cache_path = Path("data/frh01_cache.pkl")
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing {cache_path}")
    
    records = pickle.load(open(cache_path, "rb"))
    
    # 1. Amplitude per crop
    crop_order = ["Barley", "Wheat", "Rapeseed", "Corn"]
    crop_colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    crop_amps = {c: [] for c in crop_order}
    
    for r in records:
        c = r.season_id.split("_")[-1].capitalize()
        if c in crop_amps:
            crop_amps[c].append(float(np.ptp(r.ndvi_smoothed)))
            
    # 2. Stage label distribution
    all_labels = np.concatenate([r.labels for r in records])
    total_timesteps = len(all_labels)
    class_names = ["Fallow\n(Class 0)", "Vegetative\n(Class 1)", "Reproductive\n(Class 2)", "Senescence\n(Class 3)"]
    class_colors = ["#7F8C8D", "#2ECC71", "#F39C12", "#E74C3C"]
    class_counts = [np.sum(all_labels == i) for i in range(4)]
    class_pcts = [cnt / total_timesteps * 100 for cnt in class_counts]
    
    # Create 1x2 figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), dpi=300)
    
    # --- PANEL (A): Seasonal Amplitude Distribution ---
    positions = np.arange(len(crop_order))
    box_data = [crop_amps[c] for c in crop_order]
    
    bp = ax1.boxplot(box_data, positions=positions, patch_artist=True, widths=0.55,
                     medianprops=dict(color="black", linewidth=1.5),
                     whiskerprops=dict(color="#4A5568", linewidth=1.2),
                     capprops=dict(color="#4A5568", linewidth=1.2),
                     flierprops=dict(marker='o', markersize=3, alpha=0.3, markerfacecolor='gray'))
    
    for patch, color in zip(bp['boxes'], crop_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#2D3748")
        patch.set_linewidth(1.2)
        
    ax1.axhline(0.15, color="#E53E3E", linestyle="--", linewidth=1.8, label=r"Quality filter ($\Delta$NDVI $\geq$ 0.15)")
    ax1.set_xticks(positions)
    ax1.set_xticklabels([f"{c}\n(n={len(crop_amps[c])})" for c in crop_order], fontsize=10.5, fontweight="semibold")
    ax1.set_ylabel("Seasonal NDVI Amplitude ($\Delta$NDVI)", fontsize=11, fontweight="bold")
    ax1.set_title("(a) Seasonal Amplitude Across Crops", fontsize=12, fontweight="bold", pad=10)
    ax1.set_ylim(0.10, 0.82)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left", fontsize=9.5, framealpha=0.9)
    
    # Annotate mean values
    for i, c in enumerate(crop_order):
        m = np.mean(crop_amps[c])
        ax1.text(i, m + 0.02, f"$\mu$={m:.3f}", ha="center", fontsize=8.5, fontweight="bold", color="#1A202C")

    # --- PANEL (B): Stage Label Imbalance ---
    bars = ax2.bar(range(4), class_pcts, color=class_colors, width=0.55, edgecolor="#2D3748", linewidth=1.2, alpha=0.85)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(class_names, fontsize=10.5, fontweight="semibold")
    ax2.set_ylabel("Timestep Percentage (%)", fontsize=11, fontweight="bold")
    ax2.set_title("(b) Point-Wise Stage Label Distribution", fontsize=12, fontweight="bold", pad=10)
    ax2.set_ylim(0, 78)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    
    # Value annotations on top of bars
    for bar, pct, cnt in zip(bars, class_pcts, class_counts):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{pct:.2f}%\n({cnt:,})", 
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#1A202C")
                 
    plt.tight_layout()
    
    # Save to both target locations
    docs_out = Path("docs/figures/eda_distribution.png")
    plots_out = Path("results/plots/eda_distribution.png")
    docs_out.parent.mkdir(parents=True, exist_ok=True)
    plots_out.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(docs_out, dpi=300, bbox_inches="tight")
    plt.savefig(plots_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Successfully generated Fig. 1 at: {docs_out} and {plots_out}")

if __name__ == "__main__":
    generate_eda_figure()
