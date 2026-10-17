"""Generates publication-quality 1D Temporal Segmentation Mask comparison.
Visualizes Ground Truth Mask vs Predicted Mask (CBA-PhenoNet) alongside the NDVI curve
for sample test parcels across the four crop categories.
Saves to docs/figures/temporal_segmentation_masks.png.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def generate_temporal_masks():
    preds_path = Path("results/preds/cba_phenonet_42.npz")
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}")
        
    data = np.load(preds_path)
    y_true = data["y_true"]      # (399, 73)
    y_pred = data["y_pred"]      # (399, 73)
    doy = data["doy"]            # (399, 73)
    ndvi = data["ndvi"]          # (399, 73)
    season_ids = data["season_ids"] # (399,)

    # Find representative samples for each crop
    crops = ["barley", "wheat", "rapeseed", "corn"]
    crop_titles = ["Barley (Đại mạch)", "Wheat (Lúa mì)", "Rapeseed (Hạt cải dầu)", "Corn (Ngô)"]
    selected_indices = []
    
    for c in crops:
        for idx, sid in enumerate(season_ids):
            if sid.endswith(f"_{c}") and np.sum(y_true[idx] == 1) > 5 and np.sum(y_true[idx] == 3) > 5:
                selected_indices.append(idx)
                break
                
    # Class colors: 0: Fallow, 1: Vegetative, 2: Reproductive, 3: Senescence
    colors = ["#BDC3C7", "#2ECC71", "#F39C12", "#E74C3C"]
    cmap = matplotlib.colors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), dpi=300, sharex=True)

    for i, idx in enumerate(selected_indices):
        ax = axes[i]
        t_doy = doy[idx]
        t_ndvi = ndvi[idx]
        t_true = y_true[idx]
        t_pred = y_pred[idx]
        sid = season_ids[idx]
        crop_name = crop_titles[i]

        # 1. Plot NDVI Curve
        ax.plot(t_doy, t_ndvi, color="#2C3E50", linewidth=2.2, label="NDVI Trajectory", zorder=3)
        ax.fill_between(t_doy, t_ndvi, 0, color="#3498DB", alpha=0.15, zorder=2)
        ax.set_ylabel("NDVI", fontsize=10.5, fontweight="bold")
        ax.set_ylim(-0.05, 1.0)
        ax.set_xlim(t_doy[0], t_doy[-1])
        ax.grid(True, linestyle=":", alpha=0.5, zorder=1)
        ax.set_title(f"Sample Parcel {i+1}: {crop_name} (ID: {sid})", fontsize=11.5, fontweight="bold", pad=8)

        # 2. Add twin axis for the 2 segmentation mask strips
        # We place them at the bottom inside or below using an inset/pcolormesh
        # Let's create an image bar for True and Pred masks
        mask_grid = np.vstack([t_pred, t_true]) # row 0: Pred, row 1: True
        
        # Overlay color bands directly on the background of the NDVI plot
        # Or draw 2 horizontal ribbons at y = -0.04 to -0.01
        ribbon_ax = ax.inset_axes([0.0, -0.38, 1.0, 0.28])
        ribbon_ax.imshow(mask_grid, aspect="auto", cmap=cmap, norm=norm, 
                         extent=[t_doy[0], t_doy[-1], -0.5, 1.5])
        ribbon_ax.set_yticks([0, 1])
        ribbon_ax.set_yticklabels(["Pred Mask", "True Mask"], fontsize=9.5, fontweight="bold")
        ribbon_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    axes[-1].set_xlabel("Day of Year (DOY) [1 – 365]", fontsize=11, fontweight="bold", labelpad=25)
    
    # Common Legend at the top
    patches = [
        mpatches.Patch(color=colors[0], label="Class 0: Fallow (Đất trống)"),
        mpatches.Patch(color=colors[1], label="Class 1: Vegetative (Sinh trưởng)"),
        mpatches.Patch(color=colors[2], label="Class 2: Reproductive (Sinh sản / Cực đại)"),
        mpatches.Patch(color=colors[3], label="Class 3: Senescence (Già hóa / Tàn lụi)")
    ]
    fig.legend(handles=patches, loc="upper center", bbox_to_anchor=(0.5, 0.99),
               ncol=4, fontsize=10, frameon=True, facecolor="#F8F9FA", edgecolor="#CBD5E0")

    plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.6)
    
    out_path = Path("docs/figures/temporal_segmentation_masks.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Temporal Segmentation Mask visualization to: {out_path}")

if __name__ == "__main__":
    generate_temporal_masks()
