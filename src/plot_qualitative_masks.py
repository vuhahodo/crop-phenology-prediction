"""Generates Fig. 3: Qualitative temporal segmentation examples for manuscript.
Displays smoothed NDVI trajectory with paired True and Predicted temporal stage masks,
plus decoded phenological milestones (SOS, POS, EOS).
Covers 4 representative test cases matching Section 4.4 Error Analysis.
Saves to docs/figures/qualitative_masks.png and results/plots/qualitative_masks.png.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def decode_milestones(y, d, n):
    sos = None
    for t in range(len(y)-1):
        if y[t] == 0 and y[t+1] == 1:
            sos = d[t+1]
            break
    active_idx = np.where(np.isin(y, [1, 2]))[0]
    pos = d[active_idx[np.argmax(n[active_idx])]] if len(active_idx) > 0 else d[np.argmax(n)]
    eos = None
    for t in range(len(y)-1):
        if y[t] == 3 and y[t+1] == 0:
            eos = d[t]
            break
    return sos, pos, eos

def generate_qualitative_figure():
    preds_path = Path("results/preds/cba_phenonet_42.npz")
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing {preds_path}")
        
    data = np.load(preds_path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    doy = data["doy"]
    ndvi = data["ndvi"]

    # 4 distinct cases matching advisor feedback:
    # 1. Correct (Wheat)
    # 2. Correct (Corn)
    # 3. Early SOS (Barley)
    # 4. Early EOS / Difficult case (Rapeseed)
    selected = [
        (27,  "Wheat",    "Correct",                     100.0, None),
        (2,   "Corn",     "Correct",                     95.9,  None),
        (14,  "Barley",   "Early SOS",                   89.0,  "Early green-up anomaly (ΔSOS: -35d)"),
        (182, "Rapeseed", "Early EOS / Difficult case",  84.9,  "Premature senescence (ΔEOS: -35d)")
    ]

    # Stage colors: 0: Fallow, 1: Vegetative, 2: Reproductive, 3: Senescence
    colors = ["#CBD5E1", "#22C55E", "#F59E0B", "#EF4444"]
    cmap = matplotlib.colors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(11.5, 10.5), dpi=300)
    outer_gs = matplotlib.gridspec.GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 1, 1], hspace=0.42)
    panels = ["(a)", "(b)", "(c)", "(d)"]

    for i, (idx, crop, category, acc, annotation) in enumerate(selected):
        inner_gs = matplotlib.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[i], height_ratios=[1.8, 1.1], hspace=0.08)
        ax_ndvi = fig.add_subplot(inner_gs[0])
        ax_mask = fig.add_subplot(inner_gs[1], sharex=ax_ndvi)

        t_doy = doy[idx]
        t_ndvi = ndvi[idx]
        t_true = y_true[idx]
        t_pred = y_pred[idx]

        tsos, tpos, teos = decode_milestones(t_true, t_doy, t_ndvi)
        psos, ppos, peos = decode_milestones(t_pred, t_doy, t_ndvi)

        # ---------------- 1. NDVI Trajectory ----------------
        ax_ndvi.plot(t_doy, t_ndvi, color="#0F172A", linewidth=2.0, zorder=3)
        ax_ndvi.fill_between(t_doy, t_ndvi, 0, color="#38BDF8", alpha=0.15, zorder=2)
        ax_ndvi.set_ylabel("NDVI", fontsize=9.5, fontweight="bold", color="#1E293B")
        ax_ndvi.set_ylim(-0.06, 0.98)
        ax_ndvi.set_xlim(t_doy[0], t_doy[-1])
        ax_ndvi.grid(True, linestyle=":", alpha=0.55, color="#94A3B8", zorder=1)

        title_text = f"{panels[i]} {crop} — {category} (Acc: {acc:.1f}%)"
        ax_ndvi.set_title(title_text, fontsize=11, fontweight="bold", loc="left", color="#0F172A", pad=5)

        if annotation:
            ax_ndvi.text(
                0.985, 0.82, annotation,
                transform=ax_ndvi.transAxes,
                fontsize=8.5, fontweight="semibold",
                color="#B91C1C",
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#FEF2F2", edgecolor="#FCA5A5", alpha=0.92)
            )

        if tsos: ax_ndvi.axvline(tsos, color="#15803D", linestyle="-", linewidth=1.8, alpha=0.85, zorder=4)
        if psos: ax_ndvi.axvline(psos, color="#15803D", linestyle="--", linewidth=1.8, alpha=0.95, zorder=4)
        if tpos: ax_ndvi.plot(tpos, t_ndvi[np.where(t_doy == tpos)[0][0]], marker="*", markersize=8.5, color="#D97706", zorder=5)
        if teos: ax_ndvi.axvline(teos, color="#B91C1C", linestyle="-", linewidth=1.8, alpha=0.85, zorder=4)
        if peos: ax_ndvi.axvline(peos, color="#B91C1C", linestyle="--", linewidth=1.8, alpha=0.95, zorder=4)

        ax_ndvi.tick_params(axis="x", labelbottom=False, bottom=False)
        ax_ndvi.tick_params(axis="y", labelsize=8.5)

        # ---------------- 2. Prominent Masks ----------------
        # Stack: row 0 = Prediction (bottom), row 1 = Ground Truth (top) with origin='lower'
        mask_grid = np.vstack([t_pred, t_true])
        ax_mask.imshow(
            mask_grid,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=norm,
            extent=[t_doy[0], t_doy[-1], -0.5, 1.5],
            interpolation="nearest"
        )
        
        ax_mask.axhline(0.5, color="white", linewidth=2.5, zorder=3)
        ax_mask.set_yticks([0, 1])
        ax_mask.set_yticklabels(["Prediction", "Ground Truth"], fontsize=9.5, fontweight="bold", color="#1E293B")

        if tsos: ax_mask.axvline(tsos, color="#15803D", linestyle="-", linewidth=1.8, alpha=0.75, zorder=4)
        if psos: ax_mask.axvline(psos, color="#15803D", linestyle="--", linewidth=1.8, alpha=0.95, zorder=4)
        if teos: ax_mask.axvline(teos, color="#B91C1C", linestyle="-", linewidth=1.8, alpha=0.75, zorder=4)
        if peos: ax_mask.axvline(peos, color="#B91C1C", linestyle="--", linewidth=1.8, alpha=0.95, zorder=4)

        for spine in ax_mask.spines.values():
            spine.set_edgecolor("#94A3B8")
            spine.set_linewidth(1.0)

        if i < 3:
            ax_mask.tick_params(axis="x", labelbottom=False)
        else:
            ax_mask.tick_params(axis="x", labelsize=9)
            ax_mask.set_xlabel("Day of Year (DOY) [5-day intervals, T=73]", fontsize=10, fontweight="bold", color="#0F172A", labelpad=6)

    stage_patches = [
        mpatches.Patch(facecolor=colors[0], edgecolor="#94A3B8", label="0: Fallow"),
        mpatches.Patch(facecolor=colors[1], edgecolor="#15803D", label="1: Vegetative"),
        mpatches.Patch(facecolor=colors[2], edgecolor="#B45309", label="2: Reproductive"),
        mpatches.Patch(facecolor=colors[3], edgecolor="#B91C1C", label="3: Senescence")
    ]
    milestone_lines = [
        matplotlib.lines.Line2D([], [], color="#15803D", linestyle="-", linewidth=1.8, label="SOS (GT)"),
        matplotlib.lines.Line2D([], [], color="#15803D", linestyle="--", linewidth=1.8, label="SOS (Pred)"),
        matplotlib.lines.Line2D([], [], color="#D97706", marker="*", linestyle="None", markersize=8.5, label="POS (Peak)"),
        matplotlib.lines.Line2D([], [], color="#B91C1C", linestyle="-", linewidth=1.8, label="EOS (GT)"),
        matplotlib.lines.Line2D([], [], color="#B91C1C", linestyle="--", linewidth=1.8, label="EOS (Pred)")
    ]

    fig.legend(
        handles=stage_patches + milestone_lines,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.998),
        ncol=9,
        fontsize=8.5,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1"
    )

    out1 = Path("docs/figures/qualitative_masks_v2.png")
    out2 = Path("results/plots/qualitative_masks_v2.png")
    out1.parent.mkdir(parents=True, exist_ok=True)
    out2.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out1, dpi=300, bbox_inches="tight")
    plt.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved qualitative masks v2 figure to {out1} and {out2}")

if __name__ == "__main__":
    generate_qualitative_figure()
