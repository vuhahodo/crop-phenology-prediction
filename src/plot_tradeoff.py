"""Generates Accuracy vs. Inference Latency trade-off plot.
Saves figure to results/plots/accuracy_vs_latency_tradeoff.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def create_tradeoff_plot():
    fig, ax = plt.subplots(figsize=(8.2, 5.5), dpi=300)

    # Models data from Table 2:
    other_models = [
        ("RF", 62.17, 0.420, "#718096", "s", (-14, -18)),
        ("XGBoost", 9.82, 0.403, "#A0AEC0", "^", (10, -10)),
        ("1D-CNN", 0.32, 0.636, "#DD6B20", "o", (12, -6)),
        ("Bi-LSTM", 0.52, 0.778, "#3182CE", "D", (12, -6)),
    ]

    for name, lat, f1, col, marker, offset in other_models:
        ax.scatter(lat, f1, color=col, s=120, marker=marker, edgecolors="black", linewidth=1.1, zorder=4)
        ax.annotate(name, (lat, f1), xytext=offset, textcoords="offset points",
                    fontsize=9.5, fontweight="normal", color="#1A202C")

    # Attention models (5-seed test set metrics matching Table 2):
    # Star (Ours) size 180, Plus (Transformer) size 110
    ax.scatter(0.27, 0.897, color="#E53E3E", s=180, marker="*", edgecolors="#742A2A", linewidth=1.1, zorder=5)
    ax.scatter(0.28, 0.904, color="#805AD5", s=110, marker="P", edgecolors="#2D3748", linewidth=1.1, zorder=6)

    # Upper callout: Transformer with downward-arching leader line
    ax.annotate(
        "Transformer (0.28 ms, 0.904)",
        xy=(0.28, 0.904),
        xytext=(0.44, 0.946),
        fontsize=9,
        fontweight="semibold",
        color="#44337A",
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=-0.08",
            color="#805AD5",
            lw=1.3,
            shrinkA=0,
            shrinkB=6
        ),
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#FAF5FF", edgecolor="#B794F4", alpha=0.95, lw=0.9)
    )

    # Lower callout: CBA-PhenoNet with upward-arching leader line
    ax.annotate(
        "CBA-PhenoNet (Ours) (0.27 ms, 0.897)",
        xy=(0.27, 0.897),
        xytext=(0.44, 0.855),
        fontsize=9.5,
        fontweight="bold",
        color="#9B2C2C",
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=0.08",
            color="#E53E3E",
            lw=1.4,
            shrinkA=0,
            shrinkB=6
        ),
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#FFF5F5", edgecolor="#FEB2B2", alpha=0.95, lw=1.1)
    )

    # Set x-scale to log scale because latency ranges from 0.2 to 62 ms
    ax.set_xscale("log")
    ax.set_xlabel("Inference Latency (ms per season) [Log Scale]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Segmentation Macro-F1", fontsize=11, fontweight="bold")
    ax.set_title("Accuracy vs. Inference Latency Trade-off across 6 Architectures", fontsize=12, fontweight="bold", pad=12)

    # Sweet spot highlight (top-left)
    ax.axvspan(0.15, 1.0, alpha=0.10, color="#38A169", label="Optimal Efficiency Zone (DL: <1 ms, F1 > 0.6)")

    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.set_ylim(0.35, 1.01)
    ax.set_xlim(0.15, 100)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

    out_dir = Path("results/plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "accuracy_vs_latency_tradeoff.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved tradeoff plot to: {out_path}")

if __name__ == "__main__":
    create_tradeoff_plot()
