"""Task 2 (instructor-requested deep-evaluation extension): Actual-vs-Predicted
scatter plots for the best model, against the ideal y=x line.

Reads `results/predictions/<model>_predictions.csv` (written by src/train.py's
main() -- one row per (season, milestone), columns: season_id, milestone,
true_doy, pred_doy, signed_error_days, abs_error_days) and plots true_doy (X)
vs pred_doy (Y) per milestone (SOS/POS/EOS) plus a combined view, saving PNGs
to results/plots/.

Usage:
    python -m src.visualize --model cba_phenonet
    python -m src.visualize --model cba_phenonet --milestone pos   # single milestone only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless -- this repo runs from a CLI, not a notebook kernel
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score

PREDICTIONS_DIR = Path(__file__).resolve().parent.parent / "results" / "predictions"
PLOTS_DIR = Path(__file__).resolve().parent.parent / "results" / "plots"

MILESTONE_LABELS = {"sos": "SOS (Sowing)", "pos": "POS (Peak/Flowering)", "eos": "EOS (Harvest)"}
MILESTONE_COLORS = {"sos": "#4C72B0", "pos": "#DD8452", "eos": "#55A868"}


def load_predictions(model_name: str) -> pd.DataFrame:
    path = PREDICTIONS_DIR / f"{model_name}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python -m src.train --model {model_name} --member <ten>` first "
            f"(src/train.py's main() writes this file as a side effect of evaluation)."
        )
    return pd.read_csv(path)


def _scatter_with_identity_line(ax, true_vals, pred_vals, color, title):
    lo = min(true_vals.min(), pred_vals.min())
    hi = max(true_vals.max(), pred_vals.max())
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", color="gray", linewidth=1.5, label="y = x (ideal)")
    ax.scatter(true_vals, pred_vals, alpha=0.5, s=18, color=color, edgecolors="none")
    r2 = r2_score(true_vals, pred_vals) if len(true_vals) > 1 else float("nan")
    mae = (pred_vals - true_vals).abs().mean()
    ax.set_title(f"{title}\nR2={r2:.3f}, MAE={mae:.1f} days, n={len(true_vals)}")
    ax.set_xlabel("Actual DOY")
    ax.set_ylabel("Predicted DOY")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", fontsize=8)


def plot_actual_vs_predicted(model_name: str, milestone: str | None = None) -> list[Path]:
    """Saves one PNG per milestone (or just the requested one) plus a combined
    3-panel figure, to results/plots/. Returns the list of saved paths."""
    df = load_predictions(model_name)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    milestones = [milestone] if milestone else ["sos", "pos", "eos"]
    saved = []

    # Combined 1x3 panel (or 1x1 if a single milestone was requested)
    fig, axes = plt.subplots(1, len(milestones), figsize=(6 * len(milestones), 6))
    if len(milestones) == 1:
        axes = [axes]
    for ax, m in zip(axes, milestones):
        sub = df[df["milestone"] == m]
        if sub.empty:
            ax.set_title(f"{MILESTONE_LABELS[m]}\n(no data)")
            continue
        _scatter_with_identity_line(ax, sub["true_doy"], sub["pred_doy"], MILESTONE_COLORS[m], MILESTONE_LABELS[m])
    fig.suptitle(f"Actual vs. Predicted phenology dates -- {model_name}", fontsize=13)
    fig.tight_layout()
    combined_path = PLOTS_DIR / f"{model_name}_actual_vs_predicted.png"
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    saved.append(combined_path)
    print(f"Saved {combined_path}")

    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model name, e.g. cba_phenonet, transformer, rf")
    parser.add_argument("--milestone", default=None, choices=["sos", "pos", "eos"], help="plot only one milestone (default: all 3)")
    args = parser.parse_args()
    plot_actual_vs_predicted(args.model, milestone=args.milestone)


if __name__ == "__main__":
    main()
