"""Task 3 (instructor-requested deep-evaluation extension): find the worst
prediction cases so the report can discuss WHY the model fails on them.

Reads `results/predictions/<model>_predictions.csv` (see src/visualize.py for
the shared loader), takes the top N% by absolute error (across all three
milestones pooled together, since a bad SOS call is just as informative for
error analysis as a bad EOS call), and writes results/error_analysis.csv with
the parcel id (season_id), crop type, milestone, true/predicted DOY, and the
error -- ready to paste into the report.

Usage:
    python -m src.error_analysis --model cba_phenonet --top-pct 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .visualize import load_predictions

ERROR_ANALYSIS_PATH = Path(__file__).resolve().parent.parent / "results" / "error_analysis.csv"


def find_top_error_cases(model_name: str, top_pct: float = 5.0) -> pd.DataFrame:
    """Returns the top `top_pct`% of (season, milestone) rows by
    circular_abs_error_days, sorted worst-first. `top_pct=5` means the worst
    5% of all predictions (pooled across SOS/POS/EOS), matching the
    instructor's "Top 5%" request.

    Ranks by the CIRCULAR error (see evaluation.metrics.circular_abs_error),
    not the plain abs_error_days column -- ranking by the plain difference
    was found to surface false positives: predictions like true=day 346,
    pred=day 1 score ~345 "days off" by subtraction but are actually only
    ~20 real days apart once the Dec 31 -> Jan 1 wraparound is accounted for.
    Both columns are kept in the output so this is auditable, not hidden."""
    df = load_predictions(model_name).copy()
    df["crop_type"] = df["season_id"].str.rsplit("_", n=1).str[-1]

    n_total = len(df)
    n_top = max(1, int(round(n_total * top_pct / 100.0)))
    worst = df.sort_values("circular_abs_error_days", ascending=False).head(n_top).reset_index(drop=True)

    cols = ["season_id", "crop_type", "milestone", "true_doy", "pred_doy",
            "signed_error_days", "abs_error_days", "circular_abs_error_days"]
    return worst[cols]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model name, e.g. cba_phenonet, transformer, rf")
    parser.add_argument("--top-pct", type=float, default=5.0, help="percentage of worst cases to keep (default: 5)")
    args = parser.parse_args()

    worst = find_top_error_cases(args.model, top_pct=args.top_pct)
    ERROR_ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    worst.to_csv(ERROR_ANALYSIS_PATH, index=False)

    print(f"Model: {args.model} -- top {args.top_pct}% worst cases ({len(worst)} rows) saved to {ERROR_ANALYSIS_PATH}")
    print(worst.head(10).to_string(index=False))
    print("\nWorst-case crop-type distribution (does one crop dominate the errors?):")
    print(worst["crop_type"].value_counts())
    print("\nWorst-case milestone distribution (is one milestone systematically harder?):")
    print(worst["milestone"].value_counts())


if __name__ == "__main__":
    main()
