"""Shared evaluation utilities -- every team member's model MUST report through
this module so the 5 sets of results are directly comparable (see docs/methodology.md §1.4).

Two metric families:
  1. Segmentation quality: accuracy, macro-F1, per-class F1 on the per-timestep labels.
  2. Phenology-date accuracy: MAE/RMSE/R2 (in DAYS) on SOS/POS/EOS derived from the
     predicted label sequence, against ground-truth SOS/POS/EOS day-of-year.
Plus compute cost: training time and inference time, via the `Timer` context manager.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
)

LABEL_NAMES = ["FALLOW", "VEGETATIVE", "REPRODUCTIVE", "SENESCENCE"]
NUM_CLASSES = len(LABEL_NAMES)


@contextmanager
def Timer():
    """Usage:
        with Timer() as t:
            model.fit(...)
        train_seconds = t.elapsed
    """
    state = {"elapsed": None}
    start = time.perf_counter()
    try:
        yield state
    finally:
        state["elapsed"] = time.perf_counter() - start


@dataclass
class EvalResult:
    model_name: str
    member_name: str
    accuracy: float
    macro_f1: float
    per_class_f1: dict = field(default_factory=dict)
    sos_mae_days: float | None = None
    sos_rmse_days: float | None = None
    sos_r2: float | None = None
    sos_mape_pct: float | None = None
    sos_medae_days: float | None = None
    sos_acc1_pct: float | None = None
    sos_acc3_pct: float | None = None
    pos_mae_days: float | None = None
    pos_rmse_days: float | None = None
    pos_r2: float | None = None
    pos_mape_pct: float | None = None
    pos_medae_days: float | None = None
    pos_acc1_pct: float | None = None
    pos_acc3_pct: float | None = None
    eos_mae_days: float | None = None
    eos_rmse_days: float | None = None
    eos_r2: float | None = None
    eos_mape_pct: float | None = None
    eos_medae_days: float | None = None
    eos_acc1_pct: float | None = None
    eos_acc3_pct: float | None = None
    train_seconds: float | None = None
    inference_ms_per_sample: float | None = None
    n_test_seasons: int | None = None

    def to_row(self) -> dict:
        """Flat dict for appending to results/comparison_table.csv."""
        return asdict(self)


def segmentation_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """y_true, y_pred: (n_seasons, T) int label arrays. mask: (n_seasons, T) bool,
    True = real timestep, False = padding -- excluded from the score. Flattens to
    1D internally (per-timestep comparison), matching the "point-wise segmentation"
    framing in docs/methodology.md."""
    if mask is not None:
        y_true = y_true[mask]
        y_pred = y_pred[mask]
    else:
        y_true = y_true.ravel()
        y_pred = y_pred.ravel()

    per_class = f1_score(y_true, y_pred, labels=list(range(NUM_CLASSES)), average=None, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class_f1": dict(zip(LABEL_NAMES, per_class.tolist())),
    }


def derive_dates_from_labels(label_seq: np.ndarray, doy_seq: np.ndarray, ndvi_seq: np.ndarray) -> dict:
    """label_seq: (T,) predicted (or true) class ids for ONE season.
    doy_seq: (T,) day-of-year for each timestep (same length).
    ndvi_seq: (T,) smoothed NDVI for each timestep (used to locate POS precisely).

    Returns {'sos': doy or None, 'pos': doy or None, 'eos': doy or None} -- None when
    a transition doesn't occur in this season (e.g. no FALLOW before SOS in a
    fixed window that starts mid-vegetative -- flag and exclude from that metric
    rather than guessing).
    """
    sos = pos = eos = None

    for t in range(1, len(label_seq)):
        if label_seq[t - 1] == 0 and label_seq[t] in (1, 2, 3):
            sos = doy_seq[t]
            break

    veg_or_repro = np.where(np.isin(label_seq, [1, 2]))[0]
    if len(veg_or_repro) > 0:
        peak_idx = veg_or_repro[np.argmax(ndvi_seq[veg_or_repro])]
        pos = doy_seq[peak_idx]

    for t in range(1, len(label_seq)):
        if label_seq[t - 1] == 3 and label_seq[t] == 0:
            eos = doy_seq[t]
            break

    return {"sos": sos, "pos": pos, "eos": eos}


def accuracy_within_days(true_arr: np.ndarray, pred_arr: np.ndarray, tolerance_days: float) -> float:
    """% of predictions within `tolerance_days` of the true DOY. Complements
    MAE/RMSE (which can be dominated by a few large misses) with a metric
    that answers the report-friendly question "how often are we basically
    right" -- e.g. accuracy_within_days(..., 1) = "Accuracy +/-1 ngay"."""
    return float(np.mean(np.abs(true_arr - pred_arr) <= tolerance_days) * 100)


def phenology_date_metrics(
    pred_dates: list[dict], true_dates: list[dict],
) -> dict:
    """pred_dates/true_dates: list of {'sos':..,'pos':..,'eos':..} dicts, one per test
    season (output of derive_dates_from_labels, applied to predictions and to ground
    truth respectively). Rows where either side is None for a given milestone are
    dropped from THAT milestone's metric only (reported separately, see
    docs/methodology.md -- do not impute a fake date).

    Reports MAE/RMSE/R2 (original) plus MAPE, MedAE, and Accuracy@{1,3} days
    (added for the instructor-requested deeper-evaluation pass)."""
    out = {}
    for key in ("sos", "pos", "eos"):
        pairs = [(p[key], t[key]) for p, t in zip(pred_dates, true_dates) if p[key] is not None and t[key] is not None]
        if not pairs:
            out[key] = {
                "mae": None, "rmse": None, "r2": None,
                "mape": None, "medae": None, "acc1": None, "acc3": None, "n": 0,
            }
            continue
        pred_arr = np.array([p for p, _ in pairs], dtype=float)
        true_arr = np.array([t for _, t in pairs], dtype=float)
        out[key] = {
            "mae": mean_absolute_error(true_arr, pred_arr),
            "rmse": mean_squared_error(true_arr, pred_arr) ** 0.5,
            "r2": r2_score(true_arr, pred_arr) if len(pairs) > 1 else None,
            # MAPE as a percentage (sklearn returns a 0-1 fraction) -- DOY is
            # always > 0 here (1-365), so no divide-by-zero risk.
            "mape": mean_absolute_percentage_error(true_arr, pred_arr) * 100,
            "medae": median_absolute_error(true_arr, pred_arr),
            "acc1": accuracy_within_days(true_arr, pred_arr, 1),
            "acc3": accuracy_within_days(true_arr, pred_arr, 3),
            "n": len(pairs),
        }
    return out


def circular_abs_error(true_doy: float, pred_doy: float, period: float = 365.0) -> float:
    """DOY is cyclical -- day 365 and day 1 are 1 day apart, not 364. A plain
    |pred - true| overstates the error for any pair straddling the year
    boundary (found empirically: several of the "worst" error_analysis.py
    cases were true~=346/pred~=1-type pairs scoring ~345 days off by plain
    subtraction, vs. their real ~20-day distance) -- this is the corrected
    distance used for ranking/reporting worst-case rows."""
    raw = abs(pred_doy - true_doy)
    return min(raw, period - raw)


def per_season_date_table(
    pred_dates: list[dict], true_dates: list[dict], season_ids: list[str],
) -> pd.DataFrame:
    """Long-format table: one row per (season, milestone) with true/pred DOY and
    signed/absolute error -- the raw material for visualize.py's scatter plots
    and the error-analysis top-N-worst-cases report. Rows where a milestone
    wasn't detected (None) are dropped, same rule as phenology_date_metrics.

    Reports both `abs_error_days` (plain difference, matches the MAE/RMSE
    already in comparison_table.csv) and `circular_abs_error_days` (year-
    wraparound-aware, see circular_abs_error docstring) -- use the latter for
    worst-case ranking (src/error_analysis.py) since a handful of "worst"
    plain-difference rows are actually near-perfect predictions that happen to
    straddle Dec 31 -> Jan 1."""
    rows = []
    for season_id, p, t in zip(season_ids, pred_dates, true_dates):
        for milestone in ("sos", "pos", "eos"):
            if p[milestone] is None or t[milestone] is None:
                continue
            true_doy, pred_doy = float(t[milestone]), float(p[milestone])
            rows.append({
                "season_id": season_id,
                "milestone": milestone,
                "true_doy": true_doy,
                "pred_doy": pred_doy,
                "signed_error_days": pred_doy - true_doy,
                "abs_error_days": abs(pred_doy - true_doy),
                "circular_abs_error_days": circular_abs_error(true_doy, pred_doy),
            })
    return pd.DataFrame(rows)


def build_eval_result(
    model_name: str,
    member_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    doy: np.ndarray,
    ndvi: np.ndarray,
    mask: np.ndarray | None,
    train_seconds: float,
    inference_ms_per_sample: float,
    season_ids: list[str] | None = None,
) -> tuple[EvalResult, pd.DataFrame]:
    """Convenience wrapper a team member calls ONCE per trained model to get a
    fully-populated, directly-comparable result row PLUS the long-format
    per-season prediction table (for visualize.py / error analysis). See
    src/train.py for the expected call site.

    Returns (EvalResult, predictions_df). If `season_ids` isn't provided,
    synthetic ids ("season_0", "season_1", ...) are used -- the predictions
    table is still produced, just without traceable real parcel ids."""
    seg = segmentation_metrics(y_true, y_pred, mask)

    n_seasons = y_true.shape[0]
    if season_ids is None:
        season_ids = [f"season_{i}" for i in range(n_seasons)]
    pred_dates, true_dates = [], []
    for i in range(n_seasons):
        m = mask[i] if mask is not None else slice(None)
        pred_dates.append(derive_dates_from_labels(y_pred[i][m], doy[i][m], ndvi[i][m]))
        true_dates.append(derive_dates_from_labels(y_true[i][m], doy[i][m], ndvi[i][m]))
    date_metrics = phenology_date_metrics(pred_dates, true_dates)
    predictions_df = per_season_date_table(pred_dates, true_dates, season_ids)

    result = EvalResult(
        model_name=model_name,
        member_name=member_name,
        accuracy=seg["accuracy"],
        macro_f1=seg["macro_f1"],
        per_class_f1=seg["per_class_f1"],
        sos_mae_days=date_metrics["sos"]["mae"],
        sos_rmse_days=date_metrics["sos"]["rmse"],
        sos_r2=date_metrics["sos"]["r2"],
        sos_mape_pct=date_metrics["sos"]["mape"],
        sos_medae_days=date_metrics["sos"]["medae"],
        sos_acc1_pct=date_metrics["sos"]["acc1"],
        sos_acc3_pct=date_metrics["sos"]["acc3"],
        pos_mae_days=date_metrics["pos"]["mae"],
        pos_rmse_days=date_metrics["pos"]["rmse"],
        pos_r2=date_metrics["pos"]["r2"],
        pos_mape_pct=date_metrics["pos"]["mape"],
        pos_medae_days=date_metrics["pos"]["medae"],
        pos_acc1_pct=date_metrics["pos"]["acc1"],
        pos_acc3_pct=date_metrics["pos"]["acc3"],
        eos_mae_days=date_metrics["eos"]["mae"],
        eos_rmse_days=date_metrics["eos"]["rmse"],
        eos_r2=date_metrics["eos"]["r2"],
        eos_mape_pct=date_metrics["eos"]["mape"],
        eos_medae_days=date_metrics["eos"]["medae"],
        eos_acc1_pct=date_metrics["eos"]["acc1"],
        eos_acc3_pct=date_metrics["eos"]["acc3"],
        train_seconds=train_seconds,
        inference_ms_per_sample=inference_ms_per_sample,
        n_test_seasons=n_seasons,
    )
    return result, predictions_df
