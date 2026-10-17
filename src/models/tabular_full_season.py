"""Full-Season Tabular ML Baselines (RF-FullSeason and XGBoost-FullSeason):

Equips Random Forest and XGBoost with FULL SEASON CONTEXT (both forward leads and
whole-season summary statistics) to ensure fair comparison against bidirectional
sequential Deep Learning models.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.data.dataset import SeasonRecord, cyclical_doy
from src.evaluation.metrics import build_eval_result


def build_full_season_feature_table(records: list[SeasonRecord]) -> pd.DataFrame:
    """Builds a rich tabular representation where each timestep has access to both
    backward lags, forward leads, and global whole-season statistical descriptors."""
    dfs = []
    for rec in records:
        ndvi = pd.Series(rec.ndvi_smoothed)
        t_len = len(ndvi)
        doy_sin, doy_cos = cyclical_doy(rec.doy).T

        # Season global statistics
        s_min = float(ndvi.min())
        s_max = float(ndvi.max())
        s_mean = float(ndvi.mean())
        s_std = float(ndvi.std())
        s_amp = max(s_max - s_min, 1e-4)
        s_p10 = float(np.percentile(ndvi, 10))
        s_p50 = float(np.percentile(ndvi, 50))
        s_p90 = float(np.percentile(ndvi, 90))

        diffs = ndvi.diff().fillna(0)
        max_slope_up = float(diffs.max())
        max_slope_down = float(diffs.min())

        df = pd.DataFrame({
            "season_id": rec.season_id,
            "t": np.arange(t_len),
            "t_norm": np.arange(t_len) / float(t_len - 1),
            "ndvi": ndvi.values,
            "doy_sin": doy_sin,
            "doy_cos": doy_cos,
            # Backward lags
            "lag_1": ndvi.shift(1).bfill().values,
            "lag_2": ndvi.shift(2).bfill().values,
            "lag_3": ndvi.shift(3).bfill().values,
            # Forward leads (bidirectional / future context)
            "lead_1": ndvi.shift(-1).ffill().values,
            "lead_2": ndvi.shift(-2).ffill().values,
            "lead_3": ndvi.shift(-3).ffill().values,
            # Centered and causal rolling
            "rolling_mean_3": ndvi.rolling(3, min_periods=1, center=True).mean().values,
            "rolling_std_3": ndvi.rolling(3, min_periods=1, center=True).std().fillna(0).values,
            "rolling_mean_5": ndvi.rolling(5, min_periods=1, center=True).mean().values,
            "rolling_std_5": ndvi.rolling(5, min_periods=1, center=True).std().fillna(0).values,
            # Global season features
            "s_min": s_min,
            "s_max": s_max,
            "s_mean": s_mean,
            "s_std": s_std,
            "s_amp": s_amp,
            "s_p10": s_p10,
            "s_p50": s_p50,
            "s_p90": s_p90,
            "s_max_slope_up": max_slope_up,
            "s_max_slope_down": max_slope_down,
            "label": rec.labels,
        })
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


FULL_FEATURE_COLS = [
    "t_norm", "ndvi", "doy_sin", "doy_cos",
    "lag_1", "lag_2", "lag_3", "lead_1", "lead_2", "lead_3",
    "rolling_mean_3", "rolling_std_3", "rolling_mean_5", "rolling_std_5",
    "s_min", "s_max", "s_mean", "s_std", "s_amp",
    "s_p10", "s_p50", "s_p90", "s_max_slope_up", "s_max_slope_down"
]


def train_eval_full_season_tabular(model_type: str = "rf", n_jobs: int = 4):
    """Trains Full-Season RF or XGBoost on train_records and evaluates on test_records."""
    from src.train import load_records

    train_recs = load_records("train")
    test_recs = load_records("test")

    train_df = build_full_season_feature_table(train_recs)
    test_df = build_full_season_feature_table(test_recs)

    X_train = train_df[FULL_FEATURE_COLS]
    y_train = train_df["label"]

    X_test = test_df[FULL_FEATURE_COLS]
    y_test = test_df["label"]

    if model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=n_jobs)
    elif model_type == "xgboost":
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=n_jobs)
    else:
        raise ValueError(model_type)

    t_fit_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t_fit_start

    t_infer_start = time.perf_counter()
    preds_flat = model.predict(X_test)
    total_infer_time = time.perf_counter() - t_infer_start
    infer_ms_per_sample = (total_infer_time / len(test_recs)) * 1000.0

    # Reshape predictions back to per-season sequences
    y_true_list, y_pred_list, doy_list, ndvi_list, mask_list = [], [], [], [], []
    test_df["pred"] = preds_flat
    for rec in test_recs:
        sub = test_df[test_df["season_id"] == rec.season_id]
        y_true_list.append(sub["label"].values)
        y_pred_list.append(sub["pred"].values)
        doy_list.append(rec.doy)
        ndvi_list.append(rec.ndvi_smoothed)
        mask_list.append(np.ones(len(rec.doy), dtype=bool))

    # Pad sequences to max_len
    max_len = max(len(s) for s in y_true_list)
    n_seasons = len(test_recs)
    y_true_arr = np.full((n_seasons, max_len), -100, dtype=np.int64)
    y_pred_arr = np.full((n_seasons, max_len), -100, dtype=np.int64)
    doy_arr = np.zeros((n_seasons, max_len), dtype=np.int64)
    ndvi_arr = np.zeros((n_seasons, max_len), dtype=np.float32)
    mask_arr = np.zeros((n_seasons, max_len), dtype=bool)

    for i in range(n_seasons):
        L = len(y_true_list[i])
        y_true_arr[i, :L] = y_true_list[i]
        y_pred_arr[i, :L] = y_pred_list[i]
        doy_arr[i, :L] = doy_list[i]
        ndvi_arr[i, :L] = ndvi_list[i]
        mask_arr[i, :L] = True

    model_label = "RF-FullSeason" if model_type == "rf" else "XGBoost-FullSeason"
    result, pred_df = build_eval_result(
        model_name=model_label,
        member_name=f"Baseline Tabular ({model_label})",
        y_true=y_true_arr, y_pred=y_pred_arr, doy=doy_arr, ndvi=ndvi_arr, mask=mask_arr,
        train_seconds=train_seconds, inference_ms_per_sample=infer_ms_per_sample,
        season_ids=[r.season_id for r in test_recs]
    )
    return result, pred_df
