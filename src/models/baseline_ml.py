"""Member assigned to Random Forest / XGBoost baseline. Thin wrappers so this
model reports through the exact same evaluation.metrics interface as the 3 DL
models -- see src/train.py for the shared harness that calls into either side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ..data.features import FEATURE_COLUMNS, build_feature_table
from ..data.dataset import SeasonRecord


def build_rf_baseline(**kwargs) -> RandomForestClassifier:
    """TODO (member owning this model): tune n_estimators/max_depth/class_weight
    (labels are imbalanced -- FALLOW/VEGETATIVE dominate, REPRODUCTIVE is the
    hardest/rarest class, same imbalance shape as the NLP hate-speech project's
    CLEAN/OFFENSIVE/HATE split -- consider class_weight='balanced' as a first try,
    matching Ch.4's Random Forest content in the syllabus)."""
    defaults = dict(n_estimators=300, max_depth=None, class_weight="balanced", n_jobs=-1, random_state=42)
    defaults.update(kwargs)
    return RandomForestClassifier(**defaults)


def build_xgboost_baseline(**kwargs):
    """TODO (member owning this model): requires `pip install xgboost`. Same
    imbalance note as build_rf_baseline -- XGBoost doesn't take class_weight
    directly, use `sample_weight` in .fit() or scale_pos_weight per class."""
    from xgboost import XGBClassifier
    defaults = dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                     objective="multi:softprob", num_class=4, random_state=42, n_jobs=-1)
    defaults.update(kwargs)
    return XGBClassifier(**defaults)


def predict_tabular(model, test_records: list[SeasonRecord]):
    """Predict-ONLY (model must already be fit) -- used both for the main
    evaluation pass and for timing pure inference (see fit_predict_tabular,
    and src/train.py::train_ml_baseline which measures inference separately
    from training). Returns (y_true, y_pred, doy, ndvi, mask) shaped
    (n_test_seasons, T), matching build_eval_result()'s expected shape."""
    max_len = max(len(r.ndvi_smoothed) for r in test_records)
    y_true = np.full((len(test_records), max_len), -100, dtype=np.int64)
    y_pred = np.full((len(test_records), max_len), -100, dtype=np.int64)
    doy = np.zeros((len(test_records), max_len), dtype=np.int64)
    ndvi = np.zeros((len(test_records), max_len), dtype=np.float32)
    mask = np.zeros((len(test_records), max_len), dtype=bool)

    for i, record in enumerate(test_records):
        rows = build_feature_table([record])
        preds = model.predict(rows[FEATURE_COLUMNS])
        t = len(record.labels)
        y_true[i, :t] = record.labels
        y_pred[i, :t] = preds
        doy[i, :t] = record.doy
        ndvi[i, :t] = record.ndvi_smoothed
        mask[i, :t] = True

    return y_true, y_pred, doy, ndvi, mask


def fit_predict_tabular(model, train_records: list[SeasonRecord], test_records: list[SeasonRecord]):
    """Fits `model` then predicts on test_records -- the SAME shape
    build_eval_result() in evaluation/metrics.py expects, so RF/XGBoost
    results slot into the identical comparison table as the DL models despite
    the different (tabular vs sequence) internal representation."""
    train_df = build_feature_table(train_records)
    model.fit(train_df[FEATURE_COLUMNS], train_df["label"])
    return predict_tabular(model, test_records)
