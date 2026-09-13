"""Member assigned to Random Forest / XGBoost baseline: this builds the flat
feature table these models need (see docs/methodology.md §1.2 -- RF/XGBoost get
a hand-engineered tabular representation, not the raw sequence the DL models
consume; that difference IS the point of the benchmark)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import SeasonRecord, cyclical_doy


def season_to_rows(record: SeasonRecord, lags: tuple[int, ...] = (1, 2, 3), rolling_windows: tuple[int, ...] = (3, 5)) -> pd.DataFrame:
    """One SeasonRecord -> one DataFrame with one row per timestep. Lag/rolling
    features only look BACKWARD (t-1, t-2, ... and rolling windows ending at t)
    -- never forward, matching realistic deployment (you don't have future NDVI
    when deciding "is this timestep vegetative or reproductive" in near-real-time
    monitoring) and avoiding target leakage."""
    ndvi = pd.Series(record.ndvi_smoothed)
    t = len(ndvi)
    doy_sin, doy_cos = cyclical_doy(record.doy).T

    df = pd.DataFrame({
        "season_id": record.season_id,
        "t": np.arange(t),
        "ndvi": ndvi.values,
        "doy_sin": doy_sin,
        "doy_cos": doy_cos,
        "delta_ndvi_1": ndvi.diff().fillna(0).values,
        "label": record.labels,
    })
    for lag in lags:
        df[f"lag_{lag}"] = ndvi.shift(lag).bfill().values
    for w in rolling_windows:
        df[f"rolling_mean_{w}"] = ndvi.rolling(w, min_periods=1).mean().values
        df[f"rolling_std_{w}"] = ndvi.rolling(w, min_periods=1).std().fillna(0).values
    return df


def build_feature_table(records: list[SeasonRecord], **kwargs) -> pd.DataFrame:
    """Concatenates season_to_rows() over every season. `season_id` stays in the
    output so train/val/test splitting can group by it (GroupKFold) -- see
    docs/methodology.md §1.3. Drop it (and `t`) before fitting, keep it for the
    split and for reconstructing per-season predictions afterward."""
    return pd.concat([season_to_rows(r, **kwargs) for r in records], ignore_index=True)


FEATURE_COLUMNS = [
    "ndvi", "doy_sin", "doy_cos", "delta_ndvi_1",
    "lag_1", "lag_2", "lag_3",
    "rolling_mean_3", "rolling_std_3", "rolling_mean_5", "rolling_std_5",
]
