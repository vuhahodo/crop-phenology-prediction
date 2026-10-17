"""Double-Logistic (TIMESAT-style / Beck et al. 2006 / Zhang et al. 2003) Phenology Extraction Baseline:

Fits a smooth parametric double-logistic curve to each season's NDVI time series:
    y(t) = v_min + (v_max - v_min) * ( 1 / (1 + exp(-rsp * (t - sos_mid))) + 1 / (1 + exp(rau * (t - eos_mid))) - 1 )

Then extracts:
- POS: peak of the fitted double-logistic curve
- SOS: 20% seasonal amplitude threshold crossing on the green-up limb
- EOS: 20% seasonal amplitude threshold crossing on the senescence limb

This matches the classical remote sensing methodology (TIMESAT / DATimeS) evaluated in
Prikaziuk et al. (2025) and Zhang et al. (2022).
"""
from __future__ import annotations

import time
import numpy as np
from scipy.optimize import curve_fit

from src.data.dataset import SeasonRecord
from src.evaluation.metrics import phenology_date_metrics, derive_dates_from_labels


def double_logistic(t: np.ndarray, v_min: float, v_max: float,
                    sos_mid: float, rsp: float,
                    eos_mid: float, rau: float) -> np.ndarray:
    """Beck et al. (2006) / Zhang et al. (2003) standard double-logistic formulation."""
    # Clip exponents to avoid numerical overflow
    exp_up = np.clip(-rsp * (t - sos_mid), -50.0, 50.0)
    exp_down = np.clip(rau * (t - eos_mid), -50.0, 50.0)
    logistic_up = 1.0 / (1.0 + np.exp(exp_up))
    logistic_down = 1.0 / (1.0 + np.exp(exp_down))
    return v_min + (v_max - v_min) * (logistic_up + logistic_down - 1.0)


def fit_season_double_logistic(doy: np.ndarray, ndvi: np.ndarray) -> tuple[dict, bool]:
    """Fits double logistic to one parcel season.
    Returns:
        dict: {'sos': doy, 'pos': doy, 'eos': doy}
        bool: success flag (True if curve_fit converged, False if fallback was needed)
    """
    ndvi_min_raw = float(np.percentile(ndvi, 5))
    ndvi_max_raw = float(np.percentile(ndvi, 95))
    peak_idx = int(np.argmax(ndvi))
    peak_doy = float(doy[peak_idx])

    # Initial guesses
    p0 = [
        max(0.05, ndvi_min_raw),
        min(0.95, max(ndvi_max_raw, ndvi_min_raw + 0.15)),
        max(float(doy[0]), peak_doy - 40.0),
        0.08,
        min(float(doy[-1]), peak_doy + 40.0),
        0.08,
    ]
    bounds = (
        [0.0, 0.3, float(doy[0]), 0.005, float(doy[0]), 0.005],
        [0.4, 1.0, float(doy[-1]), 0.8, float(doy[-1]), 0.8],
    )

    try:
        popt, _ = curve_fit(double_logistic, doy, ndvi, p0=p0, bounds=bounds, maxfev=600)
        v_min, v_max, sos_mid, rsp, eos_mid, rau = popt
        # Dense evaluation over regular grid
        dense_doy = np.arange(int(doy[0]), int(doy[-1]) + 1, dtype=float)
        fitted_ndvi = double_logistic(dense_doy, *popt)
        success = True
    except Exception:
        # Fallback to smoothed spline/direct curve if optimization does not converge
        dense_doy = doy.astype(float)
        fitted_ndvi = ndvi
        success = False

    # Extract milestones using standard 20% amplitude threshold
    f_min = float(np.min(fitted_ndvi))
    f_max = float(np.max(fitted_ndvi))
    amplitude = max(f_max - f_min, 1e-4)
    th_20 = f_min + 0.20 * amplitude
    fit_peak_idx = int(np.argmax(fitted_ndvi))
    pos_doy = dense_doy[fit_peak_idx]

    # Search backward from peak for SOS
    sos_doy = None
    for i in range(fit_peak_idx, -1, -1):
        if fitted_ndvi[i] <= th_20:
            sos_doy = dense_doy[i]
            break
    if sos_doy is None and fit_peak_idx > 0:
        sos_doy = dense_doy[0]

    # Search forward from peak for EOS
    eos_doy = None
    for i in range(fit_peak_idx, len(fitted_ndvi)):
        if fitted_ndvi[i] <= th_20:
            eos_doy = dense_doy[i]
            break
    if eos_doy is None and fit_peak_idx < len(fitted_ndvi) - 1:
        eos_doy = dense_doy[-1]

    return {"sos": sos_doy, "pos": pos_doy, "eos": eos_doy}, success


def evaluate_double_logistic(test_records: list[SeasonRecord]) -> dict:
    """Evaluates Double-Logistic baseline on test records, reporting MAE/RMSE and latency."""
    pred_dates = []
    true_dates = []
    n_success = 0

    t_start = time.perf_counter()
    for rec in test_records:
        # True milestones from ground truth label sequence
        t_milestones = derive_dates_from_labels(rec.labels, rec.doy, rec.ndvi_smoothed)
        true_dates.append(t_milestones)

        # Fit Double Logistic
        p_milestones, success = fit_season_double_logistic(rec.doy, rec.ndvi_smoothed)
        pred_dates.append(p_milestones)
        if success:
            n_success += 1

    total_time = time.perf_counter() - t_start
    latency_ms_per_season = (total_time / len(test_records)) * 1000.0

    metrics = phenology_date_metrics(pred_dates, true_dates)
    metrics["latency_ms_per_season"] = latency_ms_per_season
    metrics["total_time_seconds"] = total_time
    metrics["fit_convergence_rate_pct"] = (n_success / len(test_records)) * 100.0
    return metrics
