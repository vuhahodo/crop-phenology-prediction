"""B0c — Train and save predictions for Baseline ML models (RF, XGBoost).

Double-Logistic is a purely analytical method (curve fitting per season), 
so it doesn't need "training", but we will run it and save its predictions
in the same format so make_tables.py can read it.
"""
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit

from src.train import load_records, Timer
from src.data.features import FEATURE_COLUMNS, build_feature_table
from src.models.baseline_ml import build_rf_baseline, build_xgboost_baseline, predict_tabular

PREDS_DIR = Path("results/preds")


# Double Logistic function (TIMESAT style)
def double_logistic(t, w, a, m1, m2, m3, m4):
    """
    w: base level
    a: amplitude
    m1: midpoint of ascending phase (SOS)
    m2: rate of ascending phase
    m3: midpoint of descending phase (EOS)
    m4: rate of descending phase
    """
    return w + a * (1 / (1 + np.exp(-m2 * (t - m1))) - 1 / (1 + np.exp(-m4 * (t - m3))))


def fit_double_logistic(ndvi_seq, doy_seq):
    """Fits double logistic to one season's NDVI and extracts dates."""
    # Scale DOY to 0-1 for stability in curve fitting
    t = doy_seq / 365.0
    y = ndvi_seq
    
    # Initial guesses
    w0 = np.clip(np.min(y), -0.99, 0.99)
    a0 = np.clip(np.max(y) - w0, 0.01, 1.99)
    m1_0 = 100 / 365.0
    m2_0 = 15.0
    m3_0 = 250 / 365.0
    m4_0 = 15.0
    p0 = [w0, a0, m1_0, m2_0, m3_0, m4_0]
    
    # Bounds
    bounds = (
        [-1, 0, 0, 1, 0, 1],
        [1, 2, 1, 50, 1, 50]
    )
    
    try:
        popt, _ = curve_fit(double_logistic, t, y, p0=p0, bounds=bounds, maxfev=1000)
        # SOS is often defined as the inflection point m1
        sos_doy = int(popt[2] * 365)
        # EOS is the inflection point m3
        eos_doy = int(popt[4] * 365)
        # POS is the peak of the curve
        y_fit = double_logistic(t, *popt)
        pos_doy = int(doy_seq[np.argmax(y_fit)])
        
        # Ensure within 1-365
        sos_doy = np.clip(sos_doy, 1, 365)
        pos_doy = np.clip(pos_doy, 1, 365)
        eos_doy = np.clip(eos_doy, 1, 365)
        
        return sos_doy, pos_doy, eos_doy
    except RuntimeError:
        # Fallback to simple max if curve_fit fails
        pos_doy = int(doy_seq[np.argmax(y)])
        return None, pos_doy, None


def run_double_logistic(test_records):
    """Run DL on test set. Note DL doesn't output per-timestep classes,
    so we can't save a y_pred array that works with unified decode. 
    Instead we will mock y_pred such that the unified decode extracts exactly
    the DL-predicted dates!
    """
    all_true, all_pred, all_doy, all_ndvi, all_mask, all_ids = [], [], [], [], [], []
    
    for r in test_records:
        t = len(r.ndvi_smoothed)
        sos, pos, eos = fit_double_logistic(r.ndvi_smoothed, r.doy)
        
        # Mock y_pred to force unified decode to find these exact dates
        y_pred = np.zeros(t, dtype=np.int64)
        
        if sos is not None and pos is not None and eos is not None and sos < pos < eos:
            # Find indices
            doy_list = r.doy.tolist()
            # Nearest idx for sos
            idx_sos = np.argmin(np.abs(r.doy - sos))
            idx_eos = np.argmin(np.abs(r.doy - eos))
            
            if idx_sos < idx_eos:
                y_pred[idx_sos:idx_eos] = 1  # VEGETATIVE
                # to satisfy POS: make POS the max NDVI
                y_pred[idx_eos:] = 3  # SENESCENCE
            
        all_true.append(r.labels)
        all_pred.append(y_pred)
        all_doy.append(r.doy)
        all_ndvi.append(r.ndvi_smoothed)
        all_mask.append(np.ones(t, dtype=bool))
        all_ids.append(r.season_id)
        
    return all_true, all_pred, all_doy, all_ndvi, all_mask, all_ids


def main():
    import warnings
    warnings.filterwarnings("ignore", message="Method 'trf' fails to compute the Jacobian")
    
    train_records = load_records("train")
    test_records = load_records("test")
    
    train_df = build_feature_table(train_records)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]
    
    # Run RF
    print("Training RF...")
    rf = build_rf_baseline()
    rf.fit(X_train, y_train)
    y_true, y_pred, doy, ndvi, mask = predict_tabular(rf, test_records)
    np.savez(PREDS_DIR / "rf_0.npz",
             y_true=y_true, y_pred=y_pred, doy=doy, ndvi=ndvi, mask=mask,
             season_ids=np.array([r.season_id for r in test_records]))
    
    # Run XGBoost
    print("Training XGBoost...")
    xgb = build_xgboost_baseline()
    xgb.fit(X_train, y_train)
    y_true, y_pred, doy, ndvi, mask = predict_tabular(xgb, test_records)
    np.savez(PREDS_DIR / "xgboost_0.npz",
             y_true=y_true, y_pred=y_pred, doy=doy, ndvi=ndvi, mask=mask,
             season_ids=np.array([r.season_id for r in test_records]))
             
    # Run Double Logistic
    print("Running Double Logistic...")
    y_true, y_pred, doy, ndvi, mask, ids = run_double_logistic(test_records)
    
    # Pad for npz arrays
    max_len = max(len(r) for r in y_true)
    N = len(y_true)
    
    y_true_pad = np.full((N, max_len), -100, dtype=np.int64)
    y_pred_pad = np.zeros((N, max_len), dtype=np.int64)
    doy_pad = np.zeros((N, max_len), dtype=np.int64)
    ndvi_pad = np.zeros((N, max_len), dtype=np.float32)
    mask_pad = np.zeros((N, max_len), dtype=bool)
    
    for i in range(N):
        t = len(y_true[i])
        y_true_pad[i, :t] = y_true[i]
        y_pred_pad[i, :t] = y_pred[i]
        doy_pad[i, :t] = doy[i]
        ndvi_pad[i, :t] = ndvi[i]
        mask_pad[i, :t] = mask[i]
        
    np.savez(PREDS_DIR / "double_logistic_0.npz",
             y_true=y_true_pad, y_pred=y_pred_pad, doy=doy_pad, 
             ndvi=ndvi_pad, mask=mask_pad, season_ids=np.array(ids))

if __name__ == "__main__":
    main()
