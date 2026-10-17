import numpy as np
import torch
import optuna
from pathlib import Path
from scipy.optimize import curve_fit
import warnings
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from xgboost import XGBClassifier

from src.train import load_records, train_dl_model
from src.data.dataset import cyclical_doy, NDVISequenceDataset
from src.models.tabular_full_season import build_full_season_feature_table, FULL_FEATURE_COLS
from src.data.features import build_feature_table

optuna.logging.set_verbosity(optuna.logging.WARNING)
PREDS_DIR = Path("results/preds")
PREDS_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 123, 2024, 7, 314, 0, 1, 999, 555, 77]


# ============================================================
# Part 1: Tabular ML Causal (Tuned)
# ============================================================
def get_tabular_data_causal(records):
    import pandas as pd
    df = build_feature_table(records)
    # Causal features: backward lags only
    features = ["lag_1", "lag_2", "lag_3", "doy_sin", "doy_cos", "ndvi"]
    X = df[features].values
    y = df["label"].values
    return X, y, df, features

def run_tuned_tabular_causal(model_type, train_records, val_records, test_records, n_trials=50):
    print(f"\nTuning {model_type} causal (Budget: {n_trials} trials)...")
    X_train, y_train, _, _ = get_tabular_data_causal(train_records)
    X_val, y_val, _, _ = get_tabular_data_causal(val_records)
    
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    test_fold = np.concatenate([-1 * np.ones(len(y_train)), np.zeros(len(y_val))])
    ps = PredefinedSplit(test_fold)
    
    if model_type == "rf":
        base = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")
        space = {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [5, 10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }
    else:
        base = XGBClassifier(random_state=42, n_jobs=-1, eval_metric="mlogloss")
        space = {
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [3, 5, 7, 9],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
        
    search = RandomizedSearchCV(base, space, n_iter=n_trials, scoring="f1_macro", cv=ps, random_state=42, n_jobs=1)
    search.fit(X_all, y_all)
    print(f"  Best val F1: {search.best_score_:.4f}")
    print(f"  Best params: {search.best_params_}")
    
    best_model = search.best_estimator_
    
    # Predict on Test
    _, _, test_df, features = get_tabular_data_causal(test_records)
    test_df["pred"] = best_model.predict(test_df[features].values)
    
    max_len = max(len(r.ndvi_smoothed) for r in test_records)
    N = len(test_records)
    y_true = np.full((N, max_len), -100, dtype=np.int64)
    y_pred = np.full((N, max_len), -100, dtype=np.int64)
    doy = np.zeros((N, max_len), dtype=np.int64)
    ndvi = np.zeros((N, max_len), dtype=np.float32)
    mask = np.zeros((N, max_len), dtype=bool)

    for i, rec in enumerate(test_records):
        sub = test_df[test_df["season_id"] == rec.season_id]
        L = len(sub)
        y_true[i, :L] = sub["label"].values
        y_pred[i, :L] = sub["pred"].values
        doy[i, :L] = rec.doy
        ndvi[i, :L] = rec.ndvi_smoothed
        mask[i, :L] = True

    return y_true, y_pred, doy, ndvi, mask


# ============================================================
# Part 2: DL Baselines Tuning & Training
# ============================================================
def tune_dl_model(model_name, n_trials=20):
    print(f"\nTuning {model_name} (Budget: {n_trials} trials)...")
    from src.train import load_records, NDVISequenceDataset
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score
    import copy

    train_records = load_records("train")
    val_records = load_records("val")
    max_len = max(len(r.ndvi_smoothed) for r in train_records + val_records)
    train_ds_loc = NDVISequenceDataset(train_records, max_len=max_len)
    val_ds_loc = NDVISequenceDataset(val_records, max_len=max_len)
    
    in_channels = train_ds_loc.n_channels

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        # We need a new dataloader generator per trial for fairness
        generator = torch.Generator().manual_seed(42)
        train_loader_loc = DataLoader(train_ds_loc, batch_size=32, shuffle=True, generator=generator)
        val_loader_loc = DataLoader(val_ds_loc, batch_size=32, shuffle=False)

        torch.manual_seed(42)
        if model_name == "cnn1d":
            from src.models.cnn1d import NDVICNN1D
            model = NDVICNN1D(in_channels=in_channels, channels=(64, 64, 80)).to(DEVICE)
        elif model_name == "bilstm":
            from src.models.bilstm import NDVIBiLSTM
            model = NDVIBiLSTM(in_channels=in_channels, hidden_size=90, cell_type="lstm").to(DEVICE)
        elif model_name == "gru":
            from src.models.bilstm import NDVIBiLSTM
            model = NDVIBiLSTM(in_channels=in_channels, hidden_size=104, cell_type="gru").to(DEVICE)
        elif model_name == "tcn":
            from src.models.tcn import NDVITCN
            model = NDVITCN(in_channels=in_channels, num_channels=(64, 64, 80)).to(DEVICE)
        elif model_name == "transformer_causal":
            from src.models.transformer import NDVITransformer
            model = NDVITransformer(in_channels=in_channels, is_causal=True).to(DEVICE)
        else:
            raise ValueError(model_name)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=NDVISequenceDataset.PAD_LABEL)
        
        best_val_f1 = -1.0
        epochs_no_improve = 0
        for epoch in range(50):
            model.train()
            for batch in train_loader_loc:
                x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss.backward()
                optimizer.step()
                
            model.eval()
            all_true, all_pred, all_mask = [], [], []
            with torch.no_grad():
                for batch in val_loader_loc:
                    x, y, mask = batch["x"].to(DEVICE), batch["y"].to(DEVICE), batch["mask"].to(DEVICE)
                    logits = model(x)
                    preds = logits.argmax(dim=-1)
                    all_true.append(y.cpu().numpy())
                    all_pred.append(preds.cpu().numpy())
                    all_mask.append(mask.cpu().numpy())
            
            yt = np.concatenate(all_true)
            yp = np.concatenate(all_pred)
            m = np.concatenate(all_mask)
            val_f1 = f1_score(yt[m], yp[m], average="macro", zero_division=0)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= 5:
                    break
        
        return best_val_f1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"  Best val F1: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params

def train_dl_10_seeds(model_name, best_params):
    print(f"\nTraining {model_name} on 10 seeds with params {best_params}...")
    import sys
    from src.train import load_records, NDVISequenceDataset
    from torch.utils.data import DataLoader
    test_records = load_records("test")
    max_len = max(len(r.ndvi_smoothed) for r in test_records)
    test_ds = NDVISequenceDataset(test_records, max_len=max_len)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # We must patch train_dl_model locally to return the trained model, or just use the same logic locally
    # It's better to copy the exact training loop here so we can save preds.
    for seed in SEEDS:
        out_file = PREDS_DIR / f"{model_name}_{seed}.npz"
        if out_file.exists():
            continue
            
        print(f"  Seed {seed}...")
        # Train model logic
        torch.manual_seed(seed)
        train_records = load_records("train")
        val_records = load_records("val")
        train_ds_loc = NDVISequenceDataset(train_records, max_len=max_len)
        val_ds_loc = NDVISequenceDataset(val_records, max_len=max_len)
        generator = torch.Generator().manual_seed(seed)
        train_loader_loc = DataLoader(train_ds_loc, batch_size=32, shuffle=True, generator=generator)
        val_loader_loc = DataLoader(val_ds_loc, batch_size=32, shuffle=False)
        
        in_channels = train_ds_loc.n_channels
        if model_name == "cnn1d":
            from src.models.cnn1d import NDVICNN1D
            model = NDVICNN1D(in_channels=in_channels, channels=(64, 64, 80)).to(DEVICE)
        elif model_name == "bilstm":
            from src.models.bilstm import NDVIBiLSTM
            model = NDVIBiLSTM(in_channels=in_channels, hidden_size=90, cell_type="lstm").to(DEVICE)
        elif model_name == "gru":
            from src.models.bilstm import NDVIBiLSTM
            model = NDVIBiLSTM(in_channels=in_channels, hidden_size=104, cell_type="gru").to(DEVICE)
        elif model_name == "tcn":
            from src.models.tcn import NDVITCN
            model = NDVITCN(in_channels=in_channels, num_channels=(64, 64, 80)).to(DEVICE)
        elif model_name == "transformer_causal":
            from src.models.transformer import NDVITransformer
            model = NDVITransformer(in_channels=in_channels, is_causal=True).to(DEVICE)
        else:
            raise ValueError(model_name)
            
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=NDVISequenceDataset.PAD_LABEL)
        
        best_val_f1 = -1.0
        best_state = None
        epochs_no_improve = 0
        from sklearn.metrics import f1_score
        for epoch in range(150):
            model.train()
            for batch in train_loader_loc:
                x, y = batch["x"].to(DEVICE), batch["y"].to(DEVICE)
                optimizer.zero_grad()
                logits = model(x)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss.backward()
                optimizer.step()
                
            model.eval()
            all_true, all_pred, all_mask = [], [], []
            with torch.no_grad():
                for batch in val_loader_loc:
                    x, y, mask = batch["x"].to(DEVICE), batch["y"].to(DEVICE), batch["mask"].to(DEVICE)
                    logits = model(x)
                    preds = logits.argmax(dim=-1)
                    all_true.append(y.cpu().numpy())
                    all_pred.append(preds.cpu().numpy())
                    all_mask.append(mask.cpu().numpy())
            
            yt = np.concatenate(all_true)
            yp = np.concatenate(all_pred)
            m = np.concatenate(all_mask)
            val_f1 = f1_score(yt[m], yp[m], average="macro", zero_division=0)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                import copy
                best_state = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= 15:
                    break
        
        # Predict on Test
        model.load_state_dict(best_state)
        model.eval()
        all_true, all_pred, all_mask, all_doy, all_ndvi, all_ids = [], [], [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                x, y, mask = batch["x"].to(DEVICE), batch["y"].to(DEVICE), batch["mask"].to(DEVICE)
                doy, ndvi = batch["doy"].numpy(), batch["ndvi"].numpy()
                season_ids = batch["season_id"]
                logits = model(x)
                preds = logits.argmax(dim=-1)
                all_true.append(y.cpu().numpy())
                all_pred.append(preds.cpu().numpy())
                all_mask.append(mask.cpu().numpy())
                all_doy.append(doy)
                all_ndvi.append(ndvi)
                all_ids.extend(season_ids)
        
        np.savez(out_file,
                 y_true=np.concatenate(all_true),
                 y_pred=np.concatenate(all_pred),
                 doy=np.concatenate(all_doy),
                 ndvi=np.concatenate(all_ndvi),
                 mask=np.concatenate(all_mask),
                 season_ids=np.array(all_ids))


# ============================================================
# Part 3: Old Tabular Full-Season & Double Logistic
# ============================================================
def run_tabular_full_season(model_type, train_records, test_records):
    """Train and predict with full-season features. Returns padded arrays."""
    train_df = build_full_season_feature_table(train_records)
    test_df = build_full_season_feature_table(test_records)

    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight="balanced",
            n_jobs=-1, random_state=42,
        )
    else:
        model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1,
        )

    model.fit(train_df[FULL_FEATURE_COLS], train_df["label"])
    test_df = test_df.copy()
    test_df["pred"] = model.predict(test_df[FULL_FEATURE_COLS])

    max_len = max(len(r.ndvi_smoothed) for r in test_records)
    N = len(test_records)
    y_true = np.full((N, max_len), -100, dtype=np.int64)
    y_pred = np.full((N, max_len), -100, dtype=np.int64)
    doy = np.zeros((N, max_len), dtype=np.int64)
    ndvi = np.zeros((N, max_len), dtype=np.float32)
    mask = np.zeros((N, max_len), dtype=bool)

    for i, rec in enumerate(test_records):
        sub = test_df[test_df["season_id"] == rec.season_id]
        L = len(sub)
        y_true[i, :L] = sub["label"].values
        y_pred[i, :L] = sub["pred"].values
        doy[i, :L] = rec.doy
        ndvi[i, :L] = rec.ndvi_smoothed
        mask[i, :L] = True

    return y_true, y_pred, doy, ndvi, mask

def double_logistic(t, base, amp, sos_mid, sos_rate, eos_mid, eos_rate):
    ascending = 1.0 / (1.0 + np.exp(-sos_rate * (t - sos_mid)))
    descending = 1.0 / (1.0 + np.exp(eos_rate * (t - eos_mid)))
    return base + amp * ascending * descending

def fit_one_season(ndvi_seq, doy_seq):
    t = doy_seq.astype(np.float64) / 365.0
    y = ndvi_seq.astype(np.float64)
    base = np.percentile(y, 10)
    amp = np.percentile(y, 90) - base
    if amp < 0.01:
        amp = 0.01

    p0 = [base, amp, 0.25, 20.0, 0.70, 20.0]
    bounds = (
        [-0.5, 0.001, 0.0, 1.0, 0.3, 1.0],
        [1.0,  2.0,   0.7, 80.0, 1.0, 80.0],
    )
    p0 = [np.clip(p, lo, hi) for p, lo, hi in zip(p0, bounds[0], bounds[1])]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(double_logistic, t, y, p0=p0, bounds=bounds, maxfev=5000)
        sos_doy = int(round(popt[2] * 365))
        eos_doy = int(round(popt[4] * 365))
        t_fine = np.linspace(0, 1, 365)
        y_fit = double_logistic(t_fine, *popt)
        pos_doy = int(np.argmax(y_fit)) + 1
        return np.clip(sos_doy, 1, 365), np.clip(pos_doy, 1, 365), np.clip(eos_doy, 1, 365)
    except (RuntimeError, ValueError):
        pos_doy = int(doy_seq[np.argmax(y)])
        return None, pos_doy, None

def run_double_logistic(test_records):
    max_len = max(len(r.ndvi_smoothed) for r in test_records)
    N = len(test_records)
    y_true = np.full((N, max_len), -100, dtype=np.int64)
    y_pred = np.zeros((N, max_len), dtype=np.int64)
    doy_arr = np.zeros((N, max_len), dtype=np.int64)
    ndvi_arr = np.zeros((N, max_len), dtype=np.float32)
    mask_arr = np.zeros((N, max_len), dtype=bool)
    ids = []

    for i, rec in enumerate(test_records):
        L = len(rec.ndvi_smoothed)
        y_true[i, :L] = rec.labels
        doy_arr[i, :L] = rec.doy
        ndvi_arr[i, :L] = rec.ndvi_smoothed
        mask_arr[i, :L] = True
        ids.append(rec.season_id)

        sos, pos, eos = fit_one_season(rec.ndvi_smoothed, rec.doy)
        labels = np.zeros(L, dtype=np.int64)
        if sos is not None and eos is not None and sos < eos:
            idx_sos = np.argmin(np.abs(rec.doy - sos))
            idx_eos = np.argmin(np.abs(rec.doy - eos))
            if pos is not None:
                idx_pos = np.argmin(np.abs(rec.doy - pos))
                idx_pos = max(idx_pos, idx_sos)
                idx_pos = min(idx_pos, idx_eos)
            else:
                idx_pos = (idx_sos + idx_eos) // 2
            if idx_sos > 0:
                labels[:idx_sos] = 0
            labels[idx_sos:idx_pos] = 1
            labels[idx_pos:idx_eos] = 2
            labels[idx_eos:] = 3
            if idx_eos < L - 1:
                labels[-1] = 0
        y_pred[i, :L] = labels

    return y_true, y_pred, doy_arr, ndvi_arr, mask_arr, np.array(ids)

def main():
    train_records = load_records("train")
    val_records = load_records("val")
    test_records = load_records("test")

    # 0. Old Tabular Full-Season & Double Logistic
    for m in ["rf", "xgboost"]:
        out_file = PREDS_DIR / f"{m}_fullseason_0.npz"
        if not out_file.exists():
            print(f"Training {m} fullseason...")
            yt, yp, d, n, mask_arr = run_tabular_full_season(m, train_records, test_records)
            np.savez(out_file, y_true=yt, y_pred=yp, doy=d, ndvi=n, mask=mask_arr,
                     season_ids=np.array([r.season_id for r in test_records]))

    if not (PREDS_DIR / "double_logistic_0.npz").exists():
        print("Running Double-Logistic...")
        yt, yp, d, n, mask_arr, ids = run_double_logistic(test_records)
        np.savez(PREDS_DIR / "double_logistic_0.npz", y_true=yt, y_pred=yp, doy=d, ndvi=n, mask=mask_arr, season_ids=ids)

    # 1. Tune & Run Tabular Causal models (RF, XGB)
    for m in ["rf", "xgboost"]:
        out_file = PREDS_DIR / f"{m}_tuned_0.npz"
        if not out_file.exists():
            yt, yp, d, n, mask_arr = run_tuned_tabular_causal(m, train_records, val_records, test_records, n_trials=50)
            np.savez(out_file, y_true=yt, y_pred=yp, doy=d, ndvi=n, mask=mask_arr,
                     season_ids=np.array([r.season_id for r in test_records]))

    # 2. Tune & Run DL Baselines
    # DL Models: CNN1D, BiLSTM, GRU, TCN, Transformer Causal
    dl_models = ["cnn1d", "bilstm", "gru", "tcn", "transformer_causal"]
    for m in dl_models:
        best_params = tune_dl_model(m, n_trials=20)
        train_dl_10_seeds(m, best_params)

    # Note: Full-Season RF/XGB and Double-Logistic are handled by older scripts or can be left intact
    print("\nDone tuning and training all baselines!")


if __name__ == "__main__":
    main()
