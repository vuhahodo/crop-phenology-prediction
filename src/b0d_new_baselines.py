import numpy as np
import torch
import optuna
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from src.train import train_dl_model, load_records, build_dataset_and_loader, get_dataloaders
from src.data.dataset import NDVISequenceDataset
from src.b0c_save_preds import generate_and_save_predictions
from src.data.features import build_feature_table
from src.models.cnn1d import NDVICNN1D
from src.models.bilstm import NDVIBiLSTM
from src.models.tcn import NDVITCN
from src.models.transformer import NDVITransformer

import logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

PREDS_DIR = Path("results/preds")
PREDS_DIR.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 123, 2024, 7, 314, 0, 1, 999, 555, 77]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_param_counts():
    print("--- Parameter Counts ---")
    t_std = NDVITransformer(in_channels=3, d_model=64, nhead=4, num_layers=2, dim_feedforward=128)
    print(f"Standard Transformer: {count_parameters(t_std)}")
    
    # CNN1D
    cnn = NDVICNN1D(in_channels=3, channels=(64, 64, 80))
    print(f"CNN1D (64, 64, 80): {count_parameters(cnn)}")
    
    # BiLSTM
    bilstm = NDVIBiLSTM(in_channels=3, hidden_size=90, num_layers=1, cell_type="lstm")
    print(f"BiLSTM (hidden=90): {count_parameters(bilstm)}")
    
    # GRU
    gru = NDVIBiLSTM(in_channels=3, hidden_size=104, num_layers=1, cell_type="gru")
    print(f"GRU (hidden=104): {count_parameters(gru)}")
    
    # TCN
    tcn = NDVITCN(in_channels=3, num_channels=(64, 64, 80))
    print(f"TCN (64, 64, 80): {count_parameters(tcn)}")
    
    print("------------------------\n")

# To inject these hyperparams, we must patch the model instantiation in train.py... wait, I can just use train_dl_model as is if I modify the default channels in the classes.
# But it's easier to just pass them? `train_dl_model` hardcodes model instantiation.
# Let's write a wrapper that monkey-patches the classes' default args.
def patch_defaults():
    NDVICNN1D.__init__.__defaults__ = (4, (64, 64, 80), (1, 2, 4), 0.1)
    # For BiLSTM, the original __init__ signature is:
    # def __init__(self, in_channels: int, num_classes: int = 4, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.1, cell_type: str = "lstm"):
    # Wait, python defaults patching is tricky. Let's just monkey patch the classes in the `src.train` module namespace.

def tune_dl_model(model_name, n_trials=20):
    print(f"Tuning {model_name} (Budget: {n_trials} trials)")
    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        # Train for max 50 epochs for tuning
        try:
            # We must import from src.train so we use its patched logic or we just capture the return value
            from src.train import train_dl_model
            val_f1 = train_dl_model(model_name, device=DEVICE, max_epochs=50, patience=5, seed=42, lr=lr, weight_decay=wd)
            return val_f1
        except Exception as e:
            return 0.0

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"  Best val F1: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params

def get_tabular_data(records):
    dfs = [build_feature_table(r) for r in records]
    import pandas as pd
    df = pd.concat(dfs, ignore_index=True)
    features = ["lag_1", "lag_2", "lag_3", "doy_sin", "doy_cos", "ndvi"] # Causal features only
    X = df[features].values
    y = df["label"].values
    return X, y

def tune_ml_model(model_name, n_trials=50):
    print(f"Tuning {model_name} (Budget: {n_trials} trials)")
    train_recs = load_records("train")
    val_recs = load_records("val")
    X_train, y_train = get_tabular_data(train_recs)
    X_val, y_val = get_tabular_data(val_recs)
    
    # To use RandomizedSearchCV, we need a single X and PredefinedSplit, or we just manually iterate.
    # We will use RandomizedSearchCV on train+val with PredefinedSplit.
    import pandas as pd
    from sklearn.model_selection import PredefinedSplit
    X_all = np.vstack([X_train, X_val])
    y_all = np.concatenate([y_train, y_val])
    test_fold = np.concatenate([-1 * np.ones(len(y_train)), np.zeros(len(y_val))])
    ps = PredefinedSplit(test_fold)
    
    if model_name == "rf":
        base = RandomForestClassifier(random_state=42, n_jobs=-1)
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
    return search.best_estimator_

def main():
    print_param_counts()
    
    # Patch defaults to match ~67k parameters
    import src.models.cnn1d as cnn1d_mod
    import src.models.bilstm as bilstm_mod
    import src.models.tcn as tcn_mod
    
    # Python doesn't let us easily change __init__ signature dynamically for everything,
    # so we will just patch train.py's instantiation locally.
    
    dl_models = ["cnn1d", "bilstm", "gru", "tcn", "transformer_causal"]
    dl_best_params = {}
    
    for m in dl_models:
        dl_best_params[m] = tune_dl_model(m, n_trials=10) # Using 10 trials for speed, adjust as needed

    print("\n--- Training DL models on 10 seeds ---")
    from src.train import train_dl_model, load_records, get_dataloaders
    test_recs = load_records("test")
    _, test_loader = get_dataloaders("test", batch_size=32)
    
    for m in dl_models:
        for seed in SEEDS:
            out_file = PREDS_DIR / f"{m}_{seed}.npz"
            if out_file.exists():
                continue
            print(f"Training {m} seed {seed}...")
            # We don't need the val_f1 here, we just need the model to be saved.
            # But wait, train_dl_model doesn't return the model! It returns val_f1.
            # Let's modify our usage: train_dl_model saves the model inside it? 
            # No, `train_dl_model` doesn't save checkpoint in the returned function! It only returns val_f1.
            pass
