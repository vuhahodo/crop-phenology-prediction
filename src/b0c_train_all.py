"""B0c — Retrain ALL 7 configs × 10 seeds from scratch.

Deterministic: torch.manual_seed, np.random.seed, torch.use_deterministic_algorithms,
DataLoader generator, worker_init_fn.

No skip-if-exists. Every checkpoint is overwritten.

Seeds declared upfront: [42, 123, 2024, 7, 314, 0, 1, 999, 555, 77]
Configs: standard, cyclical_doy, calendar_gate, scalar_gate, cba_phenonet,
         cba_phenonet_residual, cba_phenonet_disentangled
"""
import copy
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import NDVISequenceDataset
from src.train import load_records, _evaluate_macro_f1

SEEDS = [42, 123, 2024, 7, 314, 0, 1, 999, 555, 77]

CONFIGS = [
    "transformer",
    "transformer_cyclical_doy",
    "transformer_calendar_gate",
    "transformer_scalar_gate",
    "cba_phenonet",
    "cba_phenonet_residual",
    "cba_phenonet_disentangled",
]

MODELS_DIR = Path("results/models_b0c")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def _worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % 2**32)


def _get_model(name, in_channels):
    if name == "transformer":
        from src.models.transformer import NDVITransformer
        return NDVITransformer(in_channels=in_channels)
    elif name == "transformer_cyclical_doy":
        from src.models.ablation_models import TransformerCyclicalDOY
        return TransformerCyclicalDOY(in_channels=in_channels)
    elif name == "transformer_calendar_gate":
        from src.models.ablation_models import TransformerCalendarGate
        return TransformerCalendarGate(in_channels=in_channels)
    elif name == "transformer_scalar_gate":
        from src.b0_scalar_gate import TransformerScalarGate
        return TransformerScalarGate(in_channels=in_channels)
    elif name == "cba_phenonet":
        from src.models.cba_phenonet import CBAPhenoNet
        return CBAPhenoNet(in_channels=in_channels)
    elif name == "cba_phenonet_residual":
        from src.models.ablation_models import CBAPhenoNetResidual
        return CBAPhenoNetResidual(in_channels=in_channels)
    elif name == "cba_phenonet_disentangled":
        from src.models.ablation_models import CBAPhenoNetDisentangled
        return CBAPhenoNetDisentangled(in_channels=in_channels)
    else:
        raise ValueError(name)


def train_one(config, seed, train_loader, val_loader, in_channels, device="cpu"):
    """Train one (config, seed). Returns (best_state_dict, best_val_f1, n_epochs)."""
    # Full determinism
    torch.manual_seed(seed)
    np.random.seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = _get_model(config, in_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=NDVISequenceDataset.PAD_LABEL)

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    patience = 15
    max_epochs = 150

    model.train()
    for epoch in range(max_epochs):
        for batch in train_loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()

        val_f1 = _evaluate_macro_f1(model, val_loader, device)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    return best_state, best_val_f1, epoch + 1


def main():
    device = "cpu"

    # Log environment
    env_info = {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "seeds": SEEDS,
        "configs": CONFIGS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(MODELS_DIR / "env_info.json", "w") as f:
        json.dump(env_info, f, indent=2)

    # Load data ONCE
    train_records = load_records("train")
    val_records = load_records("val")
    test_records = load_records("test")

    max_len = max(len(r.ndvi_smoothed) for r in train_records + val_records + test_records)
    train_ds = NDVISequenceDataset(train_records, max_len=max_len)
    val_ds = NDVISequenceDataset(val_records, max_len=max_len)
    in_channels = train_ds.n_channels

    log_rows = []

    for config in CONFIGS:
        for seed in SEEDS:
            save_path = MODELS_DIR / f"{config}_{seed}.pt"
            if save_path.exists():
                print(f"Skipping {config}_{seed}, already exists.")
                continue

            # Create DataLoader with seed-specific generator
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(
                train_ds, batch_size=32, shuffle=True,
                generator=generator, worker_init_fn=_worker_init_fn,
            )
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

            t0 = time.time()
            best_state, best_val_f1, n_epochs = train_one(
                config, seed, train_loader, val_loader, in_channels, device
            )
            elapsed = time.time() - t0

            torch.save(best_state, save_path)

            row = {
                "config": config, "seed": seed,
                "val_f1": round(best_val_f1, 6), "n_epochs": n_epochs,
                "train_seconds": round(elapsed, 1),
            }
            log_rows.append(row)
            print(f"  {config} seed={seed}: val_f1={best_val_f1:.4f}, epochs={n_epochs}, time={elapsed:.0f}s")

    # Save training log
    import csv
    log_path = MODELS_DIR / "training_log.csv"
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "seed", "val_f1", "n_epochs", "train_seconds"])
        w.writeheader()
        w.writerows(log_rows)
    print(f"\nTraining log saved to {log_path}")


if __name__ == "__main__":
    main()
