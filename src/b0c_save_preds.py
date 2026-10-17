"""B0c — Save per-sample predictions for every (config, seed) to results/preds/.

Each file: results/preds/{config}_{seed}.npz containing:
  - y_true: (N, T) int64 
  - y_pred: (N, T) int64
  - doy: (N, T) int64
  - ndvi: (N, T) float32
  - mask: (N, T) bool
  - season_ids: list[str]
  
Uses models from results/models_b0c/ (the deterministic retrain).
Falls back to results/models/ for configs not yet retrained.
"""
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset import NDVISequenceDataset
from src.train import load_records

PREDS_DIR = Path("results/preds")
PREDS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    "transformer",
    "transformer_cyclical_doy",
    "transformer_calendar_gate",
    "transformer_scalar_gate",
    "cba_phenonet",
    "cba_phenonet_residual",
    "cba_phenonet_disentangled",
]

SEEDS = [42, 123, 2024, 7, 314, 0, 1, 999, 555, 77]


def get_model(name, in_channels):
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


def save_predictions(config, seed, test_loader, in_channels, device="cpu"):
    model = get_model(config, in_channels).to(device)
    
    # Try b0c first, then old
    pt_path = Path(f"results/models_b0c/{config}_{seed}.pt")
    if not pt_path.exists():
        pt_path = Path(f"results/models/{config}_{seed}.pt")
    if not pt_path.exists():
        print(f"  SKIP {config} seed={seed}: no checkpoint found")
        return False
    
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()
    
    all_true, all_pred, all_doy, all_ndvi, all_mask, all_ids = [], [], [], [], [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["x"].to(device))
            all_pred.append(logits.argmax(dim=-1).cpu().numpy())
            all_true.append(batch["y"].numpy())
            all_doy.append(batch["doy"].numpy())
            all_ndvi.append(batch["ndvi"].numpy())
            all_mask.append(batch["mask"].numpy())
            all_ids.extend(batch["season_id"])
    
    out_path = PREDS_DIR / f"{config}_{seed}.npz"
    np.savez(out_path,
        y_true=np.concatenate(all_true),
        y_pred=np.concatenate(all_pred),
        doy=np.concatenate(all_doy),
        ndvi=np.concatenate(all_ndvi),
        mask=np.concatenate(all_mask),
        season_ids=np.array(all_ids),
    )
    return True


def main():
    device = "cpu"
    test_records = load_records("test")
    max_len = max(len(r.ndvi_smoothed) for r in test_records)
    test_ds = NDVISequenceDataset(test_records, max_len=max_len)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    in_channels = test_ds.n_channels
    
    for config in CONFIGS:
        for seed in SEEDS:
            ok = save_predictions(config, seed, test_loader, in_channels, device)
            if ok:
                print(f"  Saved {config}_{seed}.npz")


if __name__ == "__main__":
    main()
