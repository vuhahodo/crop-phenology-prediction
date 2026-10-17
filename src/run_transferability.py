import argparse
import copy
import csv
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data.dataset import NDVISequenceDataset, SeasonRecord
from src.data.breizhcrops_loader import load_breizhcrops_records
from src.evaluation.metrics import Timer, build_eval_result, segmentation_metrics

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "transferability_study.csv"

def _evaluate_macro_f1(model, loader, device: str) -> float:
    model.eval()
    all_true, all_pred, all_mask = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(device))
            all_true.append(batch["y"].numpy())
            all_pred.append(logits.argmax(dim=-1).cpu().numpy())
            all_mask.append(batch["mask"].numpy())
    model.train()
    y_true, y_pred, mask = np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_mask)
    return segmentation_metrics(y_true, y_pred, mask)["macro_f1"]

def evaluate_on_region(model, region: str, device: str):
    print(f"\nEvaluating on {region} (Zero-Shot) ...")
    records = load_breizhcrops_records(region=region, max_per_class=1000)
    max_len = max(len(r.ndvi_smoothed) for r in records)
    ds = NDVISequenceDataset(records, max_len=max_len)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    
    model.eval()
    all_true, all_pred, all_doy, all_ndvi, all_mask, all_season_ids = [], [], [], [], [], []
    inference_start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(device))
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_true.append(batch["y"].numpy())
            all_pred.append(preds)
            all_doy.append(batch["doy"].numpy())
            all_ndvi.append(batch["ndvi"].numpy())
            all_mask.append(batch["mask"].numpy())
            all_season_ids.extend(batch["season_id"])
            
    inference_ms = (time.perf_counter() - inference_start) * 1000 / len(ds)
    return (
        np.concatenate(all_true), np.concatenate(all_pred),
        np.concatenate(all_doy), np.concatenate(all_ndvi), np.concatenate(all_mask),
        inference_ms, all_season_ids
    )

def train_and_evaluate_transfer(model_name: str, device: str = "cpu", seed: int = 42):
    torch.manual_seed(seed)
    print(f"\n=======================================================")
    print(f"RUNNING TRANSFERABILITY FOR: {model_name} [seed={seed}]")
    print(f"=======================================================")
    
    # 1. Load source data (frh01)
    print("Loading source region (frh01) for training...")
    source_records = load_breizhcrops_records(region="frh01", max_per_class=1000)
    crop_labels = [r.season_id.rsplit("_", 1)[-1] for r in source_records]
    
    # Train/val split
    train, val = train_test_split(source_records, test_size=0.2, random_state=42, stratify=crop_labels)
    
    max_len = max(len(r.ndvi_smoothed) for r in train + val)
    train_ds = NDVISequenceDataset(train, max_len=max_len)
    val_ds = NDVISequenceDataset(val, max_len=max_len)
    
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    in_channels = train_ds.n_channels
    if model_name == "cba_phenonet":
        from src.models.cba_phenonet import CBAPhenoNet
        model = CBAPhenoNet(in_channels=in_channels).to(device)
    elif model_name == "transformer_cyclical_doy":
        from src.models.ablation_models import TransformerCyclicalDOY
        model = TransformerCyclicalDOY(in_channels=in_channels).to(device)
    else:
        raise ValueError(model_name)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=NDVISequenceDataset.PAD_LABEL)
    
    # Train
    print("Training on frh01...")
    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    patience = 15
    with Timer() as timer:
        model.train()
        for epoch in range(150):
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
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    break
    train_s = timer["elapsed"]
    print(f"Training finished in {train_s:.1f}s. Best Val F1: {best_val_f1:.4f}")
    
    model.load_state_dict(best_state)
    
    # Evaluate on all regions (including frh01 for baseline comparison, but we'll use val set to save time or reload all? Let's reload test set of frh01 for true comparison, actually we used 80/20 here, so let's just evaluate on frh02 and frh03 directly)
    for target_region in ["frh02", "frh03"]:
        y_true, y_pred, doy, ndvi, mask, infer_ms, season_ids = evaluate_on_region(model, target_region, device)
        
        result, _ = build_eval_result(
            model_name=model_name,
            member_name=f"{model_name} [seed={seed}] on {target_region}",
            y_true=y_true, y_pred=y_pred, doy=doy, ndvi=ndvi, mask=mask,
            train_seconds=train_s, inference_ms_per_sample=infer_ms,
            season_ids=season_ids,
        )
        
        row = result.to_row()
        row["train_region"] = "frh01"
        row["test_region"] = target_region
        row["seed"] = seed
        
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_exists = RESULTS_PATH.exists()
        with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            
        print(f"-> {target_region} Results: F1={row['macro_f1']:.4f}, SOS MAE={row['sos_mae_days']:.1f}d, POS MAE={row['pos_mae_days']:.1f}d, EOS MAE={row['eos_mae_days']:.1f}d")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transformer_cyclical_doy", "cba_phenonet", "all"], default="all")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    models = ["transformer_cyclical_doy", "cba_phenonet"] if args.model == "all" else [args.model]
    
    for m in models:
        # We can run across 3 seeds to be robust but save time, or 5 seeds. Let's do 3 seeds: 42, 123, 2024
        for seed in [42, 123, 2024]:
            train_and_evaluate_transfer(m, args.device, seed)

if __name__ == "__main__":
    main()
