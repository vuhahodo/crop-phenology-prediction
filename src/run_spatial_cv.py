import argparse
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.train import _evaluate_macro_f1
from src.data.dataset import NDVISequenceDataset
from src.data.breizhcrops_loader import load_breizhcrops_records
from src.evaluation.metrics import Timer, build_eval_result

def run_spatial_cv(model_name: str, device: str = "cpu", seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"\n=======================================================")
    print(f"RUNNING SPATIAL CV FOR: {model_name} [seed={seed}]")
    print(f"=======================================================")
    
    # 1. Load source data
    records = load_breizhcrops_records(region="frh01", max_per_class=1000)
    
    # 2. Extract spatial groups from parcel IDs
    # season_id is like "frh01_6017310_wheat" -> fid is "6017310"
    for r in records:
        fid_str = r.season_id.split("_")[1]
        # Use first 3 digits as a commune/canton proxy (geographic cluster)
        r.spatial_group = fid_str[:3]
        
    groups = np.array([r.spatial_group for r in records])
    unique_groups = np.unique(groups)
    print(f"Found {len(unique_groups)} spatial clusters based on ID prefix.")
    
    # 3. Create 3 folds of groups
    np.random.shuffle(unique_groups)
    folds = np.array_split(unique_groups, 3)
    
    max_len = max(len(r.ndvi_smoothed) for r in records)
    
    fold_f1s = []
    
    for fold_idx, test_groups in enumerate(folds):
        print(f"\n--- Fold {fold_idx + 1}/3 ---")
        train_records = [r for r in records if r.spatial_group not in test_groups]
        test_records = [r for r in records if r.spatial_group in test_groups]
        
        # Split train into train/val
        val_size = int(len(train_records) * 0.15)
        val_records = train_records[:val_size]
        train_records = train_records[val_size:]
        
        train_ds = NDVISequenceDataset(train_records, max_len=max_len)
        val_ds = NDVISequenceDataset(val_records, max_len=max_len)
        test_ds = NDVISequenceDataset(test_records, max_len=max_len)
        
        generator = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=generator)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
        
        in_channels = train_ds.n_channels
        if model_name == "cba_phenonet":
            from src.models.cba_phenonet import CBAPhenoNet
            model = CBAPhenoNet(in_channels=in_channels).to(device)
        elif model_name == "transformer_cyclical_doy":
            from src.models.ablation_models import TransformerCyclicalDOY
            model = TransformerCyclicalDOY(in_channels=in_channels).to(device)
        elif model_name == "transformer_calendar_gate":
            from src.models.ablation_models import TransformerCalendarGate
            model = TransformerCalendarGate(in_channels=in_channels).to(device)
        elif model_name == "transformer":
            from src.models.transformer import NDVITransformer
            model = NDVITransformer(in_channels=in_channels).to(device)
        else:
            raise ValueError(model_name)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=NDVISequenceDataset.PAD_LABEL)
        
        best_val_f1 = -1.0
        best_state = None
        epochs_without_improvement = 0
        
        print(f"Training on {len(train_records)} parcels, val on {len(val_records)}, testing on {len(test_records)}...")
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
                if epochs_without_improvement >= 15:
                    break
                    
        model.load_state_dict(best_state)
        
        # Evaluate
        model.eval()
        all_true, all_pred, all_mask = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                x, y, mask = batch["x"], batch["y"], batch["mask"]
                logits = model(x.to(device))
                all_true.append(y.numpy())
                all_pred.append(logits.argmax(dim=-1).cpu().numpy())
                all_mask.append(mask.numpy())
                
        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        mask_cat = np.concatenate(all_mask)
        
        from src.evaluation.metrics import segmentation_metrics
        metrics = segmentation_metrics(y_true, y_pred, mask_cat)
        test_f1 = metrics["macro_f1"]
        print(f"Fold {fold_idx + 1} Test Macro-F1: {test_f1:.4f}")
        fold_f1s.append(test_f1)
        
    mean_f1 = np.mean(fold_f1s)
    std_f1 = np.std(fold_f1s)
    print(f"\n---> 3-Fold Spatial CV Final Result for {model_name}: {mean_f1:.4f} +/- {std_f1:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["transformer", "transformer_cyclical_doy", "transformer_calendar_gate", "cba_phenonet", "all"], default="all")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    models = ["transformer", "transformer_cyclical_doy", "transformer_calendar_gate", "cba_phenonet"] if args.model == "all" else [args.model]
    for m in models:
        run_spatial_cv(m, args.device, seed=42)

if __name__ == "__main__":
    main()
