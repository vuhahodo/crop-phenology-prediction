import argparse
import copy
import multiprocessing
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from src.train import load_records, _evaluate_macro_f1
from src.data.dataset import NDVISequenceDataset

MODELS_DIR = Path("results/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_model_seed(args):
    model_name, seed = args
    save_path = MODELS_DIR / f"{model_name}_{seed}.pt"
    if save_path.exists():
        print(f"Skipping {model_name} {seed}, already exists.")
        return
        
    device = "cpu"
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)
    
    train_records = load_records("train")
    val_records = load_records("val")
    test_records = load_records("test")
    
    max_len = max(len(r.ndvi_smoothed) for r in train_records + val_records + test_records)
    train_ds = NDVISequenceDataset(train_records, max_len=max_len)
    val_ds = NDVISequenceDataset(val_records, max_len=max_len)
    
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
    patience = 15
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
                
    torch.save(best_state, save_path)
    print(f"Finished {model_name} {seed} with val_f1={best_val_f1:.4f}")

def main():
    models = ["transformer", "transformer_cyclical_doy", "transformer_calendar_gate", "cba_phenonet"]
    seeds = [42, 123, 2024, 7, 314]
    
    tasks = [(m, s) for m in models for s in seeds]
    
    # 4 workers should max out a quad-core / 8-thread CPU well without crashing memory
    with multiprocessing.Pool(4) as pool:
        pool.map(train_model_seed, tasks)

if __name__ == "__main__":
    main()
