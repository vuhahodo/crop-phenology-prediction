import argparse
import copy
import time
import numpy as np
import torch
import scipy.stats as stats
from torch.utils.data import DataLoader
from src.train import load_records, _evaluate_macro_f1
from src.data.dataset import NDVISequenceDataset
from src.evaluation.metrics import Timer, segmentation_metrics

def apply_cloud_dropout(x, mask, drop_ratio=0.2):
    """Simulates contiguous cloud cover by zeroing out a block of the sequence."""
    x_dropped = x.clone()
    for i in range(x.shape[0]):
        t = mask[i].sum().item()
        if t == 0: continue
        drop_len = int(t * drop_ratio)
        if drop_len == 0: continue
        start = np.random.randint(0, t - drop_len + 1)
        x_dropped[i, start:start+drop_len, :] = 0.0
    return x_dropped

def evaluate_with_dropout(model, loader, device, drop_ratio):
    model.eval()
    all_true, all_pred, all_mask = [], [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch["x"], batch["y"]
            mask = batch["mask"]
            if drop_ratio > 0:
                x = apply_cloud_dropout(x, mask, drop_ratio)
            logits = model(x.to(device))
            all_true.append(y.numpy())
            all_pred.append(logits.argmax(dim=-1).cpu().numpy())
            all_mask.append(mask.numpy())
            
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    mask = np.concatenate(all_mask)
    
    metrics = segmentation_metrics(y_true, y_pred, mask)
    return metrics["macro_f1"]

def train_and_eval_seed(model_name: str, device: str, seed: int, drop_ratios: list[float]):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_records = load_records("train")
    val_records = load_records("val")
    test_records = load_records("test")
    
    max_len = max(len(r.ndvi_smoothed) for r in train_records + val_records + test_records)
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
                
    model.load_state_dict(best_state)
    
    results = {}
    for r in drop_ratios:
        # Evaluate 5 times and average to reduce random dropout noise on the same model
        f1_runs = []
        for _ in range(3): 
            f1_runs.append(evaluate_with_dropout(model, test_loader, device, r))
        results[r] = np.mean(f1_runs)
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    models = ["transformer", "transformer_cyclical_doy", "cba_phenonet"]
    seeds = [42, 123, 2024, 7, 314]
    drop_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    all_results = {m: {r: [] for r in drop_ratios} for m in models}
    
    for m in models:
        print(f"\nEvaluating model: {m}")
        for s in seeds:
            res = train_and_eval_seed(m, args.device, s, drop_ratios)
            print(f"  Seed {s} finished.")
            for r in drop_ratios:
                all_results[m][r].append(res[r])
                
    print("\n\n=== CLOUD DROPOUT MULTI-SEED RESULTS ===")
    print(f"{'Dropout':<8} | {'Standard Transformer':<22} | {'Cyclical DOY Only':<22} | {'Full CBA-PhenoNet':<22}")
    print("-" * 80)
    for r in drop_ratios:
        std_mean = np.mean(all_results["transformer"][r])
        std_std  = np.std(all_results["transformer"][r])
        doy_mean = np.mean(all_results["transformer_cyclical_doy"][r])
        doy_std  = np.std(all_results["transformer_cyclical_doy"][r])
        cba_mean = np.mean(all_results["cba_phenonet"][r])
        cba_std  = np.std(all_results["cba_phenonet"][r])
        
        print(f"{r*100:2.0f}%      | {std_mean:.4f} +/- {std_std:.4f}     | {doy_mean:.4f} +/- {doy_std:.4f}     | {cba_mean:.4f} +/- {cba_std:.4f}")
        
    print("\n\n=== PAIRED T-TEST: Full CBA vs Cyclical DOY ===")
    alpha = 0.05
    n_tests = len(drop_ratios)
    bonf_alpha = alpha / n_tests
    
    for r in drop_ratios:
        cba_vals = all_results["cba_phenonet"][r]
        doy_vals = all_results["transformer_cyclical_doy"][r]
        t_stat, p_val = stats.ttest_rel(cba_vals, doy_vals)
        padj = min(p_val * n_tests, 1.0)
        mean_diff = np.mean(cba_vals) - np.mean(doy_vals)
        sig = "*" if padj < alpha else "ns"
        print(f"Dropout {r*100:2.0f}%: CBA - DOY = {mean_diff:+.4f} | p-value = {p_val:.4f} | p_adj = {padj:.4f} ({sig})")

if __name__ == "__main__":
    main()
