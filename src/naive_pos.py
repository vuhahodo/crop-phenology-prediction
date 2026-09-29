import torch
import numpy as np
from src.train import load_records, NDVISequenceDataset

def compute_metrics(true_days, pred_days):
    valid = ~np.isnan(true_days)
    if not np.any(valid):
        return 0.0, 0.0, 0.0, 0.0, 0
    t = true_days[valid]
    p = pred_days[valid]
    
    diff = np.abs(t - p)
    # circular diff
    diff = np.minimum(diff, 365.0 - diff)
    
    mae = np.mean(diff)
    rmse = np.sqrt(np.mean(diff**2))
    medae = np.median(diff)
    acc3 = np.mean(diff <= 3) * 100
    
    return mae, rmse, medae, acc3, np.sum(valid)

def main():
    test_records = load_records("test")
    max_len = max(len(r.ndvi_smoothed) for r in test_records)
    test_ds = NDVISequenceDataset(test_records, max_len=max_len)

    true_pos = []
    pred_pos = []
    
    for idx in range(len(test_ds)):
        item = test_ds[idx]
        ndvi = item["ndvi"].numpy()
        labels = item["y"].numpy()
        mask = item["mask"].numpy()
        
        valid_stages = (labels == 1) | (labels == 2)
        valid_stages = valid_stages & mask.astype(bool)
        
        if np.any(valid_stages):
            # Target logic from b0d_milestones / dataset: argmax in Veg/Rep
            masked_ndvi = np.where(valid_stages, ndvi, -1)
            t_pos_idx = np.argmax(masked_ndvi)
            t_pos_day = t_pos_idx * 5 + 1
            
            # Naive POS: argmax of NDVI across the ENTIRE sequence
            p_pos_idx = np.argmax(np.where(mask.astype(bool), ndvi, -1))
            p_pos_day = p_pos_idx * 5 + 1
            
            true_pos.append(t_pos_day)
            pred_pos.append(p_pos_day)
        else:
            true_pos.append(np.nan)
            pred_pos.append(np.nan)

    true_pos = np.array(true_pos)
    pred_pos = np.array(pred_pos)
    
    mae, rmse, medae, acc, n = compute_metrics(true_pos, pred_pos)
    print(f"Naive POS Baseline (N={n}):")
    print(f"MAE: {mae:.2f} days")
    print(f"RMSE: {rmse:.2f} days")
    print(f"MedAE: {medae:.2f} days")
    print(f"Acc+-3: {acc:.2f}%")

if __name__ == "__main__":
    main()
