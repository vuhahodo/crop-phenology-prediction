import json
import numpy as np
import torch
import scipy.stats as stats
from src.eval_phase_a import get_model
from src.data.dataset import NDVISequenceDataset
from src.train import load_records
from torch.utils.data import DataLoader

def extract_milestones(y):
    # y shape: (N,)
    sos, pos, eos = -1, -1, -1
    for t in range(len(y) - 1):
        if y[t] == 0 and y[t+1] == 1:
            sos = t + 1
        elif y[t] == 1 and y[t+1] == 2:
            pos = t + 1
        elif y[t] == 2 and y[t+1] == 3:
            eos = t + 1
    return sos, pos, eos

def get_milestones(loader, model, device):
    model.eval()
    all_true = []
    all_pred = []
    all_doy = []
    with torch.no_grad():
        for batch in loader:
            x, y, doy = batch["x"], batch["y"], batch["doy"]
            logits = model(x.to(device))
            preds = logits.argmax(dim=-1).cpu().numpy()
            y = y.numpy()
            doy = doy.numpy()
            for i in range(x.shape[0]):
                valid_len = (y[i] != NDVISequenceDataset.PAD_LABEL).sum()
                if valid_len == 0: continue
                all_true.append((y[i, :valid_len], doy[i, :valid_len]))
                all_pred.append((preds[i, :valid_len], doy[i, :valid_len]))
    
    true_m = []
    pred_m = []
    for (ty, td), (py, pd) in zip(all_true, all_pred):
        tsos, tpos, teos = extract_milestones(ty)
        psos, ppos, peos = extract_milestones(py)
        # Convert indices to DOY
        t_doy = (td[tsos] if tsos != -1 else -1, td[tpos] if tpos != -1 else -1, td[teos] if teos != -1 else -1)
        p_doy = (pd[psos] if psos != -1 else -1, pd[ppos] if ppos != -1 else -1, pd[peos] if peos != -1 else -1)
        true_m.append(t_doy)
        pred_m.append(p_doy)
    return true_m, pred_m

def main():
    device = "cpu"
    models = ["transformer", "transformer_cyclical_doy", "transformer_calendar_gate", "cba_phenonet"]
    seeds = [42, 123, 2024, 7, 314]
    
    test_records = load_records("test")
    max_len = max(len(r.ndvi_smoothed) for r in test_records)
    test_ds = NDVISequenceDataset(test_records, max_len=max_len)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    in_channels = test_ds.n_channels
    
    # Store results: results[seed][model] = (true_m, pred_m)
    results = {s: {} for s in seeds}
    
    for s in seeds:
        for m in models:
            model = get_model(m, in_channels, device)
            pt_path = f"results/models/{m}_{s}.pt"
            model.load_state_dict(torch.load(pt_path, map_location=device))
            true_m, pred_m = get_milestones(test_loader, model, device)
            results[s][m] = (true_m, pred_m)
            
    # Compute recall and MAE
    report = "# Clean-data Milestone Level Testing\n\n"
    
    for s in seeds:
        report += f"## Seed {s}\n"
        true_m = results[s]["transformer"][0] # True is same for all models
        N = len(true_m)
        
        # Which samples have valid true milestones?
        true_sos_idx = [i for i, m in enumerate(true_m) if m[0] != -1]
        true_pos_idx = [i for i, m in enumerate(true_m) if m[1] != -1]
        true_eos_idx = [i for i, m in enumerate(true_m) if m[2] != -1]
        
        # Which samples are detected by ALL 4 models?
        intersect_sos = set(true_sos_idx)
        intersect_pos = set(true_pos_idx)
        intersect_eos = set(true_eos_idx)
        
        for m in models:
            pm = results[s][m][1]
            intersect_sos &= {i for i in true_sos_idx if pm[i][0] != -1}
            intersect_pos &= {i for i in true_pos_idx if pm[i][1] != -1}
            intersect_eos &= {i for i in true_eos_idx if pm[i][2] != -1}
            
        report += f"Valid True: SOS={len(true_sos_idx)}, POS={len(true_pos_idx)}, EOS={len(true_eos_idx)}\n"
        report += f"Intersection (detected by all): SOS={len(intersect_sos)}, POS={len(intersect_pos)}, EOS={len(intersect_eos)}\n\n"
        
        report += "| Model | Recall SOS | Recall POS | Recall EOS | MAE SOS (Self) | MAE POS (Self) | MAE EOS (Self) | MAE SOS (Int) | MAE POS (Int) | MAE EOS (Int) |\n"
        report += "|---|---|---|---|---|---|---|---|---|---|\n"
        
        for m in models:
            pm = results[s][m][1]
            
            # Recall
            rec_sos = sum(1 for i in true_sos_idx if pm[i][0] != -1) / len(true_sos_idx) if len(true_sos_idx) else 0
            rec_pos = sum(1 for i in true_pos_idx if pm[i][1] != -1) / len(true_pos_idx) if len(true_pos_idx) else 0
            rec_eos = sum(1 for i in true_eos_idx if pm[i][2] != -1) / len(true_eos_idx) if len(true_eos_idx) else 0
            
            # MAE Self
            mae_sos_self = np.mean([abs(true_m[i][0] - pm[i][0]) for i in true_sos_idx if pm[i][0] != -1])
            mae_pos_self = np.mean([abs(true_m[i][1] - pm[i][1]) for i in true_pos_idx if pm[i][1] != -1])
            mae_eos_self = np.mean([abs(true_m[i][2] - pm[i][2]) for i in true_eos_idx if pm[i][2] != -1])
            
            # MAE Intersect
            mae_sos_int = np.mean([abs(true_m[i][0] - pm[i][0]) for i in intersect_sos]) if intersect_sos else 0
            mae_pos_int = np.mean([abs(true_m[i][1] - pm[i][1]) for i in intersect_pos]) if intersect_pos else 0
            mae_eos_int = np.mean([abs(true_m[i][2] - pm[i][2]) for i in intersect_eos]) if intersect_eos else 0
            
            report += f"| {m} | {rec_sos:.2%} | {rec_pos:.2%} | {rec_eos:.2%} | {mae_sos_self:.2f} | {mae_pos_self:.2f} | {mae_eos_self:.2f} | {mae_sos_int:.2f} | {mae_pos_int:.2f} | {mae_eos_int:.2f} |\n"
            
    with open("results/b0_milestones.md", "w") as f:
        f.write(report)
        
if __name__ == "__main__":
    main()
