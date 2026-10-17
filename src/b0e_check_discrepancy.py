import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from src.train import load_records
from src.data.dataset import NDVISequenceDataset
from src.models.cba_phenonet import CBAPhenoNet

OLD_MODELS_DIR = Path("results/models")
NEW_MODELS_DIR = Path("results/models_b0c")
SEEDS = [42, 123, 2024, 7, 314]

def eval_checkpoint(ckpt_path, test_loader, device="cuda"):
    model = CBAPhenoNet(in_channels=3, num_classes=4).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    all_true, all_pred, all_mask = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y, mask = batch["x"].to(device), batch["y"].to(device), batch["mask"].to(device)
            out = model(x, padding_mask=~mask)
            preds = out.argmax(dim=-1)
            
            all_true.append(y.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
            all_mask.append(mask.cpu().numpy())
            
    yt = np.concatenate(all_true)
    yp = np.concatenate(all_pred)
    m = np.concatenate(all_mask)
    return f1_score(yt[m], yp[m], average="macro", zero_division=0)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    test_records = load_records("test")
    test_ds = NDVISequenceDataset(test_records, max_len=512)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    old_f1s = []
    new_f1s = []
    
    for seed in SEEDS:
        old_path = OLD_MODELS_DIR / f"cba_phenonet_{seed}.pt"
        new_path = NEW_MODELS_DIR / f"cba_phenonet_{seed}.pt"
        
        f1_old = eval_checkpoint(old_path, test_loader, device) if old_path.exists() else np.nan
        f1_new = eval_checkpoint(new_path, test_loader, device) if new_path.exists() else np.nan
        
        old_f1s.append(f1_old)
        new_f1s.append(f1_new)
        print(f"Seed {seed}: Old F1 = {f1_old:.4f}, New F1 = {f1_new:.4f}")
        
    print(f"\nOld 5-seed Mean: {np.nanmean(old_f1s):.4f} +/- {np.nanstd(old_f1s, ddof=1):.4f}")
    print(f"New 5-seed Mean: {np.nanmean(new_f1s):.4f} +/- {np.nanstd(new_f1s, ddof=1):.4f}")
    
if __name__ == "__main__":
    main()
