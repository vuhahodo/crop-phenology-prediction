import argparse
import copy
import multiprocessing
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.train import load_records, _evaluate_macro_f1
from src.data.dataset import NDVISequenceDataset
from src.models.transformer import SinusoidalPositionalEncoding

MODELS_DIR = Path("results/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class ScalarGate(nn.Module):
    def __init__(self, init_val=0.5):
        super().__init__()
        self.val = nn.Parameter(torch.tensor([init_val]))
        
    def forward(self, x, doy_sin_cos):
        # Ignores DOY completely, just multiplies by scalar
        return x * torch.sigmoid(self.val)

class TransformerScalarGateLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.gate = ScalarGate()

    def forward(self, x: torch.Tensor, doy_sin_cos: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder_layer(x, src_key_padding_mask=padding_mask)
        return self.gate(h, doy_sin_cos)

class TransformerScalarGate(nn.Module):
    """Configuration: Transformer + Learned Scalar Gate (ignores DOY)."""
    DOY_CHANNELS = (1, 2)
    def __init__(self, in_channels: int, num_classes: int = 4, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            TransformerScalarGateLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        doy_sin_cos = x[:, :, self.DOY_CHANNELS[0]:self.DOY_CHANNELS[1] + 1]
        h = self.pos_enc(self.input_proj(x))
        for layer in self.layers:
            h = layer(h, doy_sin_cos, padding_mask=padding_mask)
        return self.classifier(h)

def train_scalar_gate_seed(seed):
    model_name = "transformer_scalar_gate"
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
    model = TransformerScalarGate(in_channels=in_channels).to(device)
        
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

if __name__ == "__main__":
    seeds = [42, 123, 2024, 7, 314]
    with multiprocessing.Pool(4) as pool:
        pool.map(train_scalar_gate_seed, seeds)
