"""Member assigned to Time-Series Transformer / Attention: fill in / tune this
file. Same (batch, T, C) -> (batch, T, num_classes) contract as cnn1d.py /
bilstm.py -- this is the "SOTA DL" arm of the benchmark (see docs/methodology.md).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Standard Vaswani et al. (2017) fixed positional encoding. Note this encodes
    POSITION IN THE WINDOW (0..T-1), which is complementary to -- not a
    replacement for -- the sin/cos DAY-OF-YEAR channels already in the input
    (see dataset.py): position tells the model "how far into the season", DOY
    tells it "which calendar day", both matter for phenology timing."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class NDVITransformer(nn.Module):
    """
    Input:  x (batch, T, C)
    Output: logits (batch, T, num_classes)

    TODO (member owning this model):
    - `nn.TransformerEncoder` needs a `src_key_padding_mask` (batch, T) with True
      at PADDED positions (note: this is the OPPOSITE convention from the `mask`
      returned by dataset.py, which is True at REAL positions -- invert it:
      `padding_mask = ~mask`).
    - Keep num_layers/d_model small first (season sequences here are short --
      dozens of timesteps, not thousands -- a full-size Transformer is
      overkill and will overfit; this is a benchmark, not a leaderboard entry).
    - Optional ablation for the paper: compare fixed sinusoidal PE (below) vs. a
      learned positional embedding -- cheap to add, gives you one more row in
      the results table.
    """

    def __init__(self, in_channels: int, num_classes: int = 4, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.pos_enc(self.input_proj(x))
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return self.classifier(h)
