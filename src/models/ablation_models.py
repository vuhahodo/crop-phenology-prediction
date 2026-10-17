"""Ablation models for dissecting CBA-PhenoNet contributions:
1. TransformerCyclicalDOY: Standard Transformer + Cyclical DOY Positional Encoding (Eq. 1), NO Calendar Gate.
2. TransformerCalendarGate: Standard Transformer + Calendar Gating Prior (Eq. 2-3), standard sinusoidal PE (NO Cyclical DOY).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import SinusoidalPositionalEncoding
from .cba_phenonet import DayOfYearPositionalEncoding, CBAPhenoNetLayer


class TransformerCyclicalDOY(nn.Module):
    """Configuration 2: Transformer + Cyclical DOY Positional Encoding only.
    Replaces sequential index PE with continuous Day-of-Year cyclical embedding.
    No calendar gating prior is applied."""
    DOY_CHANNELS = (1, 2)

    def __init__(self, in_channels: int, num_classes: int = 4, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_enc = DayOfYearPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        doy_sin_cos = x[:, :, self.DOY_CHANNELS[0]:self.DOY_CHANNELS[1] + 1]
        h = self.input_proj(x) + self.pos_enc(doy_sin_cos)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return self.classifier(h)


class TransformerCalendarGate(nn.Module):
    """Configuration 3: Transformer + Calendar Gating Prior only.
    Retains standard sequential index sinusoidal PE, but inserts the PhenologyGate
    after each self-attention layer modulated by DOY signal."""
    DOY_CHANNELS = (1, 2)

    def __init__(self, in_channels: int, num_classes: int = 4, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.layers = nn.ModuleList([
            CBAPhenoNetLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        doy_sin_cos = x[:, :, self.DOY_CHANNELS[0]:self.DOY_CHANNELS[1] + 1]
        h = self.pos_enc(self.input_proj(x))
        for layer in self.layers:
            h = layer(h, doy_sin_cos, padding_mask=padding_mask)
        return self.classifier(h)


class ResidualPhenologyGate(nn.Module):
    def __init__(self, d_model: int, hidden: int = 16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, doy_sin_cos: torch.Tensor) -> torch.Tensor:
        gate_values = self.gate(doy_sin_cos)
        return x + x * gate_values

class CBAPhenoNetResidualLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.gate = ResidualPhenologyGate(d_model)

    def forward(self, x: torch.Tensor, doy_sin_cos: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder_layer(x, src_key_padding_mask=padding_mask)
        return self.gate(h, doy_sin_cos)

class CBAPhenoNetResidual(nn.Module):
    """Configuration A1: CBA-PhenoNet with Residual Calendar Gate."""
    DOY_CHANNELS = (1, 2)
    def __init__(self, in_channels: int, num_classes: int = 4, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_enc = DayOfYearPositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            CBAPhenoNetResidualLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        doy_sin_cos = x[:, :, self.DOY_CHANNELS[0]:self.DOY_CHANNELS[1] + 1]
        h = self.input_proj(x) + self.pos_enc(doy_sin_cos)
        for layer in self.layers:
            h = layer(h, doy_sin_cos, padding_mask=padding_mask)
        return self.classifier(h)


class CBAPhenoNetDisentangled(nn.Module):
    """Configuration A2: CBA-PhenoNet with Disentangled Calendar Gate.
    The Calendar Gate is applied only once after all transformer layers, 
    decoupling token-to-token attention from the gating mechanism."""
    DOY_CHANNELS = (1, 2)
    def __init__(self, in_channels: int, num_classes: int = 4, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_enc = DayOfYearPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        from .cba_phenonet import PhenologyGate
        self.gate = PhenologyGate(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        doy_sin_cos = x[:, :, self.DOY_CHANNELS[0]:self.DOY_CHANNELS[1] + 1]
        h = self.input_proj(x) + self.pos_enc(doy_sin_cos)
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        h = self.gate(h, doy_sin_cos)
        return self.classifier(h)
