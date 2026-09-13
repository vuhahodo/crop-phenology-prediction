"""CBA-PhenoNet -- the team's own "beyond baseline" contribution (Task 2 follow-up:
the course syllabus baselines -- RF/XGBoost, 1D-CNN, Bi-LSTM/GRU, Transformer --
are a floor, not a ceiling; this file is where the team goes further).

(Renamed from an earlier working name, "PhenoFormer", to CBA-PhenoNet to avoid
collision with an unrelated model of the same name already published
elsewhere -- no change to the architecture itself, see git history / earlier
basic-memory session notes for the original name if cross-referencing older
docs.)

Two concrete, peer-reviewed ideas adapted from remote-sensing time-series
literature (see the 2026-09-13 dataset/architecture research note in
basic-memory project timeseries-final-crop-phenology for full citations):

1. Date-aware temporal encoding (TSViT-style; Tarasiou, Chavez & Tzimiropoulos,
   CVPR 2023): use the REAL day-of-year of each observation to build the
   positional encoding, instead of the sequence INDEX (0..T-1) that plain
   NDVITransformer uses (see transformer.py's SinusoidalPositionalEncoding).
   This matters specifically because satellite revisit is irregular/gappy --
   index-based position silently assumes evenly-spaced timesteps, which is
   false once cloud gaps are interpolated onto a fixed grid but real
   acquisition density still varies by season.
2. Phenology Gate (Xu, Cai, Wei, He & Wang, Sensors 2025, DOI 10.3390/s25247488):
   a small learned gate, conditioned on the same day-of-year signal, that
   reweights attention scores so the model can up-weight timesteps near
   expected phenological transitions instead of attending uniformly.

Same (batch, T, C) -> (batch, T, num_classes) contract as the other 3 DL
models, so it plugs into the identical src/train.py harness and
results/comparison_table.csv -- this is compared AGAINST the required
baselines, not a replacement for running them.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class DayOfYearPositionalEncoding(nn.Module):
    """Sinusoidal encoding keyed on the REAL day-of-year value carried by the
    input (channels 1-2 of x, per dataset.py's cyclical_doy) rather than
    sequence index -- see module docstring, idea #1."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        # learned projection from the 2 cyclical DOY channels to d_model,
        # added to the content embedding (in place of a fixed lookup table
        # indexed by position, since "position" here IS the DOY signal).
        self.proj = nn.Linear(2, d_model)

    def forward(self, doy_sin_cos: torch.Tensor) -> torch.Tensor:
        # doy_sin_cos: (batch, T, 2)
        return self.proj(doy_sin_cos)


class PhenologyGate(nn.Module):
    """Idea #2: a per-timestep scalar gate in [0, 1], conditioned on the
    day-of-year signal, applied to the attention output before it's added
    back (a lightweight approximation of Xu et al. 2025's gated-attention
    mechanism -- see module docstring). Intuition: the model learns WHICH
    calendar windows are informative for phenology transitions (e.g. a rice
    calendar's typical sowing window) and down-weights the rest."""

    def __init__(self, d_model: int, hidden: int = 16):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, doy_sin_cos: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, d_model), doy_sin_cos: (batch, T, 2)
        gate_values = self.gate(doy_sin_cos)  # (batch, T, 1)
        return x * gate_values


class CBAPhenoNetLayer(nn.Module):
    """One Transformer encoder layer + a PhenologyGate applied to its output --
    the block CBA-PhenoNet stacks num_layers times."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.gate = PhenologyGate(d_model)

    def forward(self, x: torch.Tensor, doy_sin_cos: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.encoder_layer(x, src_key_padding_mask=padding_mask)
        return self.gate(h, doy_sin_cos)


class CBAPhenoNet(nn.Module):
    """
    Input:  x (batch, T, C) -- same contract as transformer.py. Assumes
            channels 1-2 are the cyclical DOY encoding (dataset.py convention).
    Output: logits (batch, T, num_classes)

    TODO (member extending this beyond the required baselines):
    - Ablate: date-aware encoding alone vs. + PhenologyGate vs. neither
      (= reduces to plain NDVITransformer) -- report all 3 rows in the paper's
      results table (§4.2/§4.3 in docs/paper_outline.md) to show which idea
      actually contributes, not just the combined number.
    - This is presented as the team's OWN adaptation/combination of two
      published ideas, not a re-implementation of either paper's full model --
      say so explicitly in the paper's Methodology section (academic honesty).
    """

    DOY_CHANNELS = (1, 2)  # indices into the input's channel dim -- see dataset.py

    def __init__(self, in_channels: int, num_classes: int = 4, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.pos_enc = DayOfYearPositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            CBAPhenoNetLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        doy_sin_cos = x[:, :, self.DOY_CHANNELS[0]:self.DOY_CHANNELS[1] + 1]  # (batch, T, 2)
        h = self.input_proj(x) + self.pos_enc(doy_sin_cos)
        for layer in self.layers:
            h = layer(h, doy_sin_cos, padding_mask=padding_mask)
        return self.classifier(h)
