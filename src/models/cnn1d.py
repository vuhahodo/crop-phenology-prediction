"""Member assigned to 1D-CNN: fill in / tune this file. Fully-convolutional
sequence-to-sequence design (NOT window-then-pool-then-classify) so the output
is a (batch, T, num_classes) per-timestep prediction, matching the many-to-many
segmentation framing in docs/methodology.md -- same input/output contract as
bilstm.py and transformer.py, so all 3 plug into the same src/train.py harness.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Dilated causal-ish conv block: 'same' padding keeps sequence length constant
    so no cropping/alignment bookkeeping is needed when stacking blocks."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, C_in, T)
        out = self.drop(self.act(self.bn(self.conv(x))))
        return out + self.residual(x)


class NDVICNN1D(nn.Module):
    """
    Input:  x (batch, T, C) -- see docs/methodology.md §1.2
    Output: logits (batch, T, num_classes)

    TODO (member owning this model):
    - Tune `channels` / `dilations` (start small: this is a benchmark baseline
      DL model, not meant to be the "best" model -- keep it simple and document
      why in the paper's Methodology section).
    - Try replacing BatchNorm1d with LayerNorm if batch sizes end up small
      (season counts may be limited depending on final data source).
    """

    def __init__(self, in_channels: int, num_classes: int = 4,
                 channels: tuple[int, ...] = (32, 64, 64), dilations: tuple[int, ...] = (1, 2, 4),
                 dropout: float = 0.1):
        super().__init__()
        assert len(channels) == len(dilations)
        blocks = []
        prev = in_channels
        for ch, dil in zip(channels, dilations):
            blocks.append(ConvBlock(prev, ch, kernel_size=5, dilation=dil, dropout=dropout))
            prev = ch
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Conv1d(prev, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)  # (batch, C, T) -- Conv1d expects channels-first
        for block in self.blocks:
            h = block(h)
        logits = self.classifier(h)  # (batch, num_classes, T)
        return logits.transpose(1, 2)  # (batch, T, num_classes)
