"""Member assigned to Bi-LSTM/GRU: fill in / tune this file. Same
(batch, T, C) -> (batch, T, num_classes) contract as cnn1d.py / transformer.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NDVIBiLSTM(nn.Module):
    """
    Input:  x (batch, T, C)
    Output: logits (batch, T, num_classes)

    TODO (member owning this model):
    - Set `cell_type="gru"` to get the GRU variant instead of a second file --
      the course syllabus (Ch.5) treats LSTM/GRU as sibling architectures with
      the same gating idea; report BOTH in the paper's ablation if time allows
      (cheap: same class, one constructor arg).
    - Consider `num_layers=2` + higher dropout if training season count is small
      (bidirectional doubles effective parameter count vs. a plain LSTM).
    """

    def __init__(self, in_channels: int, num_classes: int = 4, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.1, cell_type: str = "lstm"):
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell_type]
        self.rnn = rnn_cls(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional concat

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # lengths (batch,): optional, for pack_padded_sequence if variable-length
        # seasons hurt training in practice -- start WITHOUT packing (simpler),
        # add packing only if the padding-heavy batches turn out to bias the LSTM.
        h, _ = self.rnn(x)  # (batch, T, hidden*2)
        return self.classifier(h)  # (batch, T, num_classes)
