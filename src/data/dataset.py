"""Shared PyTorch Dataset for all 3 DL models (1D-CNN, Bi-LSTM, Transformer) --
see docs/methodology.md §1.2 for the exact (batch, T, C) / (batch, T) shape contract.

This is dataset-agnostic on purpose: the team has NOT yet finalized data
source/region/crop (see literature-review-crop-phenology-ndvi.md §5) -- whoever
finishes that decision should write ONE function that produces the arrays this
class expects (see `SeasonRecord` below) and nothing else in this file changes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SeasonRecord:
    """One season instance -- the unit of train/val/test splitting (see
    docs/methodology.md §1.3: split by season/plot/year, NEVER by timestep)."""
    season_id: str
    ndvi_smoothed: np.ndarray   # (T,) float32, already gap-filled + smoothed (Ch.1 pipeline)
    doy: np.ndarray             # (T,) int, day-of-year for each timestep
    labels: np.ndarray          # (T,) int64 in {0,1,2,3} -- see LABEL_NAMES in evaluation/metrics.py
    ndvi_raw: np.ndarray | None = None  # (T,) optional extra channel


def cyclical_doy(doy: np.ndarray) -> np.ndarray:
    """(T,) -> (T, 2) sin/cos encoding. DOY is cyclical (day 365 is adjacent to day 1),
    so raw DOY as a feature would create a false discontinuity at year boundaries."""
    radians = 2 * np.pi * doy.astype(np.float32) / 365.0
    return np.stack([np.sin(radians), np.cos(radians)], axis=-1)


class NDVISequenceDataset(Dataset):
    """Pads every season to `max_len` and returns (x, y, mask).

    x: (T, C) float32 -- channel 0 = smoothed NDVI, channels 1-2 = sin/cos DOY,
       optional channel 3 = raw NDVI if any record provides it.
    y: (T,) int64 -- padded positions filled with -100 (torch's default
       `ignore_index` for CrossEntropyLoss, so padding never contributes to loss).
    mask: (T,) bool -- True where real data, False where padded. Pass to
       evaluation.metrics.segmentation_metrics / build_eval_result so padding
       never contributes to reported scores either.
    """

    PAD_LABEL = -100

    def __init__(self, records: list[SeasonRecord], max_len: int | None = None):
        self.records = records
        self.max_len = max_len or max(len(r.ndvi_smoothed) for r in records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        t = len(r.ndvi_smoothed)
        assert t <= self.max_len, f"season {r.season_id} longer than max_len={self.max_len}"

        doy_enc = cyclical_doy(r.doy)  # (t, 2)
        channels = [r.ndvi_smoothed.astype(np.float32)[:, None], doy_enc]
        if r.ndvi_raw is not None:
            channels.append(r.ndvi_raw.astype(np.float32)[:, None])
        x = np.concatenate(channels, axis=-1)  # (t, C)

        pad_t = self.max_len - t
        x_padded = np.pad(x, ((0, pad_t), (0, 0)), mode="constant")
        y_padded = np.pad(r.labels.astype(np.int64), (0, pad_t), mode="constant", constant_values=self.PAD_LABEL)
        mask = np.zeros(self.max_len, dtype=bool)
        mask[:t] = True
        # Also carry doy/ndvi (unpadded semantics via mask) for downstream
        # derive_dates_from_labels() in evaluation/metrics.py.
        doy_padded = np.pad(r.doy, (0, pad_t), mode="constant")
        ndvi_padded = np.pad(r.ndvi_smoothed.astype(np.float32), (0, pad_t), mode="constant")

        return {
            "x": torch.from_numpy(x_padded),
            "y": torch.from_numpy(y_padded),
            "mask": torch.from_numpy(mask),
            "doy": torch.from_numpy(doy_padded),
            "ndvi": torch.from_numpy(ndvi_padded),
            "season_id": r.season_id,
        }

    @property
    def n_channels(self) -> int:
        return self[0]["x"].shape[-1]
