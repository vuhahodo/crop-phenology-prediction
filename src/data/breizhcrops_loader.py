"""Real data loading + pseudo-labeling pipeline, wired to the BreizhCrops dataset
(github.com/dl4sits/BreizhCrops, Rußwurm & Pelletier arXiv:1905.11893) -- chosen
per the 2026-09-13 dataset research note (see basic-memory project
timeseries-final-crop-phenology): pip-installable today, no account/request,
small enough for student-scale compute, dense single-season Sentinel-2 NDVI.

IMPORTANT DISCLOSED LIMITATION (must appear in the paper, docs/paper_outline.md
§5): BreizhCrops ships CROP-TYPE labels only -- there is no sowing/harvest
ground truth. Phenology-stage labels here are DERIVED via the classical
20%-of-seasonal-amplitude threshold method already documented in
nghien-cuu-nen-tang-crop-phenology-ndvi.md §3.1 (Jonsson & Eklundh 2004; same
method reported as used operationally for Mekong Delta rice sowing dates in
that note's §5). This is weak/pseudo-supervision: models are learning to
reproduce a classical algorithm's output, not independently-verified
agronomic ground truth. Report this as a limitation, not as validated accuracy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import breizhcrops
import breizhcrops.datasets.breizhcrops as bc_internal

from .dataset import SeasonRecord

# Resolved relative to THIS FILE, not the caller's cwd -- a plain relative
# string default ("data/breizhcrops_raw") silently re-downloaded a duplicate
# 2.4GB copy of the dataset when this was first called from a notebook whose
# kernel cwd was docs/ instead of the project root (found while building
# docs/EDA.ipynb). train.py, EDA notebooks, and any future entry point now
# all resolve to the same on-disk location regardless of cwd.
DEFAULT_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "breizhcrops_raw"

L1C_BANDS = bc_internal.SELECTED_BANDS["L1C"]
B4_IDX = L1C_BANDS.index("B4")
B8_IDX = L1C_BANDS.index("B8")
DOA_IDX = L1C_BANDS.index("doa")

# Only annual row crops trace a single clear bell-curve season -- perennial/
# pasture classes (permanent/temporary meadows, orchards, nuts) stay green
# ~year-round with no single SOS/EOS. Confirmed empirically: the smallest
# region (belle-ile) is 86% meadows and unusable for this task; use a
# mainland region (frh01-frh04) instead for enough annual-crop parcels.
ANNUAL_CROP_CLASSES = {"barley", "wheat", "rapeseed", "corn", "sunflower"}

DOY_GRID = np.arange(1, 366, 5)  # regular 5-day grid -- Ch.1 gap-filling target


def _raw_transform(x: np.ndarray):
    """Passed as BreizhCrops(transform=...) -- overrides the package's default
    transform (which drops dates and randomly subsamples timesteps; see the
    2026-09-13 research session for why that default is unusable here).
    Returns (ndvi, doy), sorted by date, NOT yet resampled/smoothed."""
    red = x[:, B4_IDX] * 1e-4
    nir = x[:, B8_IDX] * 1e-4
    ndvi = (nir - red) / (nir + red + 1e-8)
    dates = pd.to_datetime(x[:, DOA_IDX].astype("int64"))
    doy = dates.dayofyear.values
    order = np.argsort(doy)
    return ndvi[order], doy[order]


def _resample_and_smooth(ndvi: np.ndarray, doy: np.ndarray, window: int = 9, polyorder: int = 2):
    """Ch.1 preprocessing: linear interpolation onto a regular grid (handles
    irregular Sentinel-2 revisit spacing), then Savitzky-Golay smoothing --
    see nghien-cuu-nen-tang-crop-phenology-ndvi.md §4's technique-mapping table."""
    df = pd.DataFrame({"doy": doy, "ndvi": ndvi}).groupby("doy").mean().reset_index()
    interp = np.interp(DOY_GRID, df["doy"].values, df["ndvi"].values,
                        left=df["ndvi"].values[0], right=df["ndvi"].values[-1])
    win = window if window % 2 == 1 else window + 1
    win = min(win, len(interp) - (1 if len(interp) % 2 == 0 else 0))
    win = max(win, polyorder + 1 + (polyorder + 1) % 2 + 1)
    smoothed = savgol_filter(interp, window_length=win, polyorder=polyorder)
    return smoothed, DOY_GRID.copy()


def _derive_phenology_labels(ndvi: np.ndarray, amplitude_threshold: float = 0.2,
                              reproductive_frac: float = 0.12) -> np.ndarray:
    """20%-of-seasonal-amplitude threshold method. Walks outward from the NDVI
    peak to find the contiguous above-threshold region (SOS..EOS), carves a
    narrow REPRODUCTIVE window around the peak, VEGETATIVE before it,
    SENESCENCE after -- see docs/methodology.md §1.1 label scheme."""
    ndvi_min, ndvi_max = float(ndvi.min()), float(ndvi.max())
    amplitude = ndvi_max - ndvi_min
    threshold = ndvi_min + amplitude_threshold * amplitude
    peak_idx = int(np.argmax(ndvi))
    above = ndvi >= threshold

    sos_idx = peak_idx
    for t in range(peak_idx, -1, -1):
        if not above[t]:
            break
        sos_idx = t

    eos_idx = peak_idx
    for t in range(peak_idx, len(ndvi)):
        if not above[t]:
            break
        eos_idx = t

    half = max(1, int(reproductive_frac * max(eos_idx - sos_idx, 1)))
    r0 = max(sos_idx, peak_idx - half)
    r1 = min(eos_idx, peak_idx + half)

    labels = np.zeros(len(ndvi), dtype=np.int64)
    labels[sos_idx:r0] = 1        # VEGETATIVE
    labels[r0:r1 + 1] = 2         # REPRODUCTIVE
    labels[r1 + 1:eos_idx + 1] = 3  # SENESCENCE
    # outside [sos_idx, eos_idx] stays 0 (FALLOW)
    return labels


def load_breizhcrops_records(region: str = "frh01", root: str | Path = DEFAULT_ROOT,
                              level: str = "L1C", min_amplitude: float = 0.15,
                              min_observations: int = 15, max_per_class: int = 300,
                              min_class_count: int = 20, seed: int = 42) -> list[SeasonRecord]:
    """Returns list[SeasonRecord] ready for NDVISequenceDataset / build_feature_table.

    Regions like frh01 have 100k+ parcels -- reading every single one via
    ds[i] (which reopens the h5 file per item) would take far too long for
    student-scale iteration. Instead: read `ds.index` (metadata only,
    `classname`/`CODE_CULTU` columns, no time-series I/O -- fast) to filter to
    annual-crop classes and pick a stratified random sample (`max_per_class`
    per class) FIRST, then only pay the per-item read cost for that bounded
    sample. `ds.index.iloc[i]` is what BreizhCrops.__getitem__ uses internally,
    so positions here (after reset_index) line up exactly with `ds[i]`.

    Also drops flat curves and edge-peaking curves (season likely cut off by
    the calendar-year boundary) -- see inline comments.
    """
    ds = breizhcrops.BreizhCrops(region=region, root=root, year=2017, level=level,
                                  transform=_raw_transform, verbose=False)

    idx_df = ds.index.reset_index(drop=True)
    annual = idx_df[idx_df["classname"].isin(ANNUAL_CROP_CLASSES)]
    counts = annual["classname"].value_counts()
    usable_classes = counts[counts >= min_class_count].index
    annual = annual[annual["classname"].isin(usable_classes)]

    rng = np.random.RandomState(seed)
    sampled_positions: list[int] = []
    for classname, group in annual.groupby("classname"):
        n = min(max_per_class, len(group))
        sampled_positions.extend(rng.choice(group.index.values, size=n, replace=False).tolist())

    records: list[SeasonRecord] = []
    for pos in sampled_positions:
        (ndvi_raw, doy_raw), y, fid = ds[int(pos)]
        classname = ds.classname[int(y)]
        ndvi_raw = np.asarray(ndvi_raw)
        doy_raw = np.asarray(doy_raw)
        if len(ndvi_raw) < min_observations:
            continue

        smoothed, doy_grid = _resample_and_smooth(ndvi_raw, doy_raw)
        if smoothed.max() - smoothed.min() < min_amplitude:
            continue  # flat curve -- no clear season, drop (see class filter note above)
        peak_idx = int(np.argmax(smoothed))
        if peak_idx < 2 or peak_idx > len(smoothed) - 3:
            continue  # peak at the very edge of the year -- season likely cut off, drop

        labels = _derive_phenology_labels(smoothed)
        if (labels == 1).sum() == 0 or (labels == 3).sum() == 0:
            continue  # degenerate season -- no detected ramp-up or decline limb, drop
        records.append(SeasonRecord(
            season_id=f"{region}_{fid}_{classname}",
            ndvi_smoothed=smoothed.astype(np.float32),
            doy=doy_grid.astype(np.int64),
            labels=labels,
        ))
    return records
