"""Shared training/evaluation CLI harness -- ALL 5 team members run their model
through this same entry point, so results/comparison_table.csv always has 5
directly-comparable rows (docs/team_workload.md). Each member only needs to:
  1. Build their model (already stubbed in src/models/*.py).
  2. Provide train/val/test `SeasonRecord` lists (from whatever data-loading
     code the team writes once data source is finalized -- see
     literature-review-crop-phenology-ndvi.md §5, still open).
  3. Run: python -m src.train --model <name> --member "<your name>"

Usage:
    python -m src.train --model cnn1d       --member "Nguyen A"
    python -m src.train --model bilstm       --member "Tran B"
    python -m src.train --model transformer  --member "Le C"
    python -m src.train --model rf           --member "Pham D"
    python -m src.train --model xgboost      --member "Hoang E"
"""
from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data.dataset import NDVISequenceDataset, SeasonRecord
from .evaluation.metrics import Timer, build_eval_result, segmentation_metrics

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "comparison_table.csv"
PREDICTIONS_DIR = Path(__file__).resolve().parent.parent / "results" / "predictions"

_ALL_RECORDS_CACHE: list[SeasonRecord] | None = None


def _get_all_records() -> list[SeasonRecord]:
    """Loads once per process and caches -- load_records() is called 2-3x per
    run (train/val/test), and re-reading+re-sampling BreizhCrops each time
    would be wasteful (and, worse, could silently draw a DIFFERENT random
    sample per call if the RNG state weren't reset, corrupting the split)."""
    global _ALL_RECORDS_CACHE
    if _ALL_RECORDS_CACHE is None:
        from .data.breizhcrops_loader import load_breizhcrops_records
        # 1000/class (up from an initial 300/class smoke-test run) -- more data
        # for a real, non-toy comparison; frh01 has 5k-44k parcels per annual
        # crop class, plenty of headroom (see the 2026-09-13 research note).
        _ALL_RECORDS_CACHE = load_breizhcrops_records(region="frh01", max_per_class=1000)
    return _ALL_RECORDS_CACHE


def load_records(split: str) -> list[SeasonRecord]:
    """Real data pipeline (see docs/methodology.md, and the 2026-09-13 dataset
    research note in the basic-memory project timeseries-final-crop-phenology
    for why BreizhCrops was chosen): frh01 region, stratified-sampled to
    annual-crop parcels, phenology-stage-labeled via the classical threshold
    method in src/data/breizhcrops_loader.py (a disclosed pseudo-labeling
    limitation -- see docs/paper_outline.md §5, no independent SOS/POS/EOS
    ground truth exists for this dataset).

    Split 70/15/15, stratified by crop type, fixed seed. Each SeasonRecord is
    one independent parcel (not multiple rows of the same season), so this
    flat split is already leakage-safe per docs/methodology.md §1.3 -- no
    GroupKFold needed here (that matters more once/if multi-year data is added).
    """
    from sklearn.model_selection import train_test_split

    records = _get_all_records()
    crop_labels = [r.season_id.rsplit("_", 1)[-1] for r in records]

    train_val, test, train_val_labels, _ = train_test_split(
        records, crop_labels, test_size=0.15, random_state=42, stratify=crop_labels,
    )
    train, val = train_test_split(
        train_val, test_size=0.15 / 0.85, random_state=42, stratify=train_val_labels,
    )
    return {"train": train, "val": val, "test": test}[split]


def _evaluate_macro_f1(model, loader, device: str) -> float:
    """Val-set macro-F1 for early stopping -- same segmentation_metrics() every
    model reports through, so "best epoch" is picked by the actual metric that
    matters (docs/methodology.md §1.4), not just lowest val loss."""
    model.eval()
    all_true, all_pred, all_mask = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["x"].to(device))
            all_true.append(batch["y"].numpy())
            all_pred.append(logits.argmax(dim=-1).cpu().numpy())
            all_mask.append(batch["mask"].numpy())
    model.train()
    y_true, y_pred, mask = np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_mask)
    return segmentation_metrics(y_true, y_pred, mask)["macro_f1"]


def train_dl_model(model_name: str, device: str = "cpu", max_epochs: int = 150, patience: int = 15, seed: int = 42):
    # Found while re-running for the instructor's deeper-evaluation pass: this
    # function had NO seed for model init/dropout/shuffle (only the train/val/
    # test SPLIT was seeded, in load_records) -- two runs of the same model
    # produced different macro-F1 and even flipped which of
    # Transformer/CBA-PhenoNet scored higher. Seeding here makes a single run
    # reproducible; it does NOT replace proper multi-seed variance reporting
    # (still an open "future work" item, see docs/paper_outline.md §6).
    torch.manual_seed(seed)

    train_records = load_records("train")
    val_records = load_records("val")
    test_records = load_records("test")

    max_len = max(len(r.ndvi_smoothed) for r in train_records + val_records + test_records)
    train_ds = NDVISequenceDataset(train_records, max_len=max_len)
    val_ds = NDVISequenceDataset(val_records, max_len=max_len)
    test_ds = NDVISequenceDataset(test_records, max_len=max_len)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    in_channels = train_ds.n_channels
    if model_name == "cnn1d":
        from .models.cnn1d import NDVICNN1D
        model = NDVICNN1D(in_channels=in_channels).to(device)
    elif model_name == "bilstm":
        from .models.bilstm import NDVIBiLSTM
        model = NDVIBiLSTM(in_channels=in_channels).to(device)
    elif model_name == "transformer":
        from .models.transformer import NDVITransformer
        model = NDVITransformer(in_channels=in_channels).to(device)
    elif model_name == "cba_phenonet":
        from .models.cba_phenonet import CBAPhenoNet
        model = CBAPhenoNet(in_channels=in_channels).to(device)
    else:
        raise ValueError(model_name)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=NDVISequenceDataset.PAD_LABEL)

    # Real early stopping on held-out val_loader (previously loaded but never
    # used -- a fixed 20-epoch loop was a smoke-test, not a real comparison).
    # Tracks best val macro-F1, restores those weights before final test eval,
    # so "training time" reflects epochs actually needed to converge, not an
    # arbitrary fixed budget -- still comparable across models since all 4 use
    # the identical stopping rule (docs/methodology.md §1.4).
    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    with Timer() as timer:
        model.train()
        for epoch in range(max_epochs):
            for batch in train_loader:
                x, y = batch["x"].to(device), batch["y"].to(device)
                optimizer.zero_grad()
                logits = model(x)  # (batch, T, num_classes)
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
    train_seconds = timer["elapsed"]

    model.load_state_dict(best_state)
    model.eval()
    all_true, all_pred, all_doy, all_ndvi, all_mask, all_season_ids = [], [], [], [], [], []
    inference_start = time.perf_counter()
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["x"].to(device))
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_true.append(batch["y"].numpy())
            all_pred.append(preds)
            all_doy.append(batch["doy"].numpy())
            all_ndvi.append(batch["ndvi"].numpy())
            all_mask.append(batch["mask"].numpy())
            all_season_ids.extend(batch["season_id"])  # list[str], not a tensor
    inference_ms_per_sample = (time.perf_counter() - inference_start) * 1000 / len(test_ds)

    return (
        np.concatenate(all_true), np.concatenate(all_pred),
        np.concatenate(all_doy), np.concatenate(all_ndvi), np.concatenate(all_mask),
        train_seconds, inference_ms_per_sample, all_season_ids,
    )


def train_ml_baseline(model_name: str):
    from .models.baseline_ml import build_rf_baseline, build_xgboost_baseline, predict_tabular
    from .data.features import FEATURE_COLUMNS, build_feature_table

    train_records = load_records("train")
    test_records = load_records("test")
    model = build_rf_baseline() if model_name == "rf" else build_xgboost_baseline()

    train_df = build_feature_table(train_records)
    with Timer() as timer:
        model.fit(train_df[FEATURE_COLUMNS], train_df["label"])
    train_seconds = timer["elapsed"]

    # Main eval pass (predict-only, model already fit above -- fixes an
    # earlier bug where timing accidentally included a full re-fit).
    y_true, y_pred, doy, ndvi, mask = predict_tabular(model, test_records)

    # Separate, pure-inference timing pass over the whole test set (more
    # stable than timing a single season) -- model.predict() only, no fitting.
    inference_start = time.perf_counter()
    predict_tabular(model, test_records)
    inference_ms_per_sample = (time.perf_counter() - inference_start) * 1000 / len(test_records)

    season_ids = [r.season_id for r in test_records]
    return y_true, y_pred, doy, ndvi, mask, train_seconds, inference_ms_per_sample, season_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["cnn1d", "bilstm", "transformer", "cba_phenonet", "rf", "xgboost"])
    parser.add_argument("--member", required=True, help="your name, for the shared results table")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42, help="DL model init/dropout/shuffle seed -- run with 2-3 different seeds and compare mean+/-std before trusting a single run's model ranking (see docs/paper_outline.md future-work note on multi-seed reporting)")
    args = parser.parse_args()

    if args.model in ("cnn1d", "bilstm", "transformer", "cba_phenonet"):
        y_true, y_pred, doy, ndvi, mask, train_s, infer_ms, season_ids = train_dl_model(args.model, args.device, seed=args.seed)
    else:
        y_true, y_pred, doy, ndvi, mask, train_s, infer_ms, season_ids = train_ml_baseline(args.model)

    result, predictions_df = build_eval_result(
        model_name=args.model, member_name=args.member,
        y_true=y_true, y_pred=y_pred, doy=doy, ndvi=ndvi, mask=mask,
        train_seconds=train_s, inference_ms_per_sample=infer_ms,
        season_ids=season_ids,
    )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = result.to_row()
    write_header = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    # Per-season predictions (Task 2 visualize.py / Task 3 error analysis need
    # this) -- one file per model, overwritten each run with the latest weights'
    # predictions (matches "use the newest row" convention already used for
    # comparison_table.csv, see docs/paper_outline.md §4.2).
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = PREDICTIONS_DIR / f"{args.model}_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)

    print(f"Appended result for model={args.model} member={args.member} to {RESULTS_PATH}")
    print(f"Saved {len(predictions_df)} per-season predictions to {predictions_path}")
    print(row)


if __name__ == "__main__":
    main()
