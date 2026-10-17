"""B0d gap protocol: three mask treatments, five levels, three replicates."""
import csv
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader

from src.data.dataset import NDVISequenceDataset
from src.evaluation.metrics import segmentation_metrics
from src.make_tables import decode_milestones
from src.train import load_records

SEEDS = [42, 123, 2024, 7, 314, 0, 1, 999, 555, 77]
CONFIGS = ["transformer", "transformer_cyclical_doy", "transformer_calendar_gate",
           "transformer_scalar_gate", "cba_phenonet", "cba_phenonet_residual",
           "cba_phenonet_disentangled"]
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]
MODES = ["LEGACY", "RAW", "INTERP"]
REPLICATES = range(3)


def model_for(name, channels, device):
    if name == "transformer":
        from src.models.transformer import NDVITransformer
        return NDVITransformer(in_channels=channels).to(device)
    if name == "transformer_cyclical_doy":
        from src.models.ablation_models import TransformerCyclicalDOY
        return TransformerCyclicalDOY(in_channels=channels).to(device)
    if name == "transformer_calendar_gate":
        from src.models.ablation_models import TransformerCalendarGate
        return TransformerCalendarGate(in_channels=channels).to(device)
    if name == "transformer_scalar_gate":
        from src.b0_scalar_gate import TransformerScalarGate
        return TransformerScalarGate(in_channels=channels).to(device)
    if name == "cba_phenonet":
        from src.models.cba_phenonet import CBAPhenoNet
        return CBAPhenoNet(in_channels=channels).to(device)
    if name == "cba_phenonet_residual":
        from src.models.ablation_models import CBAPhenoNetResidual
        return CBAPhenoNetResidual(in_channels=channels).to(device)
    from src.models.ablation_models import CBAPhenoNetDisentangled
    return CBAPhenoNetDisentangled(in_channels=channels).to(device)


def altered_batch(x, mask, level, mode, rng):
    z = x.clone()
    for i in range(x.shape[0]):
        n = int(mask[i].sum())
        k = max(1, int(n * level))
        start = int(rng.integers(0, n - k + 1))
        ids = np.arange(start, start + k)
        if mode == "RAW":
            z[i, ids, 0] = 0
        elif mode == "LEGACY":
            z[i, ids, :] = 0
        else:
            good = np.ones(n, dtype=bool)
            good[ids] = False
            keep = np.flatnonzero(good)
            if len(keep):
                z[i, ids, 0] = torch.as_tensor(np.interp(ids, keep, z[i, keep, 0].cpu().numpy()), dtype=z.dtype)
            else:
                z[i, ids, 0] = 0
    return z


def score(model, loader, device, level, mode, rng):
    ys, ps, ms, ds, ns = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for b in loader:
            x = altered_batch(b["x"], b["mask"], level, mode, rng)
            logits = model(x.to(device))
            ys.append(b["y"].numpy())
            ps.append(logits.argmax(-1).cpu().numpy())
            ms.append(b["mask"].numpy())
            ds.append(b["doy"].numpy())
            ns.append(b["ndvi"].numpy())
    y, p, m, d, n = map(np.concatenate, (ys, ps, ms, ds, ns))
    f1 = segmentation_metrics(y, p, m)["macro_f1"]
    errs = []
    for i in range(len(y)):
        L = int(m[i].sum())
        truth = decode_milestones(y[i, :L], d[i, :L], n[i, :L])
        pred = decode_milestones(p[i, :L], d[i, :L], n[i, :L])
        errs.extend(abs(a-b) for a, b in zip(truth, pred) if a is not None and b is not None)
    return f1, float(np.mean(errs)) if errs else float("nan")


def main():
    out = Path("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    records = load_records("test")
    ds = NDVISequenceDataset(records, max_len=max(map(lambda r: len(r.ndvi_smoothed), records)))
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for config in CONFIGS:
        for seed in SEEDS:
            path = Path(f"results/models_b0c/{config}_{seed}.pt")
            if not path.is_file():
                continue
            model = model_for(config, ds.n_channels, device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            for level in LEVELS:
                for rep in REPLICATES:
                    mask_seed = seed * 100000 + int(level * 1000) * 10 + rep
                    for mode in MODES:
                        f1, mae = score(model, loader, device, level, mode, np.random.default_rng(mask_seed))
                        rows.append({"config": config, "seed": seed, "dropout": level, "replicate": rep,
                                     "mask_seed": mask_seed, "mode": mode, "macro_f1": f1, "milestone_mae_days": mae})
            del model
    raw = out / "b0d_gap_per_seed_mask.csv"
    with raw.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["config"])
        w.writeheader(); w.writerows(rows)
    grouped = {}
    for r in rows:
        grouped.setdefault((r["config"], r["seed"], r["dropout"], r["mode"]), []).append(r)
    smry = []
    for key, vals in grouped.items():
        for metric in ("macro_f1", "milestone_mae_days"):
            arr = np.array([v[metric] for v in vals], dtype=float)
            smry.append(dict(zip(("config", "seed", "dropout", "mode"), key), metric=metric,
                             n_masks=len(arr), mean=float(np.nanmean(arr)), std=float(np.nanstd(arr, ddof=1))))
    with (out / "b0d_gap_seed_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(smry[0]) if smry else ["config"])
        w.writeheader(); w.writerows(smry)
    paired = []
    # Paired tests compare model configurations within each seed and mask condition.
    for level in LEVELS:
        for mode in MODES:
            for metric in ("macro_f1", "milestone_mae_days"):
                for config in CONFIGS[1:]:
                    a, b = [], []
                    for seed in SEEDS:
                        left = grouped.get((config, seed, level, mode), [])
                        right = grouped.get((CONFIGS[0], seed, level, mode), [])
                        if left and right:
                            a.append(np.nanmean([v[metric] for v in left]))
                            b.append(np.nanmean([v[metric] for v in right]))
                    if len(a) >= 2:
                        t, p = stats.ttest_rel(a, b, nan_policy="omit")
                    else:
                        t, p = np.nan, np.nan
                    paired.append(dict(config=config, reference=CONFIGS[0], dropout=level, mode=mode,
                                        metric=metric, n_seeds=len(a), mean_diff=float(np.nanmean(np.array(a)-np.array(b))) if a else np.nan,
                                        t_stat=float(t), p_value=float(p)))
    with (out / "b0d_gap_paired_seed_tests.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0]) if paired else ["config"])
        w.writeheader(); w.writerows(paired)
    print(f"{raw}; results/tables/b0d_gap_seed_summary.csv; results/tables/b0d_gap_paired_seed_tests.csv")


if __name__ == "__main__":
    main()
