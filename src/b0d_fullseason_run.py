"""Full-season tabular runs and shared protocol-A decoding for B0d item 2."""
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.data.dataset import cyclical_doy
from src.make_tables import decode_milestones
from src.train import load_records

SEEDS = [42, 123, 2024, 7, 314, 0, 1, 999, 555, 77]


def features(records):
    X, y, per_season = [], [], []
    for rec in records:
        ndvi = np.asarray(rec.ndvi_smoothed, dtype=np.float32)
        if len(ndvi) != 73:
            raise ValueError(f"season {rec.season_id}: expected 73 NDVI, got {len(ndvi)}")
        stats = np.array([ndvi.min(), ndvi.max(), ndvi.mean(), ndvi.std(),
                          np.percentile(ndvi, 10), np.percentile(ndvi, 25),
                          np.percentile(ndvi, 50), np.percentile(ndvi, 75),
                          np.percentile(ndvi, 90), np.ptp(ndvi), np.diff(ndvi).min(),
                          np.diff(ndvi).max()], dtype=np.float32)
        cyc = cyclical_doy(rec.doy)
        time_features = np.column_stack((np.arange(73, dtype=np.float32) / 72.0, cyc))
        X.append(np.column_stack((np.tile(ndvi, (73, 1)), np.tile(stats, (73, 1)), time_features)))
        y.append(np.asarray(rec.labels, dtype=np.int64))
        per_season.append(rec)
    return np.concatenate(X), np.concatenate(y), per_season


def pack_predictions(records, preds):
    n, t = len(records), 73
    yt = np.stack([r.labels for r in records]).astype(np.int64)
    d = np.stack([r.doy for r in records]).astype(np.int64)
    nd = np.stack([r.ndvi_smoothed for r in records]).astype(np.float32)
    mask = np.ones((n, t), dtype=bool)
    return dict(y_true=yt, y_pred=np.asarray(preds).reshape(n, t), doy=d, ndvi=nd, mask=mask,
                season_ids=np.array([r.season_id for r in records]))


def protocol_metrics(data, config, seed, fit_s=None, infer_s=None):
    yt, yp, doy, ndvi = (data[k] for k in ("y_true", "y_pred", "doy", "ndvi"))
    recall, errors = {m: [] for m in ("sos", "pos", "eos")}, {m: [] for m in ("sos", "pos", "eos")}
    detected = {m: 0 for m in ("sos", "pos", "eos")}
    true_positive = {m: 0 for m in ("sos", "pos", "eos")}
    true_n = {m: 0 for m in ("sos", "pos", "eos")}
    for i in range(len(yt)):
        a = decode_milestones(yt[i], doy[i], ndvi[i])
        b = decode_milestones(yp[i], doy[i], ndvi[i])
        for j, name in enumerate(("sos", "pos", "eos")):
            true_n[name] += a[j] is not None
            detected[name] += b[j] is not None
            if a[j] is not None and b[j] is not None:
                true_positive[name] += 1
                errors[name].append(abs(int(a[j]) - int(b[j])))
    result = {"config": config, "seed": seed, "macro_f1": f1_score(yt.ravel(), yp.ravel(), average="macro"),
              "fit_seconds": fit_s, "inference_seconds": infer_s}
    for name in ("sos", "pos", "eos"):
        result[f"{name}_recall"] = true_positive[name] / true_n[name] if true_n[name] else float("nan")
        result[f"{name}_detected"] = detected[name]
        result[f"{name}_true"] = true_n[name]
        result[f"{name}_mae_days"] = float(np.mean(errors[name])) if errors[name] else float("nan")
    return result


def main():
    train, test = load_records("train"), load_records("test")
    xtr, ytr, _ = features(train)
    xte, yte, test_records = features(test)
    outdir = Path("results/preds")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in ("rf_fullseason", "xgboost_fullseason"):
        for seed in SEEDS:
            start = time.perf_counter()
            if model_name == "rf_fullseason":
                model = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1,
                                               random_state=seed)
            else:
                model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                      random_state=seed, n_jobs=-1, eval_metric="mlogloss")
            model.fit(xtr, ytr)
            fit_s = time.perf_counter() - start
            start = time.perf_counter()
            pred = model.predict(xte)
            infer_s = time.perf_counter() - start
            data = pack_predictions(test_records, pred)
            np.savez_compressed(outdir / f"{model_name}_{seed}.npz", **data)
            rows.append(protocol_metrics(data, model_name, seed, fit_s, infer_s))
    for config in ("cnn1d", "bilstm"):
        for seed in SEEDS:
            path = outdir / f"{config}_{seed}.npz"
            if path.exists():
                data = dict(np.load(path, allow_pickle=True))
                if data["y_pred"].shape[1] == 73:
                    rows.append(protocol_metrics(data, config, seed))
    output = Path("results/tables/b0d_fullseason_protocol_a_per_seed.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["config"])
        w.writeheader(); w.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
