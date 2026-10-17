"""Per-season double-logistic convergence, fit error, EOS and measured latency."""
import csv
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

from src.make_tables import decode_milestones
from src.train import load_records


def curve(t, base, amp, sos_mid, sos_rate, eos_mid, eos_rate):
    up = 1.0 / (1.0 + np.exp(-sos_rate * (t - sos_mid)))
    down = 1.0 / (1.0 + np.exp(eos_rate * (t - eos_mid)))
    return base + amp * up * down


def fit(rec):
    t = np.asarray(rec.doy, dtype=np.float64) / 365.0
    y = np.asarray(rec.ndvi_smoothed, dtype=np.float64)
    base = float(np.percentile(y, 10))
    amp = max(float(np.percentile(y, 90) - base), 0.01)
    p0 = [base, amp, 0.25, 20.0, 0.70, 20.0]
    bounds = ([-0.5, 0.001, 0.0, 1.0, 0.3, 1.0], [1.0, 2.0, 0.7, 80.0, 1.0, 80.0])
    p0 = [np.clip(p, lo, hi) for p, lo, hi in zip(p0, bounds[0], bounds[1])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        params, _ = curve_fit(curve, t, y, p0=p0, bounds=bounds, maxfev=5000)
    yhat = curve(t, *params)
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    sos = int(np.clip(round(params[2] * 365), 1, 365))
    eos = int(np.clip(round(params[4] * 365), 1, 365))
    fine = np.linspace(0, 1, 365)
    pos = int(np.argmax(curve(fine, *params))) + 1
    return sos, pos, eos, rmse


def make_pred(rec, milestones):
    sos, pos, eos = milestones
    labels = np.zeros(len(rec.doy), dtype=np.int64)
    if sos is None or eos is None or sos >= eos:
        return labels
    a = int(np.argmin(np.abs(rec.doy - sos)))
    b = int(np.argmin(np.abs(rec.doy - eos)))
    c = int(np.argmin(np.abs(rec.doy - pos))) if pos is not None else (a + b) // 2
    c = min(max(c, a), b)
    labels[a:c] = 1
    labels[c:b] = 2
    labels[b:] = 3
    if b < len(labels) - 1:
        labels[-1] = 0
    return labels


def main():
    records = load_records("test")
    rows, pred_rows = [], []
    total = time.perf_counter()
    for rec in records:
        start = time.perf_counter()
        try:
            milestones = fit(rec)
            status, rmse = "CONVERGED", milestones[3]
        except Exception as exc:
            milestones, status, rmse = (None, None, None), f"FAILED:{type(exc).__name__}", float("nan")
        latency = time.perf_counter() - start
        pred = make_pred(rec, milestones[:3] if milestones[0] is not None else milestones[:3])
        truth = decode_milestones(rec.labels, rec.doy, rec.ndvi_smoothed)
        guessed = decode_milestones(pred, rec.doy, rec.ndvi_smoothed)
        fit_eos = int(milestones[2]) if milestones[0] is not None else None
        rows.append({"season_id": rec.season_id, "fit_status": status, "fit_rmse": rmse,
                     "eos_true_doy": truth[2], "eos_fit_doy": fit_eos,
                     "eos_fit_abs_error_days": abs(int(truth[2])-fit_eos) if truth[2] is not None and fit_eos is not None else "",
                     "eos_protocol_a_decoded_doy": guessed[2],
                     "eos_protocol_a_abs_error_days": abs(int(truth[2])-int(guessed[2])) if truth[2] is not None and guessed[2] is not None else "",
                     "latency_seconds": latency})
        pred_rows.append(pred)
    total_s = time.perf_counter() - total
    ok = [r for r in rows if r["fit_status"] == "CONVERGED"]
    fit_errs = [float(r["eos_fit_abs_error_days"]) for r in rows if r["eos_fit_abs_error_days"] != ""]
    errs = [float(r["eos_protocol_a_abs_error_days"]) for r in rows if r["eos_protocol_a_abs_error_days"] != ""]
    summary = [{"n_seasons": len(rows), "n_converged": len(ok),
                "convergence_rate": len(ok) / len(rows),
                "fit_rmse_mean": float(np.mean([r["fit_rmse"] for r in ok])) if ok else float("nan"),
                "fit_rmse_median": float(np.median([r["fit_rmse"] for r in ok])) if ok else float("nan"),
                "fit_eos_reference_n": len(fit_errs), "fit_eos_mae_days": float(np.mean(fit_errs)) if fit_errs else float("nan"),
                "protocol_a_eos_detected": len(errs), "protocol_a_eos_mae_days": float(np.mean(errs)) if errs else float("nan"),
                "latency_total_seconds": total_s,
                "latency_mean_ms_per_season": 1000 * total_s / len(rows)}]
    out = Path("results/tables"); out.mkdir(parents=True, exist_ok=True)
    for name, data in (("b0d_double_logistic_per_season.csv", rows),
                       ("b0d_double_logistic_summary.csv", summary)):
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    failures = [r for r in rows if str(r["fit_status"]).startswith("FAILED")][:5]
    with (out / "b0d_double_logistic_failure_examples.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(rows[0]); w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(failures)
    recs = load_records("test")
    np.savez_compressed("results/preds/double_logistic_0.npz",
        y_true=np.stack([r.labels for r in recs]), y_pred=np.stack(pred_rows),
        doy=np.stack([r.doy for r in recs]), ndvi=np.stack([r.ndvi_smoothed for r in recs]),
        mask=np.ones((len(recs), 73), dtype=bool), season_ids=np.array([r.season_id for r in recs]))
    print("results/tables/b0d_double_logistic_per_season.csv; results/tables/b0d_double_logistic_summary.csv; results/tables/b0d_double_logistic_failure_examples.csv")


if __name__ == "__main__":
    main()
