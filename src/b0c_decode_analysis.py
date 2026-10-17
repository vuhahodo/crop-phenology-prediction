import numpy as np
from src.make_tables import load_preds, CONFIGS, SEEDS
from pathlib import Path
import csv

TABLES_DIR = Path("results/tables")

def decode_milestones(y, doy, ndvi, eos_protocol="3_to_0"):
    """
    Extracts SOS, POS, EOS according to protocol.
    SOS: first 0 -> non-0
    POS: max NDVI among {1,2}
    EOS (3_to_0): first 3 -> 0
    EOS (2_to_3): first 2 -> 3
    """
    sos = pos = eos = None
    T = len(y)
    
    # SOS
    for t in range(1, T):
        if y[t-1] == 0 and y[t] in (1, 2, 3):
            sos = doy[t]
            break
            
    # POS
    veg_repro = np.where(np.isin(y, [1, 2]))[0]
    if len(veg_repro) > 0:
        peak_idx = veg_repro[np.argmax(ndvi[veg_repro])]
        pos = doy[peak_idx]
        
    # EOS
    if eos_protocol == "3_to_0":
        for t in range(1, T):
            if y[t-1] == 3 and y[t] == 0:
                eos = doy[t]
                break
    elif eos_protocol == "2_to_3":
        for t in range(1, T):
            if y[t-1] == 2 and y[t] == 3:
                eos = doy[t]
                break
                
    return sos, pos, eos

def compute_metrics_for_protocol(data, eos_protocol):
    """Computes Precision, Recall, MAE for a single run (data dict)."""
    y_true, y_pred, mask = data["y_true"], data["y_pred"], data["mask"]
    doy, ndvi = data["doy"], data["ndvi"]
    N = y_true.shape[0]
    
    res = {"sos": {"tp":0, "pred_count":0, "true_count":0, "errors":[]},
           "pos": {"tp":0, "pred_count":0, "true_count":0, "errors":[]},
           "eos": {"tp":0, "pred_count":0, "true_count":0, "errors":[]}}
           
    for i in range(N):
        t = mask[i].sum()
        yt = y_true[i, :t]
        yp = y_pred[i, :t]
        d = doy[i, :t]
        n = ndvi[i, :t]
        
        tm = decode_milestones(yt, d, n, eos_protocol)
        pm = decode_milestones(yp, d, n, eos_protocol)
        
        for idx, key in enumerate(["sos", "pos", "eos"]):
            t_val = tm[idx]
            p_val = pm[idx]
            
            if t_val is not None:
                res[key]["true_count"] += 1
            if p_val is not None:
                res[key]["pred_count"] += 1
            if t_val is not None and p_val is not None:
                res[key]["tp"] += 1
                res[key]["errors"].append(abs(t_val - p_val))
                
    # Aggregate
    agg = {}
    for key in ["sos", "pos", "eos"]:
        tp = res[key]["tp"]
        tc = res[key]["true_count"]
        pc = res[key]["pred_count"]
        agg[f"{key}_recall"] = tp / tc if tc > 0 else 0
        agg[f"{key}_precision"] = tp / pc if pc > 0 else 0
        agg[f"{key}_mae"] = np.mean(res[key]["errors"]) if len(res[key]["errors"]) > 0 else np.nan
        
    return agg

def aggregate_seeds(metrics_list):
    """Average metrics across seeds."""
    agg = {}
    keys = metrics_list[0].keys()
    for k in keys:
        vals = [m[k] for m in metrics_list if not np.isnan(m[k])]
        agg[k] = np.mean(vals) if vals else np.nan
    return agg

def main():
    print("Running Decode Analysis for Protocol Sensitivity...")
    
    # We will output a markdown table
    out_lines = [
        "# Phân tích Độ nhạy theo Định nghĩa Giải mã (Protocol Sensitivity)",
        "",
        "Bảng dưới đây so sánh Recall, Precision và MAE cho mốc EOS dưới hai định nghĩa khác nhau: `3->0` (cuối vụ, lá rụng hoàn toàn) và `2->3` (bắt đầu già hoá). Mốc SOS và POS giữ nguyên định nghĩa tiêu chuẩn.",
        "",
        "| Config | EOS Định nghĩa | SOS R / P / MAE | POS R / P / MAE | EOS R / P / MAE |",
        "|---|---|---|---|---|"
    ]
    
    # We will compute and aggregate for all CONFIGS
    for config in CONFIGS:
        seeds_to_run = [0] if "rf" in config or "xgboost" in config or "logistic" in config else SEEDS
        
        metrics_30 = []
        metrics_23 = []
        
        for seed in seeds_to_run:
            data = load_preds(config, seed)
            if data is None: continue
            metrics_30.append(compute_metrics_for_protocol(data, "3_to_0"))
            metrics_23.append(compute_metrics_for_protocol(data, "2_to_3"))
            
        if not metrics_30: continue
        
        agg_30 = aggregate_seeds(metrics_30)
        agg_23 = aggregate_seeds(metrics_23)
        
        def fmt(a, m):
            r = a[f"{m}_recall"]*100
            p = a[f"{m}_precision"]*100
            mae = a[f"{m}_mae"]
            return f"{r:.1f}% / {p:.1f}% / {mae:.1f}d"
            
        out_lines.append(f"| **{config}** | 3->0 | {fmt(agg_30, 'sos')} | {fmt(agg_30, 'pos')} | **{fmt(agg_30, 'eos')}** |")
        out_lines.append(f"| | 2->3 | {fmt(agg_23, 'sos')} | {fmt(agg_23, 'pos')} | **{fmt(agg_23, 'eos')}** |")
        
    out_path = TABLES_DIR / "protocol_sensitivity.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
        
    print(f"Generated {out_path}")

if __name__ == "__main__":
    main()
