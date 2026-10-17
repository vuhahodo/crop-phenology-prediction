---
title: b0d_ledger
type: note
permalink: timeseries-final-crop-phenology/results/b0d-ledger
---

# B0d ledger

| Hạng mục | Trạng thái | Bằng chứng |
|---|---|---|
| 1. Gap: LEGACY/RAW/INTERP, 5 mức dropout, 3 mask/seed, mean±std, paired test theo seed | PASS (Gemini vòng 2/3; vòng 1 script re-check lỗi schema) | `results/tables/b0d_gap_per_seed_mask.csv`; `results/tables/b0d_gap_seed_summary.csv`; `results/tables/b0d_gap_paired_seed_tests.csv`; `src/b0d_gap_recheck.py` |
| 2. Full-season RF/XGBoost (73 NDVI + thống kê mùa); CNN1D/Bi-LSTM, protocol A (EOS 3→0), 10 seed | PASS (Gemini vòng 1/3) | `results/tables/b0d_fullseason_protocol_a_per_seed.csv`; `results/preds/rf_fullseason_{seed}.npz`; `results/preds/xgboost_fullseason_{seed}.npz`; `results/preds/cnn1d_{seed}.npz`; `results/preds/bilstm_{seed}.npz`; `src/b0d_fullseason_run.py` |
| 3. Double-Logistic: hội tụ, RMSE fit, 5 lỗi, EOS, latency | PASS (Gemini vòng 1/3) | `results/tables/b0d_double_logistic_summary.csv`; `results/tables/b0d_double_logistic_per_season.csv`; `results/tables/b0d_double_logistic_failure_examples.csv`; `results/preds/double_logistic_0.npz`; `src/b0d_double_logistic_diag.py` |
| 4. Ablation: RMSE, recall/số mùa phát hiện, p_adj so với 0.05 theo family | PASS (Gemini vòng 1/3) | `results/tables/b0d_ablation_metrics_per_seed.csv`; `results/tables/b0d_ablation_paired_tests.csv`; `src/b0d_ablation_metrics.py` |
| 5. Gộp thăm dò DOY-PE/no-DOY-PE, ghép seed, hậu nghiệm | PASS (Gemini vòng 1/3) | `results/tables/b0d_posthoc_doype_per_seed.csv`; `results/tables/b0d_posthoc_doype_paired_tests.csv`; `src/b0d_posthoc_doype.py` |