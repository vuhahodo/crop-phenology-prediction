---
title: team_workload
type: note
permalink: timeseries-final-crop-phenology/docs/team-workload
---

# Task 4 — Team Workload Distribution (5 thành viên)

Deadline gợi ý bám theo lịch giảng dạy thật của AI2029 (`De cuong-Phan tich du lieu chuoi thoi
gian.pdf` §7): Tuần 8 = kiểm tra giữa kỳ, Tuần 9-10 = Chương 4 (ML), Tuần 11-14 = Chương 5 (DL +
seminar Tuần 14), Tuần 15 = ôn tập + hoàn thiện đồ án. A3 (Bài tập lớn, 50% điểm tổng kết, CLO3+
CLO4) là deliverable cuối cùng.

| # | Thành viên | Vai trò chính | Deliverable / model phụ trách | File code sở hữu | Deadline gợi ý |
|---|---|---|---|---|---|
| 1 | M1 | **Data & Preprocessing lead** (critical path — 4 người còn lại phụ thuộc vào output của M1) | Chốt nguồn dữ liệu (còn mở, xem `literature-review...md` §5) + viết pipeline: gap-filling, Savitzky-Golay/Whittaker smoothing, gán nhãn 4 lớp (FALLOW/VEGETATIVE/REPRODUCTIVE/SENESCENCE) từ SOS/POS/EOS, xuất ra `list[SeasonRecord]` (implement `load_records()` trong `src/train.py`) | `src/train.py::load_records`, (mới) `src/data/loading.py` | Tuần 8-9 (trước khi 4 người kia cần dữ liệu để train) |
| 2 | M2 | **Baseline ML** (Random Forest + XGBoost) | Hoàn thiện `build_rf_baseline`/`build_xgboost_baseline`, tune hyperparameter, xử lý mất cân bằng lớp (`class_weight`/`sample_weight`), chạy `TimeSeriesSplit`/`GroupKFold` theo `methodology.md` §1.3 | `src/models/baseline_ml.py`, `src/data/features.py` | Tuần 10 (khớp lab4: RF/XGBoost + TimeSeriesSplit) |
| 3 | M3 | **1D-CNN** (Baseline DL) | Tune kiến trúc `NDVICNN1D` (số layer, dilation, dropout), so sánh với baseline ML | `src/models/cnn1d.py` | Tuần 12 |
| 4 | M4 | **Bi-LSTM/GRU** (Sequential DL) | Tune `NDVIBiLSTM` (thử cả 2 `cell_type`), xử lý sequence length không đều (packing nếu cần) | `src/models/bilstm.py` | Tuần 12-13 (khớp lab5: LSTM/GRU) |
| 5 | M5 | **Transformer/Attention (SOTA DL) + CBA-PhenoNet (đề xuất) + Evaluation/Report lead** | Tune `NDVITransformer` VÀ `CBA-PhenoNet` (đã có kết quả thật với early-stopping: macro-F1 0.887 vs 0.903, POS MAE 5.8 vs 1.2 ngày — xem README); đồng thời tổng hợp `results/comparison_table.csv` của cả 4 người kia thành bảng/biểu đồ (§4.2, §4.4 `paper_outline.md`), điều phối viết báo cáo + slide | `src/models/transformer.py`, `src/models/cba_phenonet.py`, `docs/paper_outline.md` §4 | Tuần 13 (model) + Tuần 14-15 (tổng hợp + báo cáo) |

## Ràng buộc chung cho cả 5 người (không ai được tự ý phá vỡ)

1. **Không ai tự viết lại metric** — mọi người BẮT BUỘC gọi `evaluation.metrics.build_eval_result()`
   qua `src/train.py` để kết quả nằm chung 1 file `results/comparison_table.csv`, so sánh được
   với nhau (xem `methodology.md` §1.4).
2. **Không ai tự chia train/val/test theo cách riêng** — dùng đúng split theo `season_id` mà M1
   cung cấp, để tránh data leakage khác nhau giữa các model (nếu người này leak mà người kia
   không, bảng so sánh sẽ sai lệch không công bằng).
3. Mọi thay đổi kiến trúc/hyperparameter đáng kể nên ghi lại lý do ngắn gọn trong docstring của
   file model tương ứng (đã có sẵn khung "TODO" trong từng file) — phục vụ viết Methodology
   section (`paper_outline.md` §3.4) sau này mà không cần nhớ lại từ đầu.
4. Tuần 14 (seminar theo lịch môn học): mỗi người trình bày ngắn kết quả model của mình + 1 khó
   khăn gặp phải — đúng yêu cầu CLO1 (thuyết trình) và chuẩn bị sẵn nội dung cho A3.

## Vai trò phụ liên quan CLO1 (làm việc nhóm, thuyết trình, viết báo cáo)

- M5 giữ vai trò tổng hợp cuối, nhưng **viết báo cáo là việc chung** — mỗi người tự viết phần
  Methodology (§3.4 tương ứng model của mình) và phần Results/Discussion liên quan model của
  mình trong `paper_outline.md`; M5 chỉ ghép nối + đảm bảo văn phong nhất quán, không viết thay
  toàn bộ.
- Đề xuất họp ngắn cuối mỗi tuần (Tuần 9-14) để M1 cập nhật tiến độ data pipeline (nếu chậm, mọi
  người khác bị block) — quản lý tiến độ là 1 phần chuẩn đầu ra CLO1.