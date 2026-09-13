---
title: paper_outline
type: note
permalink: timeseries-final-crop-phenology/docs/paper-outline
---

# Task 3 — Academic Paper Outline

> Neo vào `literature-review-crop-phenology-ndvi.md` (đã có annotated bibliography + 7 gap cụ
> thể) — outline dưới đây được thiết kế để **Contributions** trả lời trực tiếp gap #4 ("không có
> benchmark/protocol chuẩn để so sánh công bằng các họ phương pháp") và gap #5 (foundation
> models chưa được benchmark cho bài toán trích ngày phenology) — đây là 2 gap có sẵn bằng chứng
> tài liệu, không phải tự nhận là "mới" một cách chủ quan.

## Title (draft)

> *A Benchmark Study of Statistical, Machine Learning, and Deep Learning Methods for
> Point-wise Crop Phenology Stage Segmentation from NDVI Time Series*

(Đổi "Statistical" → tên baseline cụ thể nếu team quyết định không dùng thêm baseline threshold/
TIMESAT-style; giữ "Point-wise Segmentation" trong title vì đây là đóng góp phương pháp luận
chính, không phải chỉ là "yet another LSTM for NDVI".)

## Abstract (150–200 từ)
1. Bối cảnh 1 câu: phenology detection (SOS/POS/EOS) từ NDVI quan trọng cho nông nghiệp/an ninh
   lương thực.
2. Gap 1 câu: thiếu benchmark chuẩn so sánh Statistical/ML/DL đầu-đến-cuối trên cùng 1 protocol
   (trích dẫn gap #4, #5).
3. Method 2 câu: framing many-to-many segmentation (không phải regression đơn điểm), 4 model
   (RF/XGBoost, 1D-CNN, Bi-LSTM/GRU, Transformer), cùng bộ metric segmentation + suy ra DOY.
4. Kết quả (số thật, đo trên BreizhCrops, 3328 mùa vụ, 4 loại cây trồng, early-stopping thật trên
   validation set, mean±std trên 3 seed cho 2 model attention-based): macro-F1 tăng từ 0.42 (RF)
   lên **~0.90-0.91** (Transformer/CBA-PhenoNet, chênh lệch giữa 2 model KHÔNG có ý nghĩa thống
   kê); MAE của EOS (ngày thu hoạch) giảm còn **6.9±0.7 ngày** với CBA-PhenoNet (nhất quán ở cả
   3/3 seed) so với 8.4±1.1 ngày của Transformer thường — nằm trong mức SOTA thế giới (~9-10
   ngày, TIMESAT/LSTM) đã tổng hợp trong literature review.
5. Kết luận: kiến trúc attention-based (Transformer/CBA-PhenoNet) vượt trội rõ rệt so với
   RF/XGBoost/CNN/BiLSTM trên bài toán này; giữa 2 kiến trúc attention-based, hiệu năng tổng thể
   (macro-F1, SOS) tương đương nhau trong giới hạn nhiễu ngẫu nhiên (đo qua 3 seed), CBA-PhenoNet
   chỉ có lợi thế nhất quán và đáng kể riêng ở mốc EOS — một kết luận khiêm tốn nhưng trung thực
   hơn báo cáo 1-lần-chạy ban đầu (xem "Phát hiện về reproducibility" ở §4.2).

## 1. Introduction
- 1.1. Motivation: vai trò NDVI phenology trong giám sát mùa vụ, đặc biệt vùng nhiệt đới/lúa
  nước (dẫn case study PhenoRice + ĐBSCL từ `nghien-cuu-nen-tang-crop-phenology-ndvi.md` §5).
- 1.2. Problem statement: định nghĩa chính xác bài toán (many-to-many segmentation + suy ra DOY
  — xem `methodology.md` §1.1).
- 1.3. Research questions (RQ), kế thừa trực tiếp RQ đã đặt ra ở
  `nghien-cuu-nen-tang-crop-phenology-ndvi.md` §1, thu hẹp lại cho phạm vi benchmark:
  - RQ1: Kiến trúc DL (CNN/BiLSTM/Transformer) có vượt trội hơn ML cổ điển (RF/XGBoost) có
    feature engineering thủ công trên bài toán segmentation theo timestep này không?
  - RQ2: Độ chính xác ngày (MAE/RMSE theo SOS/POS/EOS) suy ra từ nhãn dự đoán có tương xứng với
    độ tăng chi phí tính toán (training/inference time) hay không — tức là đâu là điểm cân bằng
    accuracy–cost hợp lý cho use case thực tế (giám sát gần-thời-gian-thực)?
- 1.4. **Contributions** (mục quan trọng nhất để thoả yêu cầu "research-level"):
  1. Một **protocol benchmark thống nhất** (cùng split, cùng nhãn, cùng bộ metric — segmentation
     quality + phenology-date accuracy tính bằng ngày + compute cost) để so sánh công bằng 4 họ
     mô hình khác nhau — trực tiếp giải quyết gap #4 đã xác định trong literature review.
  2. Một **framing many-to-many segmentation** (thay vì regression 1 điểm hoặc rule-based
     threshold) cho bài toán trích SOS/POS/EOS, tận dụng đúng kiến trúc tuần tự — khác với phần
     lớn literature đã khảo sát (chủ yếu dùng threshold/curve-fitting hoặc LSTM-forecast-rồi-suy-
     ra, xem §3 literature review).
  3. (Nếu team giữ định hướng Việt Nam/ĐBSCL đã gợi ý) Bằng chứng thực nghiệm trên bối cảnh vùng
     nhiệt đới/mây nhiều — nơi gap #3 chỉ rõ validation infrastructure còn thiếu.
  4. Phân tích **accuracy–compute trade-off** rõ ràng (không chỉ "model nào tốt nhất" mà "model
     nào đáng chi phí nào") — hữu ích cho triển khai thực tế ở nơi hạ tầng tính toán hạn chế.
  5. **CBA-PhenoNet** (`src/models/cba_phenonet.py`): kết hợp date-aware positional encoding
     (TSViT-style, Tarasiou et al. CVPR 2023) + Phenology Gate (Xu et al., Sensors 2025) — đề
     xuất riêng của team (không phải re-implementation nguyên bản 1 paper). Validated qua 3 seed
     (không chỉ 1 lần chạy — xem phát hiện reproducibility ở §4.2): macro-F1 tương đương
     Transformer thường (không có ý nghĩa thống kê), nhưng thắng nhất quán ở EOS-date MAE
     (**6.9±0.7 vs 8.4±1.1 ngày, cả 3/3 seed**). Nói rõ trong Methodology đây là
     adaptation/combination, không tự nhận là phát minh kiến trúc hoàn toàn mới, và không tự
     nhận thắng ở những metric mà số liệu 3-seed không ủng hộ.

## 2. Related Work
- 2.1. Phenological metric extraction: threshold-based, TIMESAT/curve-fitting, BFAST/breakpoint
  (từ literature review §3, giữ nguyên bảng annotated bibliography, không viết lại).
- 2.2. Machine learning & deep learning cho NDVI phenology (LSTM, ConvLSTM — literature review
  §3.4).
- 2.3. Positioning: bảng so sánh ngắn (paper này vs. TIMESAT/phenex vs. PhenoRice vs. LSTM-2022)
  theo trục: {input framing, ground truth needed, benchmark hoàn chỉnh hay không, có so sánh
  compute cost không} — cột cuối chính là chỗ trống mà Contribution #1 lấp vào.

## 3. Methodology
- 3.1. Data (mô tả nguồn dữ liệu **sau khi team đã chốt** — hiện đang mở, xem literature review
  §5).
- 3.2. Preprocessing: gap-filling + smoothing (Savitzky-Golay hoặc Whittaker — nêu lý do chọn,
  theo bảng ánh xạ Chương 1 trong `nghien-cuu-nen-tang...md` §4).
- 3.3. Problem formulation: many-to-many segmentation + derived DOY (copy nguyên nội dung
  `methodology.md` §1.1–§1.2, rút gọn cho văn phong paper).
- 3.4. Models: 5 sub-section ngắn (RF/XGBoost, 1D-CNN, Bi-LSTM/GRU, Transformer, CBA-PhenoNet đề
  xuất) — mỗi model 1 đoạn giải thích kiến trúc + tham chiếu file code tương ứng (`src/models/*.py`).
- 3.5. Evaluation protocol: split theo season/year (GroupKFold/TimeSeriesSplit, §1.3), bộ metric
  đầy đủ (§1.4) — đây là phần thể hiện rõ nhất Contribution #1.

## 4. Experiments & Results
- 4.1. Experimental setup (hardware, epoch budget giống nhau giữa các model DL, hyperparameter
  chọn qua validation set — không tune trên test set).
- 4.2. Bảng kết quả chính (số thật, `results/comparison_table.csv`, 3328 mùa vụ BreizhCrops, 399
  test, early-stopping thật trên val set — patience=15, tối đa 150 epoch). Transformer/CBA-PhenoNet
  là mean±std trên 3 seed (42, 123, 2024); các model còn lại mới chạy 1 seed (nên cân nhắc mở
  rộng 3-seed tương tự nếu có thời gian):

  | Model | Macro-F1 | SOS MAE | POS MAE | EOS MAE | Train(s) | Infer(ms) |
  |---|---|---|---|---|---|---|
  | RF | 0.420 | 116.7 | 45.0 | 92.6 | 32.8 | 52.1 |
  | XGBoost | 0.403 | 88.4 | 21.2 | 67.7 | 6.7 | 9.4 |
  | 1D-CNN | 0.609 | 64.3 | 21.9 | 45.2 | 44.4 | 0.21 |
  | Bi-LSTM | 0.772 | 36.6 | 9.4 | 21.6 | 147.3 | 0.22 |
  | Transformer | 0.908±0.013 | 13.7±1.6 | 3.5±2.0 | 8.4±1.1 | ~440 | 0.30 |
  | **CBA-PhenoNet** | **0.904±0.015** | **13.4±3.3** | **2.1±0.9** | **6.9±0.7** | ~366 | 0.32 |

  **Phát hiện về reproducibility (đưa vào báo cáo như một phần của "kỹ thuật đánh giá chuyên
  sâu" giảng viên yêu cầu, không phải hạn chế cần giấu)**: lần chạy đầu tiên (1 seed duy nhất)
  từng cho kết quả "CBA-PhenoNet thắng Transformer ở mọi metric" (0.903 vs 0.887 macro-F1, 1.2 vs
  5.8 ngày POS MAE) — nhưng `src/train.py` lúc đó KHÔNG cố định seed cho init model/dropout/
  shuffle (chỉ seed phần chia train/val/test). Sau khi thêm `torch.manual_seed` và chạy lại với
  3 seed khác nhau: macro-F1 và SOS MAE của 2 model chồng lấn hoàn toàn trong khoảng mean±std —
  **không đủ bằng chứng thống kê để nói model nào thắng ở 2 metric này**. Chỉ riêng EOS MAE,
  CBA-PhenoNet thắng ở **cả 3/3 seed** (6.9±0.7 vs 8.4±1.1 ngày) — đây là khác biệt duy nhất đủ
  nhất quán để báo cáo là thật. Kết luận: so sánh 1-lần-chạy là không đủ tin cậy cho loại kiến
  trúc có nhiều nguồn ngẫu nhiên (init/dropout/shuffle) — cần multi-seed reporting trước khi
  khẳng định "model A tốt hơn model B".

  (chạy lại `python -m src.train --model <name> --member <ten> [--seed N]` sau khi tune sẽ append
  hàng mới. Với RF/XGBoost/1D-CNN/Bi-LSTM (1 seed): dùng hàng MỚI NHẤT của mỗi model. Với
  Transformer/CBA-PhenoNet: dùng TRUNG BÌNH của các hàng có `member_name` chứa `[seed=...]` +
  hàng seed=42 gốc — không dùng 1 hàng đơn lẻ, xem phát hiện reproducibility ở trên. Bản
  smoke-test đầu tiên, chưa early-stop, ở `results/comparison_table_v1_quick.csv` — KHÔNG dùng số
  đó cho báo cáo, chỉ giữ tham khảo lịch sử.)
- 4.3. Per-class analysis: lớp nào (FALLOW/VEGETATIVE/REPRODUCTIVE/SENESCENCE) khó nhất cho model
  nào — kỳ vọng REPRODUCTIVE khó nhất (hẹp, giống pattern OFFENSIVE/HATE trong project NLP khác
  đã gặp phải khi lớp hiếm).
- 4.4. Accuracy–compute trade-off plot (accuracy trục Y, inference time trục X, mỗi model 1
  điểm) — trực quan hoá Contribution #4.
- 4.5. Error analysis (đã chạy thật, xem `src/error_analysis.py` + `results/error_analysis.csv`,
  top 5% sai số lớn nhất của CBA-PhenoNet, đã sửa lỗi đo sai khi ngày vòng qua năm — xem
  `circular_abs_error_days` trong `evaluation/metrics.py`): 2 phát hiện thật đáng đưa vào báo cáo
  — (1) **rapeseed chiếm 18/44 ca lỗi nặng nhất** dù chỉ là 1/4 số lớp cây trồng (bất cân xứng rõ
  rệt, có thể do rapeseed ở Pháp gieo trồng vụ thu-đông, đường cong NDVI khác hẳn 3 cây còn lại
  gieo vụ xuân — cần giải thích trong Discussion); (2) **SOS là mốc khó nhất một cách hệ thống**
  (30/44 ca lỗi nặng nhất là SOS, so với 9 POS và 5 EOS) — nhất quán với vị trí SOS thường nằm
  gần biên `FALLOW`→`VEGETATIVE`, nơi tín hiệu NDVI biến đổi chậm và mơ hồ hơn EOS (biên độ giảm
  dốc hơn khi cây chín).

## 5. Discussion
- Trả lời RQ1/RQ2 dựa trên số liệu §4 — **không kết luận trước khi có số thật**.
- Hạn chế: gap #1 (spectral phenology ≠ agronomic phenology nếu không có ground truth thực địa),
  cỡ mẫu (nếu ít season instance); RF/XGBoost/1D-CNN/Bi-LSTM mới chạy 1 seed (chỉ Transformer và
  CBA-PhenoNet đã có 3-seed mean±std) — nêu rõ như một hạn chế còn lại, không giấu.
- So sánh với con số thế giới đã tổng hợp (RMSE ~9 ngày TIMESAT/phenex, ~10 ngày LSTM) — model
  của nhóm nằm ở đâu trên phổ đó.

## 6. Conclusion & Future Work
- Tóm tắt 3 câu: RQ nào được trả lời, model nào là lựa chọn thực tế được khuyến nghị (và cho use
  case nào — realtime monitoring vs. batch analysis cuối vụ).
- Future work: mở rộng multi-seed variance reporting sang RF/XGBoost/1D-CNN/Bi-LSTM (đã làm cho
  Transformer/CBA-PhenoNet, xem §4.2 — phát hiện quan trọng: so sánh 1-seed có thể đảo ngược kết
  luận "model nào tốt hơn"), tencode/domain-specific augmentation nếu ít dữ liệu, thử foundation
  models viễn thám (Prithvi/Presto — gap #5) như một hướng mở rộng.

## References
- Toàn bộ annotated bibliography đã có trong `literature-review-crop-phenology-ndvi.md` +
  `nghien-cuu-nen-tang-crop-phenology-ndvi.md` — copy/format lại theo style yêu cầu của trường
  (BibTeX/APA tuỳ hướng dẫn khoa), không cần tìm lại từ đầu.

## Lưu ý công bố AI (giữ đúng tinh thần đã làm ở literature review)
Nếu dùng AI (Claude) hỗ trợ viết report/soạn outline, cần ghi rõ trong mục Acknowledgment/Method
theo đúng chính sách "Assistive, not deceptive" (xem `academic-research-skills`) — không nộp bài
mà không công bố mức độ hỗ trợ AI, đặc biệt vì đây là NCKH-level submission.