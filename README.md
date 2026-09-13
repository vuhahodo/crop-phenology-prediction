---
title: README
type: note
permalink: timeseries-final-crop-phenology/readme
---

# Crop Phenology Detection from NDVI Time Series (AI2029 — Time Series Data Analysis)

Đồ án môn học **Phân tích dữ liệu chuỗi thời gian (AI2029)**, VKU — benchmark 4 họ mô hình
(Random Forest/XGBoost, 1D-CNN, Bi-LSTM/GRU, Transformer) cho bài toán trích xuất mốc phenology
(gieo sạ/trổ bông/thu hoạch) từ chuỗi NDVI, hướng tới chất lượng NCKH.

## Trạng thái hiện tại

- ✅ Literature review + gap analysis đã xong (`literature-review-crop-phenology-ndvi.md`,
  `nghien-cuu-nen-tang-crop-phenology-ndvi.md`) — 7 gap cụ thể đã xác định.
- ✅ Methodology đã chốt (`docs/methodology.md`): many-to-many segmentation (4 lớp
  FALLOW/VEGETATIVE/REPRODUCTIVE/SENESCENCE) làm khung chính, suy ra DOY (SOS/POS/EOS) làm
  metric phụ.
- ✅ **Dataset đã chốt: BreizhCrops** (Sentinel-2, Brittany/Pháp, github.com/dl4sits/BreizhCrops,
  Rußwurm & Pelletier arXiv:1905.11893) — chọn vì pip-install được ngay, không cần xin quyền,
  đủ nhỏ cho compute cấp sinh viên. **Hạn chế đã công bố**: dataset chỉ có nhãn LOẠI CÂY TRỒNG,
  KHÔNG có ground truth ngày gieo sạ/thu hoạch thật — nhãn phenology (4 lớp) được suy ra
  (pseudo-label) bằng phương pháp ngưỡng 20% biên độ cổ điển (`src/data/breizhcrops_loader.py`),
  không phải ground truth nông học đã kiểm chứng độc lập. Đây là giới hạn PHẢI nêu rõ trong báo
  cáo (`docs/paper_outline.md` §5), không phải phát hiện — team có thể thay dataset sau nếu tìm
  được nguồn tốt hơn có ground truth thật.
- ✅ **6 model đã train THẬT với early stopping (không phải smoke-test)** — 3328 mùa vụ (4 loại
  cây: barley/wheat/rapeseed/corn, 1000/lớp trước khi lọc, 70/15/15 split, 399 mùa vụ test), mỗi
  model DL train tới khi val macro-F1 hết cải thiện (patience=15 epoch, tối đa 150) thay vì số
  epoch cố định — xem `results/comparison_table.csv`:

  | Model | Macro-F1 | SOS MAE (ngày) | POS MAE (ngày) | EOS MAE (ngày) | Train (s) | Inference (ms/mùa vụ) |
  |---|---|---|---|---|---|---|
  | RF | 0.420 | 116.7 | 45.0 | 92.6 | 32.8 | 52.1 |
  | XGBoost | 0.403 | 88.4 | 21.2 | 67.7 | 6.7 | 9.4 |
  | 1D-CNN | 0.609 | 64.3 | 21.9 | 45.2 | 44.4 | 0.21 |
  | Bi-LSTM | 0.772 | 36.6 | 9.4 | 21.6 | 147.3 | 0.22 |
  | Transformer | 0.908 ± 0.013 | 13.7 ± 1.6 | 3.5 ± 2.0 | 8.4 ± 1.1 | ~440 | 0.30 |
  | **CBA-PhenoNet (đề xuất)** | **0.904 ± 0.015** | **13.4 ± 3.3** | **2.1 ± 0.9** | **6.9 ± 0.7** | ~366 | 0.32 |

  Hàng Transformer/CBA-PhenoNet là **mean ± std trên 3 seed** (42, 123, 2024) — đọc kỹ mục ngay
  dưới trước khi trích số liệu này vào báo cáo, có 1 phát hiện quan trọng về độ tin cậy.

  **⚠️ Phát hiện quan trọng (rút kinh nghiệm khi mở rộng đánh giá theo yêu cầu giảng viên): kết
  quả DL không ổn định giữa các lần chạy nếu không cố định random seed.** Lần chạy đầu tiên (chỉ
  1 seed) cho ra "CBA-PhenoNet thắng Transformer ở MỌI metric" — chạy lại với seed khác thì
  Transformer lại nhỉnh hơn. Đã tìm ra nguyên nhân: `train_dl_model` chỉ seed phần chia
  train/val/test, KHÔNG seed init model/dropout/shuffle — đã sửa (`torch.manual_seed` +
  `DataLoader` generator, xem `src/train.py`), rồi chạy **3 seed cho cả 2 model** để có kết luận
  đáng tin cậy thay vì tin 1 lần chạy may rủi:
  - **Macro-F1 và SOS**: KHÔNG có khác biệt có ý nghĩa giữa Transformer và CBA-PhenoNet (2 khoảng
    mean±std chồng lấn nhau hoàn toàn) — không thể khẳng định model nào "thắng" ở đây.
  - **POS**: không nhất quán giữa các seed (CBA-PhenoNet thắng 2/3 seed, nhưng std lớn ở cả 2
    phía) — cần thêm seed mới kết luận chắc được.
  - **EOS**: CBA-PhenoNet thắng Transformer ở **CẢ 3/3 seed** (6.9±0.7 vs 8.4±1.1 ngày) — đây là
    phát hiện duy nhất đủ nhất quán để khẳng định là thật, không phải nhiễu ngẫu nhiên.

  **Kết luận trung thực để dùng trong báo cáo**: Transformer và CBA-PhenoNet cho hiệu năng tổng
  thể tương đương nhau (không có bằng chứng thống kê nào cho thấy 1 bên vượt trội mọi mặt);
  CBA-PhenoNet có lợi thế nhất quán và có thể bảo vệ được ở riêng mốc EOS (thu hoạch). Đây là kết
  luận khiêm tốn hơn nhưng ĐÚNG hơn số liệu 1-lần-chạy ban đầu — chính là giá trị của kỹ thuật
  "multi-seed variance reporting" mà giảng viên yêu cầu mở rộng.
- ⬜ Còn lại: mỗi thành viên (`docs/team_workload.md`) tự tune/cải thiện model mình phụ trách từ
  baseline đã chạy được này, viết phần Methodology/Results tương ứng cho báo cáo
  (`docs/paper_outline.md`). Kết quả smoke-test ban đầu (chưa early-stop) được giữ lại tham khảo
  ở `results/comparison_table_v1_quick.csv`. RF/1D-CNN/Bi-LSTM/XGBoost mới chạy 1 seed — nên áp
  dụng multi-seed tương tự nếu có thời gian trước khi nộp báo cáo cuối.

## Cấu trúc repo

```
docs/
  methodology.md        Task 1 — problem formulation, input/target shape, split, metrics
  paper_outline.md       Task 3 — academic paper outline (Contributions nhắm vào gap #4, #5)
  team_workload.md        Task 4 — bảng phân công 5 thành viên
src/
  data/
    dataset.py            SeasonRecord + NDVISequenceDataset (dùng chung cho 3 model DL)
    features.py            lag/rolling-window feature builder (dùng cho RF/XGBoost)
    breizhcrops_loader.py       Real data loader: BreizhCrops -> SeasonRecord, pseudo-labeling
  models/
    cnn1d.py                1D-CNN (baseline DL)
    bilstm.py                Bi-LSTM/GRU (sequential DL)
    transformer.py            Time-Series Transformer (SOTA DL)
    cba_phenonet.py             CBA-PhenoNet (team's "beyond baseline" proposed model)
    baseline_ml.py             Random Forest / XGBoost wrappers
  evaluation/
    metrics.py                Timer, segmentation metrics, derived-DOY regression metrics —
                               MỌI model bắt buộc dùng chung file này (docs/methodology.md §1.4)
  train.py                     CLI harness dùng chung: `python -m src.train --model <name> --member "<ten>"`
results/
  comparison_table.csv         Sinh tự động, 1 hàng/lần chạy — không sửa tay
literature-review-crop-phenology-ndvi.md, nghien-cuu-nen-tang-crop-phenology-ndvi.md
  Literature review từ phiên trước (KHÔNG động vào, chỉ tham chiếu)
```

## Chạy thử (dữ liệu thật đã wired sẵn — chạy được ngay)

```bash
pip install -r requirements.txt
pip install breizhcrops xgboost   # tải dataset tự động lần chạy đầu (~2.5GB cho region frh01)
python -m src.train --model rf --member "Ten cua ban"
python -m src.train --model xgboost --member "Ten cua ban"
python -m src.train --model cnn1d --member "Ten cua ban"
python -m src.train --model bilstm --member "Ten cua ban"
python -m src.train --model transformer --member "Ten cua ban"
python -m src.train --model cba_phenonet --member "Ten cua ban"
```

Mỗi lệnh thêm 1 hàng vào `results/comparison_table.csv` — bảng đầy đủ 6 hàng (bao gồm cả
CBA-PhenoNet) đã có sẵn, xem kết quả thật ở mục Trạng thái hiện tại phía trên. Mỗi thành viên chạy
lại lệnh của model mình phụ trách sau khi tune hyperparameter — kết quả mới sẽ APPEND thêm hàng
(không tự xoá hàng cũ, tự lọc/so sánh khi tổng hợp).

## Việc cần làm tiếp theo (không ai được bỏ qua)

1. `docs/team_workload.md`: mỗi người tune model mình phụ trách để cải thiện hơn baseline hiện có.
2. Viết phần Methodology (§3.4) + Results (§4) tương ứng model của mình trong
   `docs/paper_outline.md`, dựa trên số liệu thật đã có.
3. Cân nhắc mở rộng dataset (thêm region khác của BreizhCrops, hoặc nguồn Vietnam/rice nếu tìm
   được — xem gợi ý Phase-2 trong ghi chú nghiên cứu dataset) nếu muốn tăng cỡ mẫu/đa dạng vùng.