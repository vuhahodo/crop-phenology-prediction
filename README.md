---
title: README
type: note
permalink: timeseries-final-crop-phenology/readme
---

# Hướng dẫn Tái lập (Reproducibility Guide) - Đồ án Phân tích Chuỗi thời gian (AI2029)

Đây là kho mã nguồn đầy đủ phục vụ cho báo cáo bài báo NCKH về đánh giá Robustness và khả năng chống chịu nhiễu mây của các mô hình Deep Learning (Transformer, CBA-PhenoNet, RNN, CNN) và Machine Learning (RF, XGBoost) trên tập dữ liệu chuỗi thời gian NDVI (BreizhCrops).

Toàn bộ quy trình từ huấn luyện (Train), dự đoán (Predict), phân tích thống kê TOST/Bootstrap (Tables), đến vẽ biểu đồ phân tích (Figures) đã được tự động hoá hoàn toàn qua `Makefile`.

## 1. Yêu cầu Môi trường (Environment Requirements)
Môi trường chuẩn (khuyên dùng Python 3.10+):
- `torch >= 2.0.0`
- `xgboost >= 1.7.0`
- `scikit-learn >= 1.2.0`
- `pandas`, `numpy`, `scipy`
- `matplotlib`
- `optuna` (Dành cho việc tuning siêu tham số tự động)
- `breizhcrops` (Tải dữ liệu tự động)

Cài đặt nhanh:
```bash
pip install -r requirements.txt
```

## 2. Thông tin Dữ liệu & Random Seeds
- **Tập dữ liệu**: BreizhCrops (Vùng `frh01`, chuỗi thời gian Sentinel-2 L2A). Dữ liệu sẽ được thư viện tự động tải về (~2.5GB).
- **Data Hash/Phiên bản**: Dữ liệu tải từ máy chủ `tum.de` nội bộ của BreizhCrops, đảm bảo đồng nhất (Hash tĩnh theo từng tệp h5). Các script loader (`src/data/breizhcrops_loader.py`) và bộ chia split (`src/audit_split.py`) đã khóa cứng thứ tự mẫu.
- **Tập Split**: Train (N=2329), Val (N=499), Test (N=399). Không có mẫu nào bị rò rỉ giữa các tập. Cố định qua `random_state=42`.
- **Random Seeds (Huấn luyện)**: Nhằm triệt tiêu nhiễu ngẫu nhiên và đảm bảo so sánh công bằng bằng Bootstrapping, toàn bộ các mô hình DL được huấn luyện qua 10 seeds độc lập:
  `[42, 123, 2024, 7, 314, 0, 1, 999, 555, 77]`

## 3. Thời gian chạy ước tính (Runtime Estimates)
Trên môi trường máy tính cá nhân (1 GPU tầm trung / CPU đa nhân mạnh):
- **Phân tách Dữ liệu (Audit Split)**: ~1 phút.
- **Tuning (Optuna - 5 models x 20 trials)**: ~2-3 giờ (Tính toán trên tập Val).
- **Training 10 Seeds (6 DL models x 10 seeds)**: ~2 giờ (Tính toán trên GPU, thời gian inference trên CPU ~0.3ms/mẫu).
- **Tabular ML Baselines (RF/XGBoost tuned)**: ~15 phút.
- **Đánh giá Robustness (Zero-masking eval)**: ~5 phút.
*Tổng thời gian tái lập toàn bộ bài báo (End-to-End): Khoảng 5-6 tiếng.*

## 4. Chạy Tái lập từ A đến Z (Run from Scratch)
Chỉ cần chạy lệnh duy nhất sau đây trên môi trường sạch:
```bash
make all
```

Lệnh này sẽ tuần tự thực hiện:
1. `make prepare`: Cài đặt môi trường, tải dữ liệu, chia tập Train/Val/Test.
2. `make train`: Huấn luyện / Tuning các mô hình DL trên 10 seeds (Lưu vào `results/models_b0c/`).
3. `make preds`: Huấn luyện / Tuning các mô hình ML (Tabular Causal) và xuất toàn bộ dự đoán ra `results/preds/`.
4. `make tables`: Chạy thống kê (TOST, Bootstrap CI, Holm-Bonferroni) và phân tích Protocol Sensitivity.
5. `make figures`: Vẽ các đường cong Robustness, Calendar Gate Dynamics và xuất định dạng PDF / LaTeX sang thư mục `paper/figs/` và `paper/tables/`.

*(Lưu ý: Mọi bảng số liệu LaTeX trong báo cáo NCKH đều được xuất trực tiếp từ mã nguồn, không có bất kỳ sự can thiệp / sửa số thủ công nào nhằm đảm bảo tính toàn vẹn học thuật).*