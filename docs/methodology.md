---
title: methodology
type: note
permalink: timeseries-final-crop-phenology/docs/methodology
---

# Task 1 — Problem Formulation & Methodology Design

> Bối cảnh: tài liệu này nối tiếp `literature-review-crop-phenology-ndvi.md` và
> `nghien-cuu-nen-tang-crop-phenology-ndvi.md` (đã làm ở phiên trước) — **không** mở lại câu hỏi
> nguồn dữ liệu/vùng/cây trồng (vẫn đang mở, team tự quyết định), chỉ trả lời câu hỏi kiến trúc
> bài toán để đưa vào Deep Learning.

## 1.1. Quyết định khung bài toán: Segmentation (chính) + Regression suy ra (phụ)

**Khuyến nghị: KHÔNG chọn một trong hai — dùng lai (hybrid), vì lý do kỹ thuật + rubric:**

- **Point-wise segmentation (many-to-many sequence labeling)** là khung chính: mỗi timestep
  trong chuỗi NDVI một mùa vụ được gán 1 trong 4 nhãn giai đoạn sinh trưởng. Đây là lựa chọn
  **khai thác đúng sức mạnh của Bi-LSTM/GRU và Transformer** — cả hai kiến trúc này tự nhiên sinh
  ra 1 hidden state/attention output cho MỖI timestep, nên "many-to-many" tận dụng toàn bộ khả
  năng biểu diễn tuần tự của chúng. Ngược lại, nếu chỉ hồi quy 1 giá trị DOY duy nhất
  (many-to-one), ta chỉ dùng state cuối cùng (hoặc pooling) — bỏ phí phần lớn khả năng của kiến
  trúc tuần tự, và một MLP/RF với đặc trưng thống kê đơn giản cũng có thể làm tốt tương đương,
  khiến việc benchmark DL vs ML kém thuyết phục.
- **Suy ra mốc DOY (SOS/POS/EOS) từ nhãn dự đoán** làm metric phụ: tại vị trí chuyển nhãn
  (transition point) trong chuỗi nhãn dự đoán, tính ra ngày (Day-of-Year) tương ứng, so với
  ground-truth SOS/POS/EOS bằng MAE/RMSE/R² (tính bằng đơn vị **ngày**, không phải NDVI).
- Kết quả: báo cáo được **CẢ HAI** loại metric mà yêu cầu môn học liệt kê ("MAE/RMSE/R² cho
  regression HOẶC Accuracy/F1 cho segmentation") — vượt yêu cầu tối thiểu (điểm cộng NCKH), đồng
  thời map trực tiếp vào rubric CLO4 ("so sánh hiệu quả nhiều phương pháp, chọn phương pháp phù
  hợp"): so sánh không chỉ giữa model mà còn giữa 2 GÓC ĐO (classification quality vs. ngày thực
  tế lệch bao nhiêu — cái nông dân/nhà quản lý thực sự quan tâm).
- Neo lý thuyết: khung nhãn 4 lớp bên dưới ánh xạ trực tiếp vào 3 mốc SOS/POS/EOS đã định nghĩa
  trong `nghien-cuu-nen-tang-crop-phenology-ndvi.md` §2 (theo Zhang et al. 2003) — không phát
  minh khái niệm mới, chỉ rời rạc hoá (discretize) đường cong liên tục thành nhãn theo từng đoạn.

### Sơ đồ nhãn (4 lớp)

| Nhãn | Ý nghĩa nông học | Vị trí trên đường cong NDVI |
|---|---|---|
| 0 — `FALLOW` | Đất trống / ngoài vụ | Trước SOS hoặc sau EOS |
| 1 — `VEGETATIVE` | Nảy mầm → tăng trưởng | Từ SOS đến trước POS |
| 2 — `REPRODUCTIVE` | Sinh trưởng cực đại / trổ-ra hoa | Quanh POS (đỉnh đường cong) |
| 3 — `SENESCENCE` | Chín, giảm sinh khối | Từ sau POS đến EOS |

Suy ra DOY:
- `SOS_pred` = timestep đầu tiên nhãn chuyển từ 0→1
- `POS_pred` = timestep NDVI đạt cực đại trong vùng nhãn {1,2} (không chỉ dùng nhãn 2 vì lớp này
  có thể hẹp/mất cân bằng)
- `EOS_pred` = timestep đầu tiên nhãn chuyển từ 3→0

**Hạn chế cần công bố trong báo cáo** (đã ghi ở gap #1 trong literature review): SOS/POS/EOS suy
từ đường cong phổ là construct thống kê, không hoàn toàn trùng khớp mốc nông học/BBCH quan sát
thực địa — nếu không có ground truth thực địa, phải nêu rõ đây là "phenology theo tín hiệu vệ
tinh", không phải "phenology nông học tuyệt đối".

## 1.2. Input / Target shape chính xác cho từng nhóm model

Đơn vị dữ liệu cơ bản = **1 mùa vụ đã cắt cửa sổ cố định độ dài `T` timestep** (1 "season
instance"), không phải toàn bộ chuỗi nhiều năm liên tục — tránh rò rỉ dữ liệu giữa các mùa vụ khi
chia train/val/test (xem §1.3).

### Cho 1D-CNN / Bi-LSTM / Transformer (đầu vào chuỗi thô)

```
X: (batch, T, C)   # C >= 2: kênh 0 = NDVI đã làm mượt (Savitzky-Golay/Whittaker,
                   #         theo pipeline đã chốt ở nghien-cuu-nen-tang...md §4),
                   #         kênh 1-2 = sin(2*pi*DOY/365), cos(2*pi*DOY/365)
                   #         (mã hoá tuần hoàn ngày-trong-năm, KHÔNG dùng DOY thô
                   #         vì không liên tục ở ranh giới năm)
                   #         kênh tuỳ chọn thêm: NDVI thô (chưa mượt), EVI nếu có.
y: (batch, T)      # nhãn lớp {0,1,2,3} tại từng timestep — int64, dùng cho
                   # CrossEntropyLoss / F.cross_entropy(logits.transpose(1,2), y)

model output logits: (batch, T, num_classes=4)
```

Padding/mask: nếu độ dài mùa vụ không đều (thu hoạch sớm/muộn), pad về `T_max` và dùng
`attention_mask`/`lengths` để loại bỏ phần pad khỏi loss và metric (không đếm nhãn ở vùng pad).

### Cho Random Forest / XGBoost (baseline ML — bảng đặc trưng phẳng)

Không đưa chuỗi thô vào RF/XGBoost (chúng không hiểu thứ tự thời gian) — đúng tinh thần Chương 4
môn học ("Feature Engineering: lag features, rolling windows"): mỗi TIMESTEP trở thành **1 dòng**
trong bảng, đặc trưng xây từ lịch sử k bước trước đó của CHÍNH mùa vụ đó (không nhìn tương lai):

```
Một dòng ứng với timestep t của season s:
  lag_1 ... lag_k           : NDVI(t-1) .. NDVI(t-k)
  rolling_mean_w, rolling_std_w  : thống kê cửa sổ trượt độ rộng w (vd 3, 5)
  doy_sin_t, doy_cos_t       : mã hoá ngày trong năm tại t
  delta_ndvi_1               : NDVI(t) - NDVI(t-1) (đạo hàm rời rạc bậc 1 — tín hiệu quan trọng
                                cho việc bắt điểm uốn SOS/POS/EOS, xem §3.2 literature review)
target: label(t) in {0,1,2,3}
```
=> X_tabular: (n_rows, n_features), y_tabular: (n_rows,) — **cùng một tập nhãn/timestep** với DL
models, chỉ khác cách biểu diễn đầu vào (bảng phẳng có feature engineering thủ công vs. chuỗi thô
để model tự học biểu diễn) — đây chính là điểm so sánh cốt lõi mà đồ án cần làm nổi bật.

## 1.3. Chia train/val/test — tránh rò rỉ dữ liệu

Chia theo **đơn vị season instance** (hoặc theo thửa ruộng/năm nếu dùng nhiều năm/nhiều vùng),
**không** chia ngẫu nhiên theo từng timestep — nếu không, các timestep liền kề của cùng 1 mùa vụ
sẽ lọt vào cả train và test, gây leakage nghiêm trọng (2 timestep cạnh nhau gần như giống hệt
nhau). Dùng `GroupKFold` (group = season_id) hoặc `TimeSeriesSplit` theo năm (train các năm sớm,
test năm sau) — đúng yêu cầu môn học (Chương 4: "kỹ thuật kiểm thử chéo tránh rò rỉ dữ liệu trong
chuỗi thời gian").

## 1.4. Bộ metric chuẩn cho mọi model (bắt buộc thống nhất giữa 5 thành viên)

| Nhóm | Metric | Ghi chú |
|---|---|---|
| Segmentation quality | Accuracy, macro-F1, per-class F1 | macro-F1 vì lớp `REPRODUCTIVE` thường hiếm/hẹp (mất cân bằng, giống bài toán ViHSD đã gặp ở project khác — ưu tiên macro không phải accuracy) |
| Phenology-date accuracy (suy ra) | MAE, RMSE, R² (đơn vị: **ngày**) cho từng mốc SOS/POS/EOS riêng biệt | So sánh trực tiếp với con số ~9 ngày (TIMESAT/phenex) và ~10 ngày (LSTM) đã tìm thấy trong literature review — framing "so với SOTA thế giới" |
| Compute cost | Training time (giây, wall-clock, cùng 1 GPU/CPU) | Đo bằng `evaluation/metrics.py::Timer`, cùng số epoch/early-stopping criterion cho mọi model để công bằng |
| Compute cost | Inference time (ms/season, đo trên tập test, batch=1 để mô phỏng inference thực tế) | |

Toàn bộ 4 nhóm này được implement 1 lần trong `src/evaluation/metrics.py`, dùng chung cho cả 5
thành viên — không ai tự viết lại cách tính (tránh 5 con số không so sánh được với nhau).