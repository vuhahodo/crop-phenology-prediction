---
title: image_data_extension_decision
type: note
permalink: timeseries-final-crop-phenology/docs/image-data-extension-decision
---

# Quyết định đang mở: có nên chuyển sang dữ liệu ảnh không?

> File này ghi lại định hướng đang bàn (2026-09-13), **CHƯA LÀM GÌ** — đang chờ user hỏi ý kiến
> giảng viên rồi quay lại chọn hướng. Đọc file này trước khi động vào bất cứ thứ gì liên quan đến
> "dữ liệu ảnh"/"PASTIS"/"Google Earth Engine" trong project này.

## Bối cảnh

Hiện tại project dùng **BreizhCrops** (NDVI trung bình theo từng thửa ruộng, KHÔNG phải ảnh không
gian — xem `src/data/breizhcrops_loader.py`) cho cả 6 model (RF/XGBoost/1D-CNN/Bi-LSTM/Transformer/
CBA-PhenoNet), kết quả thật đã có trong `results/comparison_table.csv` (macro-F1 tốt nhất 0.903,
CBA-PhenoNet). Xem `README.md` để có bảng số liệu đầy đủ.

Giảng viên được cho là **không đánh giá cao dữ liệu "dạng bảng"** (tabular) vì đơn giản/nhanh.
BreizhCrops mà project đang dùng, dù xuất phát từ ảnh vệ tinh Sentinel-2, đã bị đóng gói sẵn thành
giá trị trung bình mỗi band/thời điểm mỗi thửa ruộng — về cấu trúc dữ liệu đưa vào model, đây gần
như là 1 chuỗi số theo thời gian (không có chiều không gian H×W), nên có thể bị đánh giá tương tự
"dữ liệu dạng bảng" dù bản chất là chuỗi thời gian.

**Việc đang chờ**: user sẽ nhắn tin hỏi giảng viên xem cách tiếp cận hiện tại (NDVI parcel-level,
đúng trọng tâm môn Time Series Data Analysis — xem `docs/methodology.md`) có được chấp nhận
không, hay cần phát triển thêm hướng ảnh để "khó hơn"/phù hợp hơn. **Không tự ý làm gì thêm về
hướng ảnh cho tới khi có câu trả lời này.**

## 3 hướng đã cân nhắc nếu giảng viên yêu cầu chuyển sang ảnh thật

### 1. Ảnh biểu đồ NDVI (không phải ảnh vệ tinh thật) + Claude vision — rẻ nhất, không rủi ro
Vẽ đường cong NDVI (matplotlib) thành ảnh, cho Claude (model có vision) đọc ảnh để trích đặc
trưng thị giác (hình dạng đường cong, độ mơ hồ tại điểm chuyển pha...), ghép vào model bằng
stacking ensemble — đúng pattern đã dùng THÀNH CÔNG ở project NLP khác cùng phiên
(`Final/scripts/build_ensemble.py` trong repo CV/NLP, macro-F1 tăng 0.706→0.735 nhờ thêm feature
từ Claude). Không cần tải gì thêm, không rủi ro hết đĩa, chạy được ngay.
**Hạn chế phải nói thật**: đây KHÔNG PHẢI ảnh vệ tinh không gian thật, chỉ là ảnh minh hoạ đồ thị.

### 2. Ảnh vệ tinh thật qua Google Earth Engine — thật nhất về mặt "ảnh", nhưng cần user tự làm 1 bước
Kéo riêng từng patch Sentinel-2 nhỏ (vd 128×128 pixel) cho từng thửa ruộng qua GEE API thay vì
tải nguyên khối dữ liệu lớn — nhẹ đĩa (chỉ tải đúng patch cần). **Nhưng cần user tự đăng nhập
Google/OAuth** trước — đây là bước tương tác không tự động hoá được (giống việc user từng phải tự
tạo lại Kaggle token).

### 3. Tải nguyên PASTIS (ảnh vệ tinh thật, đã đóng gói sẵn) — nặng nhất, rủi ro nhất
PASTIS (Zenodo record 5012942): 2433 patch 128×128 pixel, 10 band Sentinel-2, 38-61 timestep/mùa
vụ, có nhãn parcel — bản S2-only ~28.8GB, bản S1+S2 ~54GB.
**Ràng buộc đĩa đã kiểm tra thật (2026-09-13)**: lúc bàn hướng này ổ D: chỉ còn 18GB trống, sau đó
user dọn thêm còn ~30GB — vẫn CHỈ đủ margin mỏng cho bản 28.8GB (S2-only), KHÔNG đủ cho bản 54GB.
Cần train model 3D-CNN/ConvLSTM thật trên pixel — độ phức tạp/thời gian tăng nhiều so với mọi
model đã build trong project này (nhiều khả năng cần GPU Kaggle, không chạy CPU local như hiện
tại được nữa).

## Khuyến nghị (nếu giảng viên yêu cầu làm khó hơn)
Ưu tiên hướng 1 trước (rẻ, nhanh, tận dụng pattern đã proven ở project NLP) — chỉ làm hướng 2/3
nếu giảng viên cụ thể yêu cầu ảnh vệ tinh không gian thật, vì chi phí/rủi ro cao hơn hẳn.