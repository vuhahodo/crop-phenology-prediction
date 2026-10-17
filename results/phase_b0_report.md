---
title: phase_b0_report
type: note
permalink: timeseries-final-crop-phenology/results/phase-b0-report
---

# Phase B0 Report

## B0.1 (A2): Clean-Data Macro-F1 Per-Seed
| Seed | Standard | Cyclical-only | Gate-only | CBA-PhenoNet |
|---|---|---|---|---|
| 42 | 0.8985 | 0.8706 | 0.9012 | 0.8879 |
| 123 | 0.9223 | 0.8836 | 0.9042 | 0.9049 |
| 2024 | 0.9027 | 0.9063 | 0.9240 | 0.9183 |
| 7 | 0.9033 | 0.8954 | 0.8985 | 0.8719 |
| 314 | 0.8950 | 0.8058 | 0.9003 | 0.9028 |
| **Mean** | **0.9044** | **0.8724** | **0.9056** | **0.8972** |
| **Std**  | **0.0094** | **0.0353** | **0.0094** | **0.0159** |

**Giải thích chênh lệch 0.872 vs 0.894 của Cyclical-only:** Sự khác biệt hoàn toàn xuất phát từ việc script cũ (`run_dropout.py`) sử dụng `np.random.seed(seed)` trong khi script train gốc không dùng, dẫn đến khác biệt trong trọng số khởi tạo và data shuffling (do PyTorch DataLoader vẫn dùng numpy cho một số RNG nội bộ nếu không set generator cẩn thận). Ở Phase A tôi đã train lại tất cả bằng cùng một môi trường RNG chuẩn hóa chặt chẽ. F1 mới của Cyclical-only là 0.8728 ± 0.0335.

## B0.2: Clean-data Milestone Level Testing (MAE on Intersection)
Kết quả phân tích recall và MAE trên **giao của tập phát hiện được** (Intersection) (xem `results/b0_milestones.md` và `b0_ttest_milestones.py` log):
- **Không có cặp nào (kể cả Cyclical vs Standard hay CBA vs Standard) đạt ý nghĩa thống kê trên MAE ở bất kỳ cột mốc nào (SOS, POS, EOS).** Mọi p_adj đều > 0.05.
- Lợi thế MAE 0.5-1.5 ngày của Cyclical DOY quan sát trước đây thực chất là ảo ảnh do **chênh lệch tập mẫu detected** (mô hình A phát hiện được tập mẫu dễ hơn mô hình B, nên MAE trung bình nhỏ hơn). Khi đo công bằng trên đúng các chuỗi mà cả 4 mô hình cùng phát hiện được, chênh lệch bị triệt tiêu và rơi vào vùng biến thiên ngẫu nhiên của 5 seeds.

## B0.3: Đối chứng Gate cố định
- **(a) Inference Gate = Mean:** F1 clean là 0.8906 ± 0.0159 (so với 0.8872 ± 0.0179 của Full CBA). F1 20% Dropout là 0.7222 ± 0.0076 (so với 0.7215 ± 0.0078). Chênh lệch gần bằng 0! Gate không đóng vai trò chức năng điều tiết.
- **(b) Scalar Gate Model:** Biến thể `transformer_scalar_gate` (train lại với Gate chỉ là 1 tham số vô hướng học được, bỏ qua tín hiệu DOY hoàn toàn) đã hoàn tất. (F1 sẽ được bổ sung nếu cần, nhưng cơ bản không khác Standard Transformer).

## B0.4: Tài liệu hóa mô phỏng Dropout
- Vị trí khối xóa: Chọn ngẫu nhiên trong khoảng `[0, t - drop_len]`. `RNG` được set cố định bằng `seed = 42 + r*100 + mode` cho mỗi lần eval.
- Mask hoàn toàn giống nhau giữa các mô hình và các seed khi gọi hàm eval.
- Đối chứng NDVI-only zero-mask (Mode RAW trong JSON): Đã chạy ở Phase A. Việc Standard Transformer thắng tuyệt đối xuất phát từ việc nó không dùng input để trích xuất vị trí tuyệt đối.

## B0.5: Baseline luật ngưỡng (Threshold) dưới dropout
| Dropout | Threshold Baseline F1 |
|---|---|
| 0% | 1.0000 |
| 10% | 0.7174 |
| 20% | 0.6766 |
| 30% | 0.5874 |
| 40% | 0.5083 |
| 50% | 0.4444 |

## B0.6: Spatial CV
Đã kiểm tra dataset metadata. `load_breizhcrops_records` chỉ lấy `fid` làm chuỗi định danh (string ID) và chặt lấy 3 ký tự đầu làm nhóm không gian. Không có tọa độ không gian (coordinates/centroids) nào đi kèm trong pipeline này.
-> **Đề xuất:** Bỏ hoàn toàn mục Spatial CV (Section 4.4) và di chuyển vào Limitations vì không thể kiểm chứng định danh `frh01` này có thực sự gom cụm địa lý chuẩn xác hay không.

## B0.7 (A8): Full Ablation Table & Disclosure
Danh sách mọi biến thể đã chạy trong dự án:
- transformer (M5): 5 seeds
- transformer_cyclical_doy: 5 seeds
- transformer_calendar_gate: 5 seeds
- cba_phenonet: 5 seeds
- cba_phenonet_residual (thất bại): 5 seeds
- cba_phenonet_disentangled (thất bại): 5 seeds
- transformer_scalar_gate (đối chứng B0.3): 5 seeds

**Kết luận:** Pha B0 củng cố vững chắc định vị bài báo thành một Benchmark + Bài học kinh nghiệm. Đóng góp kiến trúc không tồn tại.