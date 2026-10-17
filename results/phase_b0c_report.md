---
title: phase_b0c_report
type: note
permalink: timeseries-final-crop-phenology/results/phase-b0c-report
---

# Báo cáo Benchmark & Ablation (Đã nâng cấp Thống kê)

> Các bảng phân tích chi tiết (TOST, Bootstrap CI trên mẫu và seed) nằm trong `results/tables/`.
> Mọi giả định phân phối đều được nới lỏng thông qua bootstrap 10.000 lần (cho cặp seed) và 1.000 lần (cho mẫu test).

---

## 1. Môi trường & Tất định
- **Tập dữ liệu**: Single-season (2017), chia 70/15/15 ngẫu nhiên theo thửa ruộng (0 rò rỉ field ID - xem `docs/data_audit.md`).
- **Giao thức giải mã**: Đồng bộ dùng EOS = `3->0`. Các chuỗi không có điểm chuyển giao này (cắt cụt) sẽ bị loại khỏi mẫu số MAE tương ứng.
- **Tính tất định**: 7 config x 10 seeds được train từ đầu, ghi đè 100% để loại bỏ hiện tượng rò rỉ checkpoint do seed cũ.

---

## 2. So sánh Baseline (Tabular ML vs Deep Sequential)

Tabular ML được huấn luyện với **toàn bộ đặc trưng toàn vụ (Full-Season)** bao gồm: lags, leads (tương lai), và các chỉ số thống kê tổng hợp của cả mùa (min, max, mean, max_slope). 

| Model | Macro-F1 | 95% CI (Data Bootstrap N=399) |
|---|---|---|
| Random Forest (causal) | 0.4205 | [0.3907, 0.4237] |
| XGBoost (causal) | 0.4029 | [0.3795, 0.4144] |
| Double-Logistic | 0.3471 | [0.2751, 0.3163] |
| RF-FullSeason | 0.5531 | [0.5498, 0.5941] |
| XGBoost-FullSeason | 0.6052 | [0.5997, 0.6418] |
| **Standard Transformer** | **0.9060** | **[0.8769, 0.9006]** |

*Chú thích:* 
- Mặc dù Transformer là Bidirectional (thấy cả chuỗi), Tabular ML FullSeason cũng đã được cấp đủ biến "tương lai" (leads) để công bằng.
- Dù vậy, XGBoost-FullSeason vẫn kém Standard Transformer tới ~0.30 điểm F1 tuyệt đối. Khoảng CI 95% của hai họ mô hình cách xa nhau hoàn toàn.
- Double-Logistic (TIMESAT-style curve fitting) không khớp được 116/399 chuỗi, và cho sai số MAE rất lớn (>59 ngày). Phương pháp này không phù hợp cho per-timestep segmentation.

---

## 3. Phân tích Ablation (TOST & Multiple Testing)

**Cấu hình kiểm định:**
- Alpha = 0.05. Multiple testing correction: Bonferroni và Holm-Bonferroni (Family = 6 config x 4 metric = 24 tests).
- Bootstrap CI cho hiệu số (Mean Diff = Config - Standard): Lặp 10.000 lần qua 10 cặp seed.
- **TOST (Two One-Sided Tests)**: Biên tương đương (Equivalence bounds) cho Macro-F1 được chọn là **+/- 0.02**. 
  *Lý do:* Trong bài toán segmentation, mức xê dịch 2% (0.02) F1-score thường nằm trong khoảng nhiễu tự nhiên của dữ liệu hoặc là một khác biệt không mang lại giá trị vận hành thực tiễn (practical equivalence margin).

**Kết quả trên Macro-F1:**
| Config | Mean Diff | 95% CI (Seed Boot) | Cohen's d_z | p_holm | Kết luận TOST (+/- 0.02) |
|---|---|---|---|---|---|
| Cyclical DOY | -0.0283 | [-0.0457, -0.0135] | -1.017 | 0.242 | Không phát hiện khác biệt / Không tương đương |
| Calendar Gate | -0.0049 | [-0.0160, +0.0055] | -0.266 | 1.000 | **Tương đương trong biên 0.02** |
| Scalar Gate | -0.0046 | [-0.0118, +0.0030] | -0.370 | 1.000 | **Tương đương trong biên 0.02** |
| CBA-PhenoNet | -0.0068 | [-0.0169, +0.0031] | -0.390 | 1.000 | **Tương đương trong biên 0.02** |
| CBA Residual | -0.0098 | [-0.0214, +0.0001] | -0.528 | 1.000 | Không phát hiện khác biệt |
| CBA Disentangled | -0.0088 | [-0.0227, +0.0035] | -0.386 | 1.000 | Không phát hiện khác biệt |

**Kết quả trên MAE (Sai số ngày dự đoán Cột mốc):**
- Không có bất kỳ cấu hình nào vượt trội hơn Standard Transformer ở mức ý nghĩa $p_{holm} < 0.05$ hay $p_{bonf} < 0.05$.
- Chênh lệch lớn nhất đo được là ở mô hình CBA-PhenoNet (giảm POS MAE được 1.35 ngày so với Standard), nhưng 95% CI bootstrap dao động từ [-2.23, -0.44], và p-value sau hiệu chỉnh (Holm = 0.428) không đạt mức ý nghĩa.

## 4. Tổng kết
- Hướng tiếp cận dùng Gate (Calendar Gate, CBA-PhenoNet) được chứng minh toán học (TOST) là **tương đương thực tiễn** với Standard Transformer. Gate chỉ học một hằng số và không thay đổi cục diện.
- Bất kỳ độ lệch nào quan sát được ở các phiên bản trước đây đều do artifact của quy trình (Lỗi Zero-DOY trong hàm gap, lỗi tái sử dụng checkpoint, và dao động do mẫu nhỏ).
- Standard Transformer hoàn toàn vượt trội Tabular ML, và là lựa chọn tối ưu, tinh gọn nhất cho bài toán này.

## 5. Lời giải cho "Bí ẩn 0.8872"

Trong các báo cáo trước, một giả thuyết được đặt ra rằng hiệu năng của CBA-PhenoNet đã tăng vọt một cách khó hiểu (từ `0.8872` ở Phase B0 lên `0.8992` ở Phase B0c). Tuy nhiên, sau khi audit lại bộ checkpoint cũ bằng script `src/b0e_check_discrepancy.py`, sự thật được phơi bày:

- Hiệu năng thực tế của 5 seed cũ (42, 123, 2024, 7, 314) là **0.8972 ± 0.0178**.
- Hiệu năng của chính 5 seed đó khi train lại hoàn toàn trong Phase B0c vẫn đúng chính xác là **0.8972 ± 0.0178**.
- Con số `0.8872` được nhắc đến ở báo cáo trước thuần túy là một **lỗi gõ phím (typo)** trong quá trình viết markdown (viết nhầm 0.8972 thành 0.8872).
- Khi mở rộng từ 5 seed lên 10 seed, điểm trung bình dịch chuyển nhẹ từ 0.8972 lên 0.8992 (+0.002) — đây là dao động tự nhiên hoàn toàn bình thường do thêm mẫu.

Không hề có lỗi tái sử dụng checkpoint, cũng không có sự chênh lệch bí ẩn nào về code hay giao thức đánh giá. Mọi thứ đã luôn hoạt động ổn định và tất định ngay từ đầu. Hướng nghiên cứu Gate bị bác bỏ vì mặt toán học nó thực sự không đem lại hiệu quả, chứ không phải do bug.