---
title: nghien-cuu-nen-tang-crop-phenology-ndvi
type: note
permalink: timeseries-final-crop-phenology/nghien-cuu-nen-tang-crop-phenology-ndvi
---

# Nghiên cứu nền tảng: Theo dõi vòng đời cây trồng (Crop Phenology) qua chuỗi thời gian NDVI

> Thực hiện theo phương pháp luận của skill `deep-research` (chế độ `lit-review`) trong bộ `academic-research-skills` — áp dụng thủ công quy trình Đặt câu hỏi nghiên cứu → Tìm kiếm hệ thống → Phân loại/chấm điểm nguồn → Tổng hợp chủ đề → Annotated bibliography. Xem mục "Hạn chế & công bố AI" ở cuối tài liệu.

## 1. Câu hỏi nghiên cứu & phạm vi

**RQ chính:** Các mốc thời gian sinh trưởng chủ đạo của cây trồng — **gieo sạ** (sowing/emergence), **phát triển − ra hoa** (green-up → peak growth), và **thu hoạch** (senescence/harvest) — có thể được xác định định lượng, khách quan và có thể lặp lại như thế nào từ chuỗi thời gian chỉ số thực vật NDVI thu từ ảnh vệ tinh quang học?

**Sub-questions:**
1. Đường cong NDVI theo mùa vụ phản ánh sinh lý cây trồng ra sao, và các "phenological metrics" chuẩn (SOS/POS/EOS) được định nghĩa thế nào trên đường cong đó?
2. Các họ thuật toán trích xuất mốc thời gian từ chuỗi NDVI gồm những gì, ưu/nhược điểm ra sao?
3. Nhiễu (mây, khí dung, góc chụp, dữ liệu khuyết) ảnh hưởng thế nào, và được xử lý bằng những kỹ thuật time series nào?
4. Bằng chứng thực nghiệm trên cây lúa — đặc biệt Đồng bằng sông Cửu Long (ĐBSCL) — nói gì về độ chính xác?
5. Machine learning / deep learning đóng vai trò gì so với phương pháp cổ điển, và điều này liên hệ thế nào với nội dung học phần Phân tích dữ liệu chuỗi thời gian (AI2029)?

**Ngoài phạm vi:** thiết kế chi tiết mô hình dự báo cụ thể cho đồ án (sẽ làm ở bước tiếp theo, sau khi có nền tảng lý thuyết này); dữ liệu radar (Sentinel-1/SAR) chỉ được nhắc đến để đối chiếu, không đi sâu.

---

## 2. Cơ sở khoa học: đường cong NDVI theo mùa vụ

NDVI = (NIR − Red)/(NIR + Red) phản ánh mật độ diệp lục/sinh khối xanh. Với một vụ mùa, NDVI vẽ nên một đường cong hình chuông bất đối xứng theo thời gian: thấp khi đất trống → tăng khi cây nảy mầm và phát triển → đạt đỉnh khi sinh khối/diện tích lá cực đại (thường quanh giai đoạn trổ/ra hoa) → giảm khi cây chín và được thu hoạch, quay về mức nền.

Bộ "phenological metrics" chuẩn được dùng phổ biến nhất trong tài liệu ([USGS EROS — Deriving Phenological Metrics from NDVI](https://www.usgs.gov/core-science-systems/eros/phenology/science/deriving-phenological-metrics-ndvi); Zhang et al., 2003):

| Mốc | Ký hiệu | Ý nghĩa nông học | Ứng với |
|---|---|---|---|
| Start of Season | SOS | NDVI bắt đầu tăng khỏi nền đất trống | Gieo sạ/nảy mầm |
| Peak of Season | POS | NDVI đạt cực đại | Sinh trưởng cực đại/trổ bông |
| End of Season | EOS | NDVI giảm về mức nền | Chín/thu hoạch |

---

## 3. Bốn họ phương pháp trích xuất mốc thời gian (tổng hợp chủ đề)

### 3.1 Ngưỡng biên độ (threshold-based)
SOS/EOS được định nghĩa là thời điểm NDVI vượt qua (hoặc giảm dưới) một tỉ lệ phần trăm cố định của biên độ tăng/giảm theo mùa (thường 10–20%) (Jönsson & Eklundh, 2004). Đơn giản, minh bạch, dễ diễn giải, nhưng nhạy với nhiễu dư sau làm mượt và với việc chọn ngưỡng. Đây là phương pháp được áp dụng thực tế cho lúa ở ĐBSCL: ngưỡng NDVI tăng vượt 20% biên độ mùa vụ, sau khi làm mượt bằng bộ lọc Savitzky–Golay, được dùng để xác định ngày gieo sạ.

### 3.2 Khớp hàm số (function-fitting / curve-based)
Chuỗi NDVI thô được khớp bằng một hàm số mượt: double-logistic (nền tảng của phần mềm **TIMESAT** — Jönsson & Eklundh, 2002, 2004), hàm Gauss bất đối xứng, phân tích Fourier/**HANTS** (Roerink, Menenti & Verhoef, 2000), bộ lọc **Savitzky–Golay** (Chen et al., 2004), hay **Whittaker smoother** (Atzberger & Eilers, 2011). Mốc pha vật hậu khi đó là các điểm uốn (inflection points) hoặc cực trị đạo hàm của đường cong đã khớp (Zhang et al., 2003) — một dạng khớp mô hình phi tuyến, xử lý tương tự bài toán tách xu hướng/mùa vụ bằng phân rã cổ điển nhưng ở cấp độ hàm liên tục thay vì tổng cộng gộp (additive decomposition).

So sánh nhiều thuật toán (CropPhenology, phenex, TIMESAT, phenofit...) trên cùng bộ dữ liệu cho thấy: TIMESAT dự đoán ngày gieo sạ tốt hơn, trong khi phenex dự đoán ngày thu hoạch tốt hơn; sai số RMSE ghi nhận khoảng 9 ngày so với khảo sát thực địa ([Estimating Crop Sowing and Harvesting Dates Using Satellite Vegetation Index: A Comparative Analysis](https://doi.org/10.3390/rs15225366), 2023).

### 3.3 Phân rã & phát hiện breakpoint (decomposition-based changepoint)
**BFAST** (Breaks For Additive Season and Trend; Verbesselt, Hyndman, Newnham & Culvenor, 2010) phân rã chuỗi NDVI thành **xu hướng + mùa vụ + phần dư** — cùng cấu trúc với phân rã chuỗi thời gian cổ điển (Chương 2 học phần) — rồi dò breakpoint bằng kiểm định thay đổi cấu trúc trên từng thành phần. Breakpoint trong thành phần **mùa vụ** thường tương ứng với **thay đổi vật hậu** (lệch lịch gieo sạ, đổi giống...), còn breakpoint trong **xu hướng** tương ứng với nhiễu loạn phi mùa vụ (cháy, sâu bệnh, chuyển đổi đất) ([Verbesselt et al., 2010](https://robjhyndman.com/papers/bfast1.pdf)).

Đáng chú ý: một trong các đồng tác giả BFAST là **Rob J. Hyndman** — chính tác giả giáo trình chính *Forecasting: Principles and Practice* của học phần AI2029 — cho thấy một cầu nối trực tiếp giữa lý thuyết phân rã/kiểm định breakpoint đã học và bài toán crop phenology.

### 3.4 Machine learning / Deep learning
Các nghiên cứu gần đây (2022–2025) dùng **LSTM** để dự báo NDVI đến hết vụ rồi suy ra mốc pha vật hậu từ chuỗi dự báo, đạt RMSE≈0.23 (NDVI chuẩn hóa) và sai lệch mốc thời gian trung bình khoảng ±10 ngày ([Crop Phenology Stage Forecasting and Detection using NDVI Time-Series and LSTM](https://www.researchgate.net/publication/362847466_Crop_Phenology_Stage_Forecasting_and_Detection_using_NDVI_Time-Series_and_LSTM), IEEE 2022). Kiến trúc mới hơn như **ConvLSTM** kết hợp đặc trưng không gian (CNN) với mô hình hoá thời gian (LSTM) cho phân loại giai đoạn vật hậu quy mô vùng (2025).

Quan trọng: ML/DL **không thay thế** mà **bổ sung** cho các phương pháp cổ điển — vẫn cần chuỗi NDVI đã làm sạch (tiền xử lý Chương 1) làm đầu vào, và thường kết hợp đặc trưng trễ/cửa sổ trượt (Chương 4) hoặc mạng tuần tự (Chương 5) để dự báo/phân loại pha.

---

## 4. Tiền xử lý dữ liệu NDVI — ánh xạ trực tiếp với nội dung học phần

Nguồn nhiễu chính: mây/bóng mây, khí dung khí quyển, hiệu ứng góc quan trắc (BRDF), và dữ liệu khuyết theo chu kỳ composite (ví dụ 8/16 ngày của MODIS). Các kỹ thuật xử lý ánh xạ gần như 1-1 với Chương 1 của học phần:

| Vấn đề dữ liệu | Kỹ thuật xử lý | Tương ứng Chương 1 (AI2029) |
|---|---|---|
| Dữ liệu khuyết do mây | HANTS (Fourier), nội suy tuyến tính/spline | Xử lý missing values |
| Nhiễu dư sau lọc mây | Savitzky–Golay, Whittaker smoother, double-logistic | Làm mịn (smoothing) |
| Phương sai không ổn định giữa mùa | Chuẩn hoá biên độ theo ngưỡng phần trăm | Scaling/chuẩn hoá |

Không có phương pháp nào thắng tuyệt đối: Savitzky–Golay hiệu quả khi thiếu dữ liệu dài hạn nhưng có thể không mô phỏng đúng hình dạng sinh học thật của đường cong trong giai đoạn tăng trưởng nhanh; Whittaker linh hoạt hơn về không gian nhưng nhạy với việc chọn tham số làm mượt (λ) ([Performance of Smoothing Methods for Reconstructing NDVI Time-Series and Estimating Vegetation Phenology from MODIS Data](https://www.mdpi.com/2072-4292/9/12/1271), 2017).

---

## 5. Case study: cây lúa, đặc biệt Đồng bằng sông Cửu Long (ĐBSCL)

- **PhenoRice** (Boschetti et al., 2017): thuật toán rule-based trên chuỗi MODIS, dùng đồng thời NDFI và EVI để phát hiện đặc trưng "ngập nước nông nghiệp" (flooding) — dấu hiệu đặc trưng của canh tác lúa nước — từ đó suy ra SOS (gieo sạ/cấy), POS (trổ bông) và EOS (thu hoạch) một cách tự động trên diện rộng.
- Nghiên cứu tại ĐBSCL dùng Savitzky–Golay + ngưỡng 20% biên độ tăng để xác định ngày gieo sạ, ghi nhận **lịch gieo sạ dịch chuyển tới ~20 ngày** do xâm nhập mặn và hạn hán — cho thấy chuỗi NDVI có thể dùng để phát hiện sự thích ứng canh tác trước biến đổi khí hậu, không chỉ đo lịch mùa vụ "chuẩn".
- Đối chiếu với thống kê chính thức, các đợt gieo sạ và diện tích canh tác ước tính từ viễn thám có sai số dưới 12%.
- Các nghiên cứu song song dùng Sentinel-1 (SAR, không phụ thuộc mây) để lập bản đồ cường độ mùa vụ và ngày gieo sạ ở quy mô 10m trên toàn ĐBSCL (2016–2024) — hướng bổ sung khi NDVI quang học bị gián đoạn nặng bởi mây trong mùa mưa.

---

## 6. Khoảng trống & gợi ý pipeline khả thi cho đồ án môn học

Đề xuất một pipeline dùng đúng bộ công cụ đã/đang học trong AI2029, đi từ dữ liệu NDVI thô đến trích xuất mốc phenology:

1. **Thu thập & tiền xử lý (Chương 1):** lấy chuỗi NDVI (Sentinel-2/MODIS, qua Google Earth Engine hoặc dữ liệu công khai có sẵn) cho một hoặc vài thửa/vùng canh tác; xử lý missing values, làm mượt (Savitzky–Golay hoặc Whittaker).
2. **Phân rã (Chương 2):** áp dụng phân rã cổ điển (STL-style) để tách trend–seasonal–residual; so sánh với kết quả khớp double-logistic.
3. **Trích mốc pha (SOS/POS/EOS):** kết hợp ngưỡng biên độ (đơn giản, baseline) với phát hiện breakpoint kiểu BFAST trên thành phần mùa vụ (liên hệ kiểm định tính dừng/breakpoint ở Chương 3 nếu muốn mở rộng).
4. **Dự báo (Chương 4–5):** xây đặc trưng trễ/cửa sổ trượt (Random Forest/XGBoost) hoặc LSTM/GRU để dự báo NDVI k bước tới, từ đó cảnh báo sớm mốc thu hoạch dự kiến.
5. **Đánh giá (CLO4):** so sánh RMSE/MAPE giữa mốc dự đoán và mốc thực tế (lịch nông vụ địa phương/ground truth nếu có), dùng TimeSeriesSplit để validation, đúng theo rubric đánh giá của học phần.

**Khoảng trống đáng lưu ý trong tài liệu đã khảo sát:** phần lớn nghiên cứu định lượng độ chính xác (RMSE ~9 ngày, sai lệch ±10 ngày) dùng ground truth từ khảo sát thực địa hoặc thống kê nông nghiệp — một hạn chế chung là thiếu ground truth ở cấp thửa ruộng tại nhiều vùng, kể cả Việt Nam; đây cũng là hạn chế cần nêu rõ nếu đồ án không có dữ liệu ground truth đối chiếu.

---

## 7. Bảng tài liệu tham khảo đã chú thích (Annotated Bibliography)

| # | Tài liệu | Đóng góp chính | Bậc bằng chứng |
|---|---|---|---|
| 1 | Jönsson, P., & Eklundh, L. (2002). Seasonality extraction by function fitting to time-series of satellite sensor data. *IEEE TGRS*. / (2004). TIMESAT — a program for analyzing time-series of satellite sensor data. *Computers & Geosciences*. | Nền tảng double-logistic + phần mềm TIMESAT | Peer-reviewed, phương pháp gốc |
| 2 | Zhang, X. et al. (2003). Monitoring vegetation phenology using MODIS. *Remote Sensing of Environment*. | Định nghĩa SOS/POS/EOS bằng điểm uốn đạo hàm | Peer-reviewed, phương pháp gốc |
| 3 | Roerink, G. J., Menenti, M., & Verhoef, W. (2000). Reconstructing cloudfree NDVI composites using Fourier analysis of time series. *Int. J. Remote Sensing*. | HANTS gap-filling | Peer-reviewed, phương pháp gốc |
| 4 | Chen, J. et al. (2004). A simple method for reconstructing a high-quality NDVI time-series data set based on the Savitzky–Golay filter. *Remote Sensing of Environment*. | Bộ lọc SG cho NDVI | Peer-reviewed, phương pháp gốc |
| 5 | Atzberger, C., & Eilers, P. H. C. (2011). A time series for monitoring vegetation activity and phenology at 10-daily time steps. *Int. J. Digital Earth*. | Whittaker smoother cho NDVI | Peer-reviewed, phương pháp gốc |
| 6 | Verbesselt, J., Hyndman, R., Newnham, G., & Culvenor, D. (2010). Detecting trend and seasonal changes in satellite image time series. *Remote Sensing of Environment* ([PDF](https://robjhyndman.com/papers/bfast1.pdf)). | Thuật toán BFAST | Peer-reviewed, phương pháp gốc — đồng tác giả là Rob J. Hyndman |
| 7 | Boschetti, M. et al. (2017). PhenoRice: A method for automatic extraction of spatio-temporal information on rice crops using satellite data time series. *Remote Sensing of Environment*. | Thuật toán chuyên biệt cho lúa, dùng flooding signal | Peer-reviewed |
| 8 | [Estimating Crop Sowing and Harvesting Dates Using Satellite Vegetation Index: A Comparative Analysis](https://doi.org/10.3390/rs15225366) (2023). *Remote Sensing* (MDPI). | So sánh TIMESAT/phenex/CropPhenology, RMSE ~9 ngày | Peer-reviewed |
| 9 | [Crop Phenology Stage Forecasting and Detection using NDVI Time-Series and LSTM](https://www.researchgate.net/publication/362847466_Crop_Phenology_Stage_Forecasting_and_Detection_using_NDVI_Time-Series_and_LSTM). IEEE (2022). | LSTM dự báo NDVI → suy mốc pha | Peer-reviewed (hội nghị) |
| 10 | [Performance of Smoothing Methods for Reconstructing NDVI Time-Series and Estimating Vegetation Phenology from MODIS Data](https://www.mdpi.com/2072-4292/9/12/1271) (2017). *Remote Sensing* (MDPI). | So sánh SG/LOESS/spline/AG/DL | Peer-reviewed |
| 11 | Nghiên cứu ngày gieo sạ lúa ĐBSCL dưới ảnh hưởng hạn/mặn (SG filter + ngưỡng 20% biên độ) — xem kết quả tổng hợp tại mục 5. | Ứng dụng thực địa Việt Nam | Peer-reviewed (cần người dùng tự truy xuất DOI đầy đủ, xem Hạn chế) |
| 12 | [Deep learning-based phenology extraction and crop classification in arid oasis using Sentinel-2 time series](https://pmc.ncbi.nlm.nih.gov/articles/PMC13183482/). *J. Zhejiang Univ.-SCIENCE B* (2025). | Kết hợp Sentinel-2 + DL trích SOS/POS/EOS | Peer-reviewed |

---

## 8. Hạn chế & công bố sử dụng AI

- Đây là nghiên cứu tổng hợp nhanh theo **chế độ `lit-review`** của skill `deep-research` (bộ `academic-research-skills`), áp dụng **thủ công** phương pháp luận của skill (đặt câu hỏi nghiên cứu → tìm kiếm hệ thống → tổng hợp chủ đề → annotated bibliography), **không** thông qua việc cài đặt plugin chính thức và **không** chạy đầy đủ vòng lặp 13-agent (chưa có bước Devil's Advocate checkpoint hay Source Verification Agent tự động chấm điểm từng nguồn).
- Phần lớn tài liệu được xác minh qua tiêu đề, tóm tắt và trích đoạn kết quả tìm kiếm web; **chưa đọc toàn văn** từng bài. Một số chi tiết trích dẫn (số tập/số trang chính xác) dựa trên hiểu biết nền có sẵn về các công trình kinh điển (Zhang 2003, Jönsson & Eklundh, Roerink 2000, Chen 2004, Atzberger & Eilers 2011) — **người dùng nên tự tra cứu và xác minh DOI/số trang chính xác trước khi trích dẫn chính thức trong báo cáo học thuật.**
- Mục #11 (nghiên cứu ĐBSCL) tổng hợp từ kết quả tìm kiếm nhưng chưa xác định được tên tác giả/DOI cụ thể trong phiên nghiên cứu này — cần người dùng tự tìm và bổ sung nếu muốn trích dẫn.
- Báo cáo này được tổng hợp với sự hỗ trợ của Claude (Anthropic) sử dụng công cụ tìm kiếm web (WebSearch) trong phiên làm việc ngày 2026-09-12.