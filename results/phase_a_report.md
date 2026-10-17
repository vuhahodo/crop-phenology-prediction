---
title: phase_a_report
type: note
permalink: timeseries-final-crop-phenology/results/phase-a-report
---

# Phase A Report: Rigorous Extraction & Statistical Testing

## A1 & A2: Raw Per-Seed Data & Clean Performance Verification
The bug in previous `apply_cloud_dropout` zeroed out the DOY channels as well as NDVI, blinding the Cyclical DOY and CBA models to time and heavily biasing the evaluation towards the Standard Transformer (which uses fixed sequence index). The new simulation (Mode: Interp) masks only NDVI, applies linear interpolation and Savitzky-Golay filtering to mimic real preprocessing.

### Interpolated SG Masking (More Realistic)
**Dropout 10%**
- transformer: [0.7844797160508351, 0.7829838810816465, 0.7806154419970273, 0.7955281831283447, 0.7818001278877282] (Mean: 0.7851 ± 0.0054)
- transformer_cyclical_doy: [0.752181575189932, 0.7552522756163651, 0.7674643485649671, 0.7485625154799994, 0.7295205620304273] (Mean: 0.7506 ± 0.0123)
- transformer_calendar_gate: [0.7809294184797703, 0.7937292156611806, 0.758303900245751, 0.7902868183852144, 0.7866156480681372] (Mean: 0.7820 ± 0.0126)
- cba_phenonet: [0.7526035374677329, 0.7664031565674663, 0.7499418992825978, 0.7645376493828622, 0.7567360588317849] (Mean: 0.7580 ± 0.0065)

**Dropout 20%**
- transformer: [0.7516663434102719, 0.7479346656704593, 0.746132782228618, 0.7661194637942986, 0.7471691343231298] (Mean: 0.7518 ± 0.0074)
- transformer_cyclical_doy: [0.7112483915764292, 0.7214548372570013, 0.7261986163087286, 0.7102562886882032, 0.6973007440109843] (Mean: 0.7133 ± 0.0100)
- transformer_calendar_gate: [0.7451457261022807, 0.7565179513019695, 0.7192441192258993, 0.7515773988835619, 0.7496206236643772] (Mean: 0.7444 ± 0.0131)
- cba_phenonet: [0.7187294408486683, 0.7274715501218325, 0.7094831341316604, 0.7319811060760841, 0.7196237804576965] (Mean: 0.7215 ± 0.0078)

**Dropout 30%**
- transformer: [0.651773397144013, 0.6457121940044641, 0.6458738076491244, 0.6527899237004345, 0.6451099479675548] (Mean: 0.6483 ± 0.0033)
- transformer_cyclical_doy: [0.6140422828558324, 0.6287127913975726, 0.6310097429702979, 0.6281473735523642, 0.6150495391629366] (Mean: 0.6234 ± 0.0073)
- transformer_calendar_gate: [0.6448747586690997, 0.6444417399003541, 0.6331719502228269, 0.6428002577285559, 0.6365879408171152] (Mean: 0.6404 ± 0.0047)
- cba_phenonet: [0.6233498117753501, 0.6293764080386216, 0.6153546087827205, 0.6330129610439279, 0.6308362138857536] (Mean: 0.6264 ± 0.0064)

**Dropout 40%**
- transformer: [0.574519127395275, 0.5744323188842851, 0.5754856731355023, 0.5840238775777499, 0.5744721939676535] (Mean: 0.5766 ± 0.0037)
- transformer_cyclical_doy: [0.5562118374581718, 0.5582048674727949, 0.5607497272140767, 0.565986551298368, 0.5718730382737057] (Mean: 0.5626 ± 0.0057)
- transformer_calendar_gate: [0.5732561086062318, 0.5808791337162849, 0.5525183197680018, 0.5701667623296963, 0.5674950354615779] (Mean: 0.5689 ± 0.0093)
- cba_phenonet: [0.5502582529152393, 0.5619478446634448, 0.5510014518757135, 0.5671572187615648, 0.5654944485307614] (Mean: 0.5592 ± 0.0072)

**Dropout 50%**
- transformer: [0.5089675490737361, 0.5092056716606795, 0.5155368928944781, 0.5129790638411605, 0.5141347221317824] (Mean: 0.5122 ± 0.0026)
- transformer_cyclical_doy: [0.5004599397523193, 0.5082290646976326, 0.507550825265992, 0.5043822847840658, 0.511930856731129] (Mean: 0.5065 ± 0.0039)
- transformer_calendar_gate: [0.4991547439800687, 0.5108527403935894, 0.49853129436524063, 0.49780576358135276, 0.5029789585756285] (Mean: 0.5019 ± 0.0048)
- cba_phenonet: [0.49386972362712167, 0.5009500443261086, 0.49800412295281954, 0.5028255068970424, 0.5137060141448045] (Mean: 0.5019 ± 0.0066)

## A3: Performance Drop Paired T-Test
Tested on Performance Drop = (F1_clean - F1_drop). Pairs: CBA vs Cyclical, CBA vs Std, Cyclical vs Std, Gate-only vs Std. Mode: Interp.
Using n=5 seeds. Bonferroni alpha = 0.05 / 20 = 0.0025.

### cba_phenonet vs transformer_cyclical_doy
- **10%**: Diff = +0.0174, CI95%=[-0.0315, 0.0662], p=0.3794, p_adj=1.0000 (ns), d=0.44
- **20%**: Diff = +0.0166, CI95%=[-0.0367, 0.0700], p=0.4357, p_adj=1.0000 (ns), d=0.39
- **30%**: Diff = +0.0218, CI95%=[-0.0274, 0.0710], p=0.2856, p_adj=1.0000 (ns), d=0.55
- **40%**: Diff = +0.0282, CI95%=[-0.0294, 0.0859], p=0.2455, p_adj=1.0000 (ns), d=0.61
- **50%**: Diff = +0.0294, CI95%=[-0.0227, 0.0816], p=0.1922, p_adj=1.0000 (ns), d=0.70

### cba_phenonet vs transformer
- **10%**: Diff = +0.0198, CI95%=[-0.0058, 0.0455], p=0.0984, p_adj=1.0000 (ns), d=0.96
- **20%**: Diff = +0.0231, CI95%=[-0.0033, 0.0496], p=0.0721, p_adj=1.0000 (ns), d=1.09
- **30%**: Diff = +0.0147, CI95%=[-0.0131, 0.0424], p=0.2161, p_adj=1.0000 (ns), d=0.66
- **40%**: Diff = +0.0102, CI95%=[-0.0161, 0.0365], p=0.3414, p_adj=1.0000 (ns), d=0.48
- **50%**: Diff = +0.0031, CI95%=[-0.0223, 0.0285], p=0.7530, p_adj=1.0000 (ns), d=0.15

### transformer_cyclical_doy vs transformer
- **10%**: Diff = +0.0025, CI95%=[-0.0331, 0.0380], p=0.8561, p_adj=1.0000 (ns), d=0.09
- **20%**: Diff = +0.0065, CI95%=[-0.0351, 0.0481], p=0.6868, p_adj=1.0000 (ns), d=0.19
- **30%**: Diff = -0.0072, CI95%=[-0.0485, 0.0342], p=0.6559, p_adj=1.0000 (ns), d=-0.21
- **40%**: Diff = -0.0180, CI95%=[-0.0696, 0.0336], p=0.3869, p_adj=1.0000 (ns), d=-0.43
- **50%**: Diff = -0.0264, CI95%=[-0.0745, 0.0218], p=0.2035, p_adj=1.0000 (ns), d=-0.68

### transformer_calendar_gate vs transformer
- **10%**: Diff = +0.0044, CI95%=[-0.0277, 0.0364], p=0.7257, p_adj=1.0000 (ns), d=0.17
- **20%**: Diff = +0.0086, CI95%=[-0.0245, 0.0417], p=0.5093, p_adj=1.0000 (ns), d=0.32
- **30%**: Diff = +0.0091, CI95%=[-0.0135, 0.0317], p=0.3255, p_adj=1.0000 (ns), d=0.50
- **40%**: Diff = +0.0090, CI95%=[-0.0214, 0.0394], p=0.4591, p_adj=1.0000 (ns), d=0.37
- **50%**: Diff = +0.0115, CI95%=[-0.0142, 0.0373], p=0.2808, p_adj=1.0000 (ns), d=0.56

## A4: Absolute F1 T-Test (CBA vs Standard, Mode: Interp)
Bonferroni alpha = 0.05 / 5 = 0.01.

- **10%**: Diff = -0.0270, p=0.0007, p_adj=0.0036 (*)
- **20%**: Diff = -0.0303, p=0.0005, p_adj=0.0023 (*)
- **30%**: Diff = -0.0219, p=0.0025, p_adj=0.0126 (*)
- **40%**: Diff = -0.0174, p=0.0050, p_adj=0.0248 (*)
- **50%**: Diff = -0.0103, p=0.0258, p_adj=0.1290 (ns)

## A5: Gate g(t) Diagnostics
- Layer 0 g(t): Mean = 0.7097, Std (over time) = 0.0122
- Layer 1 g(t): Mean = 0.7628, Std (over time) = 0.0198
-> **Conclusion:** The gate values are virtually constant across the year! The model is not learning a season-dependent prior, but rather a constant scalar multiplier.

## A8: Full Ablation Compilation
Ablation over 6 configs across 5 seeds using extracted checkpoints/results.
(Detailed ablation stats can be compiled from CSVs, but primary findings are identical to previous ablation results.)

## Điểm bất thường / Điểm mâu thuẫn tôi thấy
1. **Lỗi mô phỏng mây che khuất:** Hàm `apply_cloud_dropout` cũ đã zero-out cả kênh thời gian DOY. Điều này chọc mù các mô hình dùng Cyclical DOY, khiến chúng bị phạt oan một cách nặng nề. Khi sửa lại chỉ drop kênh NDVI và nội suy (Mode: Interp), hiệu năng của các mô hình gần như tiệm cận nhau và khác biệt rơi vào vùng không có ý nghĩa thống kê.
2. **Cổng Calendar Gate là hằng số:** Phân tích A5 cho thấy `g(t)` gần như không đổi theo thời gian (std theo DOY rất nhỏ). Điều này bác bỏ hoàn toàn lập luận 'cổng điều tiết sự chú ý theo lịch mùa vụ'. Nó chỉ đóng vai trò như một hằng số gating dư thừa.
3. **Đóng góp của Calendar Gate là ảo ảnh:** Khi thử nghiệm công bằng với Interp-dropout, sức chịu đựng mây của CBA-PhenoNet chủ yếu đến từ hàm nội suy và Savitzky-Golay tiền xử lý, chứ không phải kiến trúc mạng.
4. **Lỗi báo cáo Spatial CV:** `± 0.014` trong bản thảo cũ là độ lệch chuẩn giữa 3 folds của 1 seed (seed 42), nhưng lại được báo cáo dưới cùng một định dạng với std của 5 random seeds, gây hiểu lầm nghiêm trọng về độ tin cậy.