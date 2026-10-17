---
title: b0_milestones
type: note
permalink: timeseries-final-crop-phenology/results/b0-milestones
---

# Clean-data Milestone Level Testing

## Seed 42
Valid True: SOS=260, POS=399, EOS=399
Intersection (detected by all): SOS=231, POS=356, EOS=357

| Model | Recall SOS | Recall POS | Recall EOS | MAE SOS (Self) | MAE POS (Self) | MAE EOS (Self) | MAE SOS (Int) | MAE POS (Int) | MAE EOS (Int) |
|---|---|---|---|---|---|---|---|---|---|
| transformer | 97.31% | 98.25% | 97.99% | 8.68 | 8.47 | 7.92 | 6.97 | 5.34 | 5.77 |
| transformer_cyclical_doy | 96.54% | 96.99% | 95.49% | 12.91 | 7.42 | 6.33 | 11.56 | 6.54 | 5.38 |
| transformer_calendar_gate | 94.62% | 94.99% | 94.49% | 9.70 | 6.65 | 6.82 | 8.57 | 5.14 | 4.86 |
| cba_phenonet | 95.00% | 93.98% | 96.49% | 9.90 | 5.19 | 5.13 | 8.44 | 4.42 | 4.29 |
## Seed 123
Valid True: SOS=260, POS=399, EOS=399
Intersection (detected by all): SOS=239, POS=378, EOS=372

| Model | Recall SOS | Recall POS | Recall EOS | MAE SOS (Self) | MAE POS (Self) | MAE EOS (Self) | MAE SOS (Int) | MAE POS (Int) | MAE EOS (Int) |
|---|---|---|---|---|---|---|---|---|---|
| transformer | 98.08% | 98.25% | 98.25% | 7.92 | 4.35 | 4.52 | 6.67 | 3.77 | 3.95 |
| transformer_cyclical_doy | 94.62% | 96.74% | 97.24% | 12.07 | 5.97 | 5.91 | 9.14 | 5.61 | 5.50 |
| transformer_calendar_gate | 96.92% | 97.74% | 96.74% | 9.17 | 6.19 | 6.26 | 9.25 | 5.66 | 5.04 |
| cba_phenonet | 98.46% | 99.25% | 98.75% | 7.29 | 5.77 | 5.33 | 6.15 | 4.93 | 4.50 |
## Seed 2024
Valid True: SOS=260, POS=399, EOS=399
Intersection (detected by all): SOS=240, POS=374, EOS=375

| Model | Recall SOS | Recall POS | Recall EOS | MAE SOS (Self) | MAE POS (Self) | MAE EOS (Self) | MAE SOS (Int) | MAE POS (Int) | MAE EOS (Int) |
|---|---|---|---|---|---|---|---|---|---|
| transformer | 96.15% | 96.49% | 96.99% | 9.42 | 4.97 | 4.77 | 8.62 | 3.68 | 3.33 |
| transformer_cyclical_doy | 97.31% | 97.99% | 97.49% | 9.43 | 4.68 | 4.67 | 7.90 | 4.44 | 3.93 |
| transformer_calendar_gate | 96.15% | 97.99% | 98.75% | 7.16 | 4.19 | 3.63 | 7.21 | 4.02 | 3.28 |
| cba_phenonet | 98.08% | 98.50% | 97.99% | 7.33 | 5.36 | 4.19 | 5.15 | 3.72 | 3.36 |
## Seed 7
Valid True: SOS=260, POS=399, EOS=399
Intersection (detected by all): SOS=236, POS=364, EOS=348

| Model | Recall SOS | Recall POS | Recall EOS | MAE SOS (Self) | MAE POS (Self) | MAE EOS (Self) | MAE SOS (Int) | MAE POS (Int) | MAE EOS (Int) |
|---|---|---|---|---|---|---|---|---|---|
| transformer | 96.54% | 97.74% | 96.49% | 9.32 | 7.41 | 6.40 | 7.44 | 5.22 | 5.50 |
| transformer_cyclical_doy | 93.46% | 95.49% | 93.98% | 7.16 | 4.97 | 5.20 | 7.27 | 4.79 | 5.04 |
| transformer_calendar_gate | 98.08% | 97.24% | 96.74% | 9.24 | 8.39 | 8.43 | 7.12 | 6.46 | 5.26 |
| cba_phenonet | 96.54% | 95.99% | 94.49% | 11.81 | 6.50 | 7.43 | 8.86 | 5.84 | 6.95 |
## Seed 314
Valid True: SOS=260, POS=399, EOS=399
Intersection (detected by all): SOS=233, POS=360, EOS=351

| Model | Recall SOS | Recall POS | Recall EOS | MAE SOS (Self) | MAE POS (Self) | MAE EOS (Self) | MAE SOS (Int) | MAE POS (Int) | MAE EOS (Int) |
|---|---|---|---|---|---|---|---|---|---|
| transformer | 95.38% | 95.74% | 97.49% | 11.13 | 6.95 | 6.47 | 9.16 | 4.38 | 4.26 |
| transformer_cyclical_doy | 95.38% | 96.74% | 90.98% | 25.42 | 10.69 | 9.21 | 22.04 | 8.78 | 8.77 |
| transformer_calendar_gate | 97.69% | 95.74% | 96.74% | 8.80 | 6.51 | 6.84 | 8.95 | 4.38 | 4.79 |
| cba_phenonet | 98.46% | 97.24% | 98.25% | 7.29 | 3.80 | 4.31 | 7.12 | 3.58 | 4.27 |