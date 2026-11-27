# NYU-654 clean rescore (602 frames; 52 leaked test ids removed)

Leaked: 50 via own-data rows, 3 via NYU-train rows (overlap 1). Baselines never trained on our manifest, so their shift is pure subset effect.

| Source | Model | d1 full | d1 clean | Δ | AbsRel full | AbsRel clean | RMSE full | RMSE clean | d1_si full | d1_si clean |
|---|---|---|---|---|---|---|---|---|---|---|
| bench | moge2-l+r9 | 94.1 | 93.9 | -0.1 | 0.087 | 0.088 | 0.366 | 0.367 | 96.2 | 96.2 |
| bench | unik3d-b+r9 | 91.4 | 91.4 | +0.0 | 0.107 | 0.107 | 0.410 | 0.411 | 95.7 | 95.7 |
| bench | ssdi-S21_scale2+v2-l+r0 | 90.1 | 89.8 | -0.3 | 0.102 | 0.104 | 0.394 | 0.399 | 92.0 | 91.8 |
| bench | moge2-b+r9 | 90.0 | 89.8 | -0.2 | 0.109 | 0.110 | 0.428 | 0.429 | 95.6 | 95.6 |
| bench | unik3d-s+r9 | 88.5 | 88.5 | +0.0 | 0.123 | 0.123 | 0.438 | 0.436 | 94.4 | 94.4 |
| bench | student-S21_scale2 | 88.8 | 88.4 | -0.4 | 0.108 | 0.110 | 0.412 | 0.417 | 92.0 | 91.8 |
| bench | student-S16_prox | 88.4 | 88.0 | -0.4 | 0.108 | 0.110 | 0.415 | 0.419 | 91.8 | 91.6 |
| bench | student-S10_ultra_L | 87.8 | 87.4 | -0.4 | 0.110 | 0.112 | 0.430 | 0.435 | 91.6 | 91.4 |
| bench | student-b10_last | 86.7 | 86.3 | -0.4 | 0.116 | 0.118 | 0.446 | 0.451 | 90.6 | 90.5 |
| bench | student-b10_best | 86.4 | 86.0 | -0.4 | 0.117 | 0.119 | 0.452 | 0.456 | 90.3 | 90.2 |
| bench | student-b1_mix | 86.1 | 85.7 | -0.5 | 0.119 | 0.121 | 0.435 | 0.441 | 90.2 | 89.9 |
| bench | student-b20core_last | 85.6 | 85.3 | -0.3 | 0.120 | 0.122 | 0.448 | 0.451 | 89.3 | 89.2 |
| tta | student-b20dws_last+flip | 85.6 | 85.2 | -0.4 | 0.121 | 0.122 | 0.446 | 0.450 | 89.8 | 89.7 |
| bench | moge2-s+r9 | 85.0 | 85.1 | +0.1 | 0.123 | 0.123 | 0.470 | 0.467 | 94.2 | 94.2 |
| tta | student-b20dws_last+ms+flip | 85.4 | 85.1 | -0.3 | 0.121 | 0.122 | 0.452 | 0.455 | 89.7 | 89.6 |
| finetune | student-b20dws_ft2_ctrl_last__ft2ctrl | 85.4 | 85.0 | -0.4 | 0.121 | 0.123 | 0.450 | 0.454 | 89.6 | 89.5 |
| bench | student-b20dws_best | 85.4 | 85.0 | -0.4 | 0.122 | 0.123 | 0.449 | 0.452 | 89.6 | 89.5 |
| bench | student-b20dws_last | 85.4 | 85.0 | -0.4 | 0.122 | 0.123 | 0.449 | 0.452 | 89.6 | 89.5 |
| tta | student-b20dws_last | 85.4 | 85.0 | -0.4 | 0.122 | 0.123 | 0.449 | 0.452 | 89.6 | 89.5 |
| tta | student-b20dws_last+ms | 85.3 | 85.0 | -0.4 | 0.122 | 0.123 | 0.453 | 0.456 | 89.5 | 89.4 |
| tta | student-b20dws_ft2_all_last+flip | 85.2 | 84.8 | -0.4 | 0.122 | 0.123 | 0.457 | 0.461 | 89.5 | 89.4 |
| finetune | student-b20dws_ft2_all_last__ft2all | 84.9 | 84.6 | -0.4 | 0.123 | 0.124 | 0.460 | 0.464 | 89.3 | 89.2 |
| tta | student-b20dws_ft2_all_last | 84.9 | 84.6 | -0.4 | 0.123 | 0.124 | 0.460 | 0.464 | 89.3 | 89.2 |
| finetune | student-b20dws_ft_ctrl_last__ftctrl | 85.0 | 84.6 | -0.4 | 0.123 | 0.124 | 0.461 | 0.465 | 89.3 | 89.2 |
| tta | student-b20dws_ft2_all_last+ms+flip | 84.9 | 84.6 | -0.4 | 0.122 | 0.124 | 0.465 | 0.468 | 89.4 | 89.3 |
| tta | student-b20dws_ft2_all_last+ms | 84.8 | 84.4 | -0.4 | 0.123 | 0.124 | 0.466 | 0.469 | 89.2 | 89.1 |
| finetune | student-b20dws_ft_hyper_last__fthyper | 84.1 | 83.7 | -0.4 | 0.126 | 0.127 | 0.472 | 0.475 | 88.9 | 88.7 |
| bench | student-b20_best | 82.6 | 82.5 | -0.1 | 0.136 | 0.136 | 0.489 | 0.488 | 87.3 | 87.2 |
| tta | student-b20dws_ft2_all_last+sz576x768 | 79.3 | 79.2 | -0.1 | 0.164 | 0.164 | 0.489 | 0.488 | 87.9 | 87.8 |
| tta | student-b20dws_last+sz576x768 | 77.6 | 77.5 | -0.0 | 0.172 | 0.172 | 0.505 | 0.504 | 88.5 | 88.3 |
| bench | student-b20_last | 69.3 | 69.3 | -0.0 | 0.198 | 0.197 | 0.642 | 0.638 | 76.6 | 76.7 |
