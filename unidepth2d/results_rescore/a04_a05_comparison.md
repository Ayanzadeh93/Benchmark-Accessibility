# A04/A05 rescoring - IBims-1 (results_rescore)

bF1 columns: legacy = protocol.boundary_f1_si (LEGACY, drops edgeless images); v2 = boundary_f1_si_v2 (both_empty='zero'); DP = DepthPro SI-bF1 (masked = valid-mask extension, the HEADLINE DepthPro number on IBims since its GT has holes; nomask = DepthPro reference, unmasked, GT holes counted as edges - not a headline number here). (n) = n_contributing; for legacy it counts every image not dropped (dropped = no GT edge OR no predicted edge). 'stored' = legacy value in results_bench (2026-11-23 stack).

| run | K to model | K=None policy | n | d1 | AbsRel | bF1 legacy (n) | stored legacy | bF1 v2 (n) | DP masked, headline (n) | DP nomask, holes=edges (n) |
|---|---|---|---|---|---|---|---|---|---|---|
| ibims__student-b10_last+kdef__a05__noK | none (assumed default K) | default_k | 100 | 0.6204 | 0.2115 | 0.0446 (78) | - | 0.0130 (99) | 0.0079 (100) | 0.0065 (100) |
| ibims__student-b10_last__a04 | GT K | legacy | 100 | 0.6681 | 0.2090 | 0.0442 (76) | 0.0442 | 0.0123 (99) | 0.0074 (100) | 0.0062 (100) |
| ibims__student-b10_last__a05__noK | none (assumed default K) | legacy | 100 | 0.5271 | 0.2537 | 0.0629 (94) | - | 0.0406 (99) | 0.0293 (100) | 0.0244 (100) |
| ibims__student-b20core_last+kdef__a05__noK | none (assumed default K) | default_k | 100 | 0.7185 | 0.1804 | 0.0403 (66) | - | 0.0085 (99) | 0.0045 (100) | 0.0038 (100) |
| ibims__student-b20core_last__a04 | GT K | legacy | 100 | 0.7427 | 0.1793 | 0.0405 (65) | 0.0405 | 0.0083 (99) | 0.0044 (100) | 0.0037 (100) |
| ibims__student-b20core_last__a05__noK | none (assumed default K) | legacy | 100 | 0.7223 | 0.1878 | 0.0477 (80) | - | 0.0152 (99) | 0.0094 (100) | 0.0079 (100) |
| ibims__student-b20dws_last+kdef__a05__noK | none (assumed default K) | default_k | 100 | 0.7056 | 0.1824 | 0.0321 (70) | - | 0.0088 (99) | 0.0052 (100) | 0.0044 (100) |
| ibims__student-b20dws_last__a04 | GT K | legacy | 100 | 0.7483 | 0.1792 | 0.0319 (71) | 0.0318 | 0.0087 (99) | 0.0051 (100) | 0.0043 (100) |
| ibims__student-b20dws_last__a05__noK | none (assumed default K) | legacy | 100 | 0.7233 | 0.1891 | 0.0381 (81) | - | 0.0122 (99) | 0.0074 (100) | 0.0062 (100) |

dws4 gate (bF1v2, paired image bootstrap, b20core -> b20dws, n=99): diff +0.00042, 95% CI [-0.00086, +0.00163] -> ACCEPT (CI includes 0)
