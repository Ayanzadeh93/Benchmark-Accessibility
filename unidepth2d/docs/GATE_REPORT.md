# Reproduction gate report

## sun / v2-l  (n=4396, variant=raw, file=sun-raw__v2-l+r9.json)
- d1: published 96.4, measured 95.71 (diff -0.69) -> FAIL [HARD]
- arel: published 6.8, measured 7.80 (diff +1.00) -> FAIL [HARD]
- d1_ssi: published 96.4, measured 98.06 (diff +1.66) -> PASS [soft]  (info: SSI-aligned reads high for metric models — not a target)
- F1: published 84.6, measured 84.24 (diff -0.36) -> PASS [soft]
- rho_a: published 93.4, measured 90.11 (diff -3.29) -> FAIL [soft]

## sun / v2-s  (n=4396, variant=raw, file=sun-raw__v2-s+r9.json)
- d1: published 90.8, measured 89.52 (diff -1.28) -> FAIL [HARD]  (V2-S HF-checkpoint drift, PROTOCOL_NOTES §2b #5)
- arel: published 10.5, measured 11.18 (diff +0.68) -> FAIL [HARD]  (V2-S HF-checkpoint drift, PROTOCOL_NOTES §2b #5)
- d1_ssi: published 90.8, measured 97.19 (diff +6.39) -> FAIL [soft]  (info: SSI-aligned reads high for metric models — not a target)
- F1: published 74.2, measured 75.31 (diff +1.11) -> PASS [soft]
- rho_a: published 87.7, measured 84.12 (diff -3.58) -> FAIL [soft]

## ibims / v2-l  (n=100, variant=default, file=ibims__v2-l+r9.json)
- d1: published 94.5, measured 94.59 (diff +0.09) -> PASS [HARD]
- arel: published 7.8, measured 8.48 (diff +0.68) -> FAIL [HARD]
- d1_ssi: published 94.5, measured 98.66 (diff +4.16) -> FAIL [soft]  (info: SSI-aligned reads high for metric models — not a target)
- F1: published 70.9, measured 73.17 (diff +2.27) -> FAIL [soft]
- rho_a: published 74.1, measured 72.45 (diff -1.65) -> PASS [soft]

## ibims / v2-s  (n=100, variant=default, file=ibims__v2-s+r9.json)
- d1: published 86.6, measured 82.53 (diff -4.07) -> FAIL [HARD]  (V2-S HF-checkpoint drift, PROTOCOL_NOTES §2b #5)
- arel: published 13.5, measured 14.16 (diff +0.66) -> FAIL [HARD]  (V2-S HF-checkpoint drift, PROTOCOL_NOTES §2b #5)
- d1_ssi: published 86.6, measured 97.54 (diff +10.94) -> FAIL [soft]  (info: SSI-aligned reads high for metric models — not a target)
- F1: published 62.4, measured 63.08 (diff +0.68) -> PASS [soft]
- rho_a: published 67.5, measured 65.14 (diff -2.36) -> FAIL [soft]

---
**GATE NOT PASSED** — do not mix published numbers into our tables.
