# Angle-Closure Screening Split Summary

| split | participants | eyes | closed_eyes | open_eyes | images | median_images_per_eye |
| --- | --- | --- | --- | --- | --- | --- |
| test | 41 | 81 | 8 | 73 | 2482 | 30.0 |
| train | 186 | 368 | 33 | 335 | 9787 | 26.0 |
| val | 40 | 80 | 8 | 72 | 2307 | 26.0 |

## Leakage Check

- train_val: n=0
- train_test: n=0
- val_test: n=0

## Label Definition

Angle closure = eye-level Shaffer angle grade 0 or 1; missing/not seen grades excluded.
