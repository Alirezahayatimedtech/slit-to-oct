# 80/80 Follow-Up Experiments After Exp9

Date: 2026-06-01.

Goal: test the most efficient routes from the current best model toward
sensitivity and specificity near `0.80/0.80` for strict angle-closure screening
(`Shaffer 0/1` versus `2/3/4`).

## Starting Point

Best model before this block:

```text
slit-project/paper2_runs/exp9_convnext_tiny_unfrozen_angle6_regularized_cv/
```

Method:

- ConvNeXt-Tiny, unfrozen.
- Angle-focused anatomy regression targets:
  - ACD
  - lens vault
  - AOD500 temporal/nasal
  - TISA500 temporal/nasal
- Usable labeled views only: `center`, `van_nasal`, `van_temporal`.
- Per-image anatomy predictions averaged to eye level.
- Logistic regression angle-closure risk model.
- Patient-level 5-fold validation.

Best exp9 result, using fold-internal validation-balanced thresholds:

| AUROC | AUPRC | Sensitivity | Specificity | Balanced min | 70/70 folds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.737 | 0.235 | 0.707 | 0.722 | 0.674 | 2/5 |

Interpretation: exp9 is still the leading model. It crosses mean `70/70`, but
not stable `80/80`. The main weaknesses are fold instability and threshold
transfer.

## Experiment 10: Threshold Calibration on Exp9

Run path:

```text
slit-project/paper2_runs/exp10_exp9_oof_threshold_calibration/
```

Question: can better threshold selection move exp9 closer to `80/80` without
retraining the image model?

Method:

- Reused exp9 out-of-fold eye-level predictions.
- For each held-out fold, selected thresholds using the other four folds.
- Tested balanced-min, Youden, sensitivity-70, and sensitivity-80 threshold
rules.

Key result:

| Threshold rule | Sensitivity | Specificity | Balanced min |
| --- | ---: | ---: | ---: |
| Current fold internal balanced threshold | 0.707 | 0.722 | 0.674 |
| Balanced threshold from other OOF folds | 0.667 | 0.697 | 0.614 |
| Sensitivity-80 threshold from other OOF folds | 0.745 | 0.615 | 0.549 |

Interpretation:

Threshold transfer is unstable. The model ranking signal is real, but a
threshold learned from other folds does not generalize well enough. This means
we should not claim that threshold tuning alone solves the target.

Decision:

- Keep exp9 as the best image/anatomy model.
- Do not rely on fold-internal threshold selection as final evidence.
- Improve calibration only after label/view quality is reviewed.

## Experiment 11: Clean-Label Sensitivity Analysis, Excluding Grade 2

Run path:

```text
slit-project/paper2_runs/exp11_convnext_tiny_unfrozen_angle6_regularized_exclude_grade2_cv/
```

Question: is Shaffer grade `2` the main barrier to better separation?

Method:

- Same architecture and training recipe as exp9.
- Excluded all grade `2` eyes from training and validation.
- Binary task becomes grade `0/1` versus grade `3/4`.
- Patient-level 5-fold validation.

Label counts:

| Grade | Eyes |
| ---: | ---: |
| 0 | 12 |
| 1 | 24 |
| 3 | 317 |
| 4 | 47 |

Key result, using fold-internal validation-balanced thresholds:

| AUROC | AUPRC | Sensitivity | Specificity | Balanced min | 70/70 folds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.726 | 0.227 | 0.722 | 0.750 | 0.655 | 2/5 |

Comparison to exp9:

| Model | AUROC | Sensitivity | Specificity | Balanced min |
| --- | ---: | ---: | ---: | ---: |
| Exp9 strict 0/1 vs 2/3/4 | 0.737 | 0.707 | 0.722 | 0.674 |
| Exp11 exclude grade 2 | 0.726 | 0.722 | 0.750 | 0.655 |

Interpretation:

Removing grade `2` improves mean sensitivity and specificity, but it does not
improve AUROC or fold stability. The clean-label run supports the idea that
grade `2` is clinically ambiguous, but it does not by itself create a stable
`80/80` model.

Decision:

- Keep grade-2 exclusion as a sensitivity analysis, not the primary endpoint.
- Clinical re-review of grade `1/2/3` borderline eyes is still the highest-value
  next data step.

## Experiment 12: ConvNeXt-Small Capacity Test

Run path:

```text
slit-project/paper2_runs/exp12_convnext_small_unfrozen_angle6_regularized_cv/
```

Question: does a larger ConvNeXt backbone improve the angle-6 anatomy stack?

Method:

- ConvNeXt-Small, unfrozen.
- Same angle-6 anatomy targets.
- Learning rate reduced to `3e-5`.
- Weight decay `5e-4`.
- Patient-level 5-fold validation.

Key result, using fold-internal validation-balanced thresholds:

| AUROC | AUPRC | Sensitivity | Specificity | Balanced min | 70/70 folds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.708 | 0.189 | 0.758 | 0.655 | 0.655 | 2/5 |

Interpretation:

ConvNeXt-Small did not improve over ConvNeXt-Tiny. It increased sensitivity
under fold-internal thresholds but lost specificity and AUROC. This is probably
over-capacity for the current small positive class and noisy labels.

Decision:

- Do not continue architecture scaling now.
- ConvNeXt-Tiny remains the preferred backbone.

## Experiment 13 and 14: OOF Calibration for Exp11 and Exp12

Run paths:

```text
slit-project/paper2_runs/exp13_exp11_oof_threshold_calibration/
slit-project/paper2_runs/exp14_exp12_oof_threshold_calibration/
```

Question: do the clean-label and ConvNeXt-Small runs have more stable threshold
transfer than exp9?

Key result:

| Source model | Threshold from other OOF folds: sensitivity | Specificity | Balanced min |
| --- | ---: | ---: | ---: |
| Exp9 strict ConvNeXt-Tiny angle-6 | 0.667 | 0.697 | 0.614 |
| Exp11 exclude grade 2 | 0.611 | 0.668 | 0.544 |
| Exp12 ConvNeXt-Small | 0.605 | 0.693 | 0.515 |

Interpretation:

Neither clean-label training nor ConvNeXt-Small fixed calibration transfer. The
threshold problem is not just a model-capacity problem.

Decision:

- Stop broad architecture experiments.
- Focus on clinical label review, grade-boundary analysis, and image/view
  quality filtering.

## Error Pattern

From exp9 and exp11:

- False negatives remain mostly grade `0/1`, especially grade `1`.
- False positives remain common among grade `2/3` in strict analysis.
- After excluding grade `2`, many false positives are grade `3`, which suggests
  either:
  - some grade `3` eyes have narrow-looking slit/anatomy features,
  - image quality/view issues are creating noisy predicted anatomy,
  - or eye-level Shaffer grade labels are not perfectly aligned with the
    photographed anatomy.

## Current Ranking After This Block

| Rank | Candidate | Status | Reason |
| ---: | --- | --- | --- |
| 1 | Exp9 ConvNeXt-Tiny angle-6 strict-label model | Best primary model | Highest AUROC and best balanced-min under strict primary label |
| 2 | Exp11 grade-2-excluded ConvNeXt-Tiny | Sensitivity analysis | Better mean sensitivity/specificity, but lower AUROC and unstable folds |
| 3 | Exp12 ConvNeXt-Small angle-6 | Failed capacity test | Lower AUROC and specificity than exp9 |
| 4 | Threshold-only calibration | Not sufficient | Other-fold thresholds degrade balanced performance |

## Next Practical Steps Toward 80/80

1. Clinically review false negatives and false positives from exp9 and exp11,
   prioritizing grade `1`, grade `2`, and grade `3` borderline cases.
2. Add a stricter image-quality/view-quality filter before retraining:
   - usable view only is already active;
   - next filter should inspect beam quality, limbal visibility, blur, and
     grossly off-target images.
3. Build a per-eye quality-weighted aggregation rule:
   - keep ConvNeXt-Tiny angle-6;
   - down-weight images with poor beam/view quality rather than averaging all
     usable images equally.
4. Re-run exp9 after label and image-quality cleanup.
5. Only after cleanup, revisit calibration with pooled out-of-fold predictions.

Bottom line:

> The best route to `80/80` is not a larger backbone. The signal says to keep
> ConvNeXt-Tiny angle-focused anatomy regression and improve label/image quality
> plus eye-level aggregation.
