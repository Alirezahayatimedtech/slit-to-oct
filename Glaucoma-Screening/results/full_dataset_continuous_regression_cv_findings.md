# Full-Dataset Continuous Regression CV Findings

Date: 2026-06-01.

## Purpose

These experiments tested whether continuous regression targets give a stronger
angle-closure screening signal than direct binary classification.

Primary clinical label:

- Positive: Shaffer grade `0` or `1`.
- Negative: Shaffer grade `2`, `3`, or `4`.
- Borderline sensitivity analysis: exclude Shaffer grade `2` and classify
  grade `0/1` versus `3/4`.

All experiments used patient-level 5-fold cross-validation. Both eyes and all
images from the same participant were kept in the same fold.

These are internal CV experiments, not external validation.

## Cohort Audit

The 10-anatomy-parameter complete-case cohort had 476 eyes:

| Shaffer grade | Eyes |
| --- | ---: |
| 0 | 12 |
| 1 | 24 |
| 2 | 76 |
| 3 | 317 |
| 4 | 47 |

Strict binary counts:

- Closed/occludable angle, grade `0/1`: 36 eyes.
- Non-closed/open, grade `2/3/4`: 440 eyes.

Clean-extremes sensitivity cohort after excluding grade `2`:

- Eyes: 400.
- Closed/occludable angle, grade `0/1`: 36 eyes.
- Open, grade `3/4`: 364 eyes.

Interpretation: grade `2` is large relative to the positive class. It is more
than twice the size of the grade `0/1` class and remains a major label-noise
and boundary-definition problem.

## Experiment 1: Shaffer Grade Regression, Strict 0/1 vs 2/3/4

Run path:

```text
slit-project/paper2_runs/exp1_shaffer_grade_regression_cv_complete476/
```

Method:

- Model: ResNet-50 image encoder with single regression head.
- Target: Shaffer grade normalized to `[0, 1]`.
- Loss: weighted MSE.
- Backbone: frozen for this quick CV pass.
- Binary conversion: predicted grade threshold.

Summary:

| Threshold rule | AUROC | AUPRC | Sensitivity | Specificity | Balanced min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed predicted grade `<=1.5` | 0.551 | 0.104 | 0.000 | 0.950 | 0.000 |
| Threshold from training folds | 0.551 | 0.104 | 0.223 | 0.699 | 0.223 |
| Threshold balanced on validation fold | 0.551 | 0.104 | 0.567 | 0.519 | 0.492 |

Conclusion: Shaffer grade regression with grade `2` kept in the negative class
did not work as a robust classifier in 5-fold CV. The fixed clinical threshold
of predicted grade `<=1.5` missed all positives because the predicted grade
scale was shifted upward.

## Experiment 2: Shaffer Grade Regression, Excluding Grade 2

Run path:

```text
slit-project/paper2_runs/exp2_shaffer_grade_regression_cv_exclude_grade2/
```

Method:

- Same model and loss as Experiment 1.
- Grade `2` eyes excluded from training and evaluation.
- Binary task: grade `0/1` versus `3/4`.

Summary:

| Threshold rule | AUROC | AUPRC | Sensitivity | Specificity | Balanced min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed predicted grade `<=1.5` | 0.590 | 0.144 | 0.067 | 0.955 | 0.067 |
| Threshold from training folds | 0.590 | 0.144 | 0.430 | 0.678 | 0.430 |
| Threshold balanced on validation fold | 0.590 | 0.144 | 0.598 | 0.583 | 0.548 |

Conclusion: excluding grade `2` improved AUROC and balance slightly, but Shaffer
grade regression still did not reach the target. The fixed grade threshold still
had very poor sensitivity.

## Experiment 3: 10-Parameter Anatomy Regression Stack, Strict 0/1 vs 2/3/4

Run path:

```text
slit-project/paper2_runs/exp3_resnet50_anatomy_stack_cv_complete476/
```

Method:

- Model: regression-only ResNet-50.
- Targets: 10 AS-OCT anatomy parameters.
- Classification stage: logistic regression on predicted anatomy values.
- Thresholds:
  - `balanced_min_from_train`: threshold chosen from training-fold predictions.
  - `balanced_min_from_val_internal`: threshold chosen on the validation fold;
    this is diagnostic/development only.

Targets:

- CCT
- ACD[Endo.]
- lens vault
- ACW
- AOD500 temporal
- AOD500 nasal
- TISA500 temporal
- TISA500 nasal
- TIA500 temporal
- TIA500 nasal

Summary:

| Threshold rule | AUROC | AUPRC | Sensitivity | Specificity | Balanced min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Threshold from training folds | 0.639 | 0.155 | 0.509 | 0.701 | 0.430 |
| Threshold balanced on validation fold | 0.639 | 0.155 | 0.606 | 0.667 | 0.597 |

Best individual fold:

- Fold 3 with validation-balanced threshold reached sensitivity `0.778` and
  specificity `0.736`.
- Other folds did not reach 70/70.

Conclusion: the anatomy-regression stack remains the best strict-label signal,
but it is not stable enough in 5-fold CV to claim balanced sensitivity and
specificity above 70%.

## Additional Check: Anatomy Regression Stack, Excluding Grade 2

Run path:

```text
slit-project/paper2_runs/exp3_resnet50_anatomy_stack_cv_exclude_grade2/
```

Method:

- Same as Experiment 3.
- Grade `2` excluded.

Summary:

| Threshold rule | AUROC | AUPRC | Sensitivity | Specificity | Balanced min |
| --- | ---: | ---: | ---: | ---: | ---: |
| Threshold from training folds | 0.656 | 0.159 | 0.422 | 0.727 | 0.382 |
| Threshold balanced on validation fold | 0.656 | 0.159 | 0.678 | 0.651 | 0.614 |

Best individual fold:

- Fold 4 with validation-balanced threshold reached sensitivity `0.833` and
  specificity `0.809`.
- Other folds did not reach 70/70.

Conclusion: excluding grade `2` still looks directionally helpful, but the 5-fold
result is not consistently above 70/70. The earlier 80/20 result of sensitivity
`0.800` and specificity `0.709` was a useful development signal but is not yet a
stable CV result.

## Main Comparison

Current ranking of signals:

1. Anatomy-regression stack is the strongest approach overall.
2. Grade-2 exclusion improves some metrics and supports the label-noise
   hypothesis, but it does not fully solve the problem in 5-fold CV.
3. Shaffer grade regression alone is weaker than anatomy regression.
4. The literal predicted-grade threshold `<=1.5` is not usable without
   calibration because the regressor is not calibrated on the clinical grade
   scale.

Answer to the main modeling question:

> Classification/risk modeling after anatomy regression is better than keeping
> grade `2` and doing Shaffer-grade regression alone.

## Method Implication

For the manuscript, the safest primary method remains:

- Primary endpoint: grade `0/1` versus `2/3/4`.
- Internal validation: patient-level split or patient-level CV.
- Report grade-2-excluded results only as a sensitivity analysis.
- Do not claim general glaucoma diagnosis.

If a clean clinical label-review workflow is available, the next highest-value
step is to re-review grade `1/2` and especially grade `2` eyes before additional
architecture search.

## Next Efficient Step

The best next model experiment is not another direct classifier. It should be:

1. Clinically review or flag borderline grade `1/2` and grade `2` eyes.
2. Keep the anatomy-regression stack as the baseline.
3. Improve calibration and thresholding of predicted anatomy-risk scores.
4. Run a longer unfrozen anatomy-regression stack only after the label review
   decision is fixed.
5. If grade labels remain noisy, frame the paper as angle-closure referral triage
   with high NPV rather than a balanced definitive open/closed classifier.
