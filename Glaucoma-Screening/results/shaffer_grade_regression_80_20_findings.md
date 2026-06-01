# Shaffer Grade Regression 80/20 Findings

Date: 2026-05-27.

## Experiment

Goal: test whether predicting the continuous/ordinal Shaffer angle grade works
better than direct binary classification.

Design:

- Split: patient-level 80/20 train/validation.
- No separate test set.
- Input: slit-lamp images.
- Model: ResNet-50 image encoder with a single regression head.
- Target: eye-level Shaffer grade `0..4`.
- Eye-level aggregation: mean predicted grade across all images for that eye.
- Binary conversion: closed if predicted grade is below a threshold.
- Primary fixed threshold tested: predicted grade `<= 1.5`.
- Secondary internal threshold: validation-balanced predicted-grade threshold.

## Label Counts

The grade-regression run used all eyes with valid Shaffer grades and usable
images. This is larger than the 10-anatomy-parameter experiment because it does
not require complete AS-OCT biomarker targets.

| Shaffer grade | Eyes |
| --- | ---: |
| 0 | 21 |
| 1 | 28 |
| 2 | 86 |
| 3 | 346 |
| 4 | 48 |

Total:

- Eyes: 529
- Participants: 267
- Positive eyes for strict closure (`0/1`): 49
- Negative eyes (`2/3/4`): 480

## Results

### Unweighted Grade Regression

Run path:

```text
slit-project/paper2_runs/angle_grade_regression_80_20_fast/
```

| Split | Threshold rule | Predicted-grade threshold | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | Balanced min | TP/FP/TN/FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Validation | Fixed cutoff | 1.500 | 0.721 | 0.169 | 0.000 | 1.000 | NA | 0.907 | 0.000 | 0/0/98/10 |
| Validation | Balanced internally | 2.591 | 0.721 | 0.169 | 0.700 | 0.622 | 0.159 | 0.953 | 0.622 | 7/37/61/3 |

Validation grade-regression performance:

- MAE: 0.640 Shaffer grades
- Pearson r: 0.256

### Weighted Grade Regression

Run path:

```text
slit-project/paper2_runs/angle_grade_regression_80_20_weighted_fast/
```

This run used inverse-frequency weighting by Shaffer grade to reduce collapse
toward the majority grade range.

| Split | Threshold rule | Predicted-grade threshold | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | Balanced min | TP/FP/TN/FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Validation | Fixed cutoff | 1.500 | 0.755 | 0.190 | 0.000 | 0.980 | 0.000 | 0.906 | 0.000 | 0/2/96/10 |
| Validation | Balanced internally | 2.037 | 0.755 | 0.190 | 0.700 | 0.755 | 0.226 | 0.961 | 0.700 | 7/24/74/3 |

Validation grade-regression performance:

- MAE: 0.831 Shaffer grades
- Pearson r: 0.333

## Interpretation

Shaffer-grade regression is a stronger simple image-only signal than the prior
direct binary classifier. The weighted grade-regression run reached validation
AUROC `0.755`, with sensitivity `0.700` and specificity `0.755` after internal
threshold balancing.

The literal clinical cutoff `1.5` did **not** work in these quick runs. The
predicted grades were shifted upward, so nearly all closed eyes had predicted
grades above `1.5`. The useful signal is rank/order discrimination, not
well-calibrated grade scale yet.

Current conclusion:

> Predicting Shaffer grade as an ordinal/regression target is more promising
> than direct binary image classification, but the predicted grade requires
> calibration or ordinal/class-balanced training before using `1.5` as a fixed
> clinical threshold.

## Next Step

Use Shaffer-grade prediction as a candidate baseline, but improve it before
final comparison:

1. Train with a class-balanced ordinal or cumulative-link loss instead of plain
   regression.
2. Calibrate predicted grade on validation.
3. Test whether grade `2` should be treated as borderline/excluded in a
   sensitivity analysis.
4. Compare calibrated Shaffer-grade regression against the anatomy-stack model
   on the same patient-level 80/20 split.
