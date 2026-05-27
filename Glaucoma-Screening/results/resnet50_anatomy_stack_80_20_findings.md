# ResNet-50 Anatomy Stack 80/20 Findings

Date: 2026-05-27.

## Experiment

Goal: quickly test a label-first and anatomy-first strategy for strict
angle-closure screening without a separate test set.

Design:

- Split: patient-level 80/20 train/validation.
- No cross-validation for this quick experiment.
- Primary label: Shaffer grade `0/1` closed versus `2/3/4` open.
- Label-cleaning sensitivity check: repeat after excluding Shaffer grade `2`.
- Image model: regression-only ResNet-50 predicting 10 AS-OCT anatomical
  parameters.
- Classification: logistic regression on predicted anatomical values.
- Threshold rule: balance sensitivity and specificity. Both a training-derived
  threshold and an internal validation-balanced threshold were recorded.

Predicted anatomical parameters:

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

## Label Audit

Strict analytic set:

- Eyes: 476
- Participants: 258
- Positive eyes: 36
- Negative eyes: 440

Eye-level grade counts:

| Shaffer grade | Eyes |
| --- | ---: |
| 0 | 12 |
| 1 | 24 |
| 2 | 76 |
| 3 | 317 |
| 4 | 47 |

Interpretation: grade `2` is a large borderline group and should be reviewed or
handled as a sensitivity analysis before final modeling claims.

## Results

### Strict `0/1` vs `2/3/4`

Run path:

```text
slit-project/paper2_runs/resnet50_anatomy_stack_80_20_fast/
```

| Threshold rule | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | Balanced min | TP/FP/TN/FN | Reached 70/70 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Balanced from training predictions | 0.764 | 0.162 | 0.571 | 0.767 | 0.167 | 0.957 | 0.571 | 4/20/66/3 | No |
| Balanced on validation internally | 0.764 | 0.162 | 0.714 | 0.721 | 0.172 | 0.969 | 0.714 | 5/24/62/2 | Yes |

Shortfall review:

- Training-threshold errors included angle grades `0`, `1`, `2`, `3`, and `4`.
- Many false positives were grade `2/3` eyes with predicted narrow anatomy.
- Three of seven validation positives were missed by the training-derived
  threshold.

### Excluding Shaffer Grade `2`

Run path:

```text
slit-project/paper2_runs/resnet50_anatomy_stack_80_20_exclude_grade2_fast/
```

| Threshold rule | AUROC | AUPRC | Sensitivity | Specificity | PPV | NPV | Balanced min | TP/FP/TN/FN | Reached 70/70 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Balanced from training predictions | 0.711 | 0.236 | 0.800 | 0.696 | 0.143 | 0.982 | 0.696 | 4/24/55/1 | No |
| Balanced on validation internally | 0.711 | 0.236 | 0.800 | 0.709 | 0.148 | 0.982 | 0.709 | 4/23/56/1 | Yes |

Shortfall review:

- After excluding grade `2`, the remaining errors were grades `0` and `3`.
- Sensitivity improved to 80%.
- Specificity crossed 70% only when the threshold was balanced on validation.

## Interpretation

This quick run supports the label-first hypothesis. Removing Shaffer grade `2`
borderline eyes improved the clinically important sensitivity/specificity
balance for the anatomy-stacking strategy.

The result is still internal development evidence because the threshold was
balanced on the validation set. It should not be described as locked-test or
external-validation performance.

Best current signal from this experiment:

> Regression-only ResNet-50 anatomical prediction plus logistic regression,
> excluding Shaffer grade `2`, achieved validation sensitivity 0.800 and
> specificity 0.709 when the threshold was balanced internally on validation.

## Next Efficient Step

Before longer training:

1. Review Shaffer `1/2` and grade `2` borderline cases clinically if possible.
2. Keep strict `0/1` vs `2/3/4` as the primary endpoint.
3. Use the grade-2-excluded model only as a sensitivity analysis unless a
   clinical re-grade supports exclusion.
4. Run a short unfrozen ResNet-50 version on the same 80/20 split.
5. Inspect shortfall participants before spending time on 5-fold validation.
