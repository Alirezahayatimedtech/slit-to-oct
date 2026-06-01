# Best-Model Follow-Up Experiments

Date: 2026-06-01.

## Baseline Being Improved

Current best baseline before this experiment block:

- ConvNeXt-Tiny image encoder.
- Frozen ImageNet backbone.
- Regression-only prediction of 10 AS-OCT anatomy targets.
- Per-image anatomy predictions averaged to eye level.
- Logistic regression risk model for strict angle closure.
- Primary label: Shaffer grade `0/1` positive versus `2/3/4` negative.
- Patient-level 5-fold CV.

Baseline strict-label result:

| Model | Threshold rule | AUROC | AUPRC | Sensitivity | Specificity | Balanced min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny all-10 anatomy, mean aggregation | Validation-balanced | 0.655 | 0.180 | 0.657 | 0.642 | 0.616 |

## Experiment 1: View-Aware Aggregation

Run path:

```text
slit-project/paper2_runs/exp6_convnext_feature_experiments_from_best/
```

Method:

- Reused existing ConvNeXt-Tiny per-image anatomy predictions.
- Aggregated predictions separately for `center`, `van_nasal`, and
  `van_temporal`.
- Per view, computed mean, standard deviation, minimum, 10th percentile, and
  image count.
- Logistic regression classified strict angle closure from these features.

Result:

| Model | Threshold rule | AUROC | AUPRC | Sensitivity | Specificity | Balanced min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| View-aware binary aggregation | Validation-balanced | 0.620 | 0.133 | 0.607 | 0.669 | 0.579 |

Decision: **Failed.**

View-aware aggregation was worse than simple eye-level mean aggregation. This
suggests that the expanded feature set overfits the small positive class or that
the current per-image predictions are not reliable enough for view-specific
feature engineering.

## Experiment 2: Three-Class Risk Model

Run path:

```text
slit-project/paper2_runs/exp6_convnext_feature_experiments_from_best/
```

Method:

- Reused existing ConvNeXt-Tiny mean-aggregated anatomy predictions.
- Trained multinomial logistic regression with three classes:
  - closed: grade `0/1`
  - borderline: grade `2`
  - open: grade `3/4`
- Used predicted probability of the closed class as the binary risk score for
  strict `0/1` versus `2/3/4` evaluation.

Result:

| Model | Threshold rule | AUROC | AUPRC | Sensitivity | Specificity | Balanced min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Three-class mean-anatomy risk | Validation-balanced | 0.634 | 0.164 | 0.657 | 0.656 | 0.616 |

Decision: **Neutral / failed to improve.**

Three-class risk modeling matched the baseline balanced-min but reduced AUROC.
It is useful as a label-analysis tool, but not currently better as the main
classifier.

## Experiment 3: Target-Focused Anatomy Model

Run path:

```text
slit-project/paper2_runs/exp7_convnext_tiny_angle6_anatomy_stack_cv_complete476/
```

Method:

- ConvNeXt-Tiny frozen backbone.
- Regression targets reduced from 10 to 6 angle-relevant biomarkers:
  - ACD
  - lens vault
  - AOD500 temporal
  - AOD500 nasal
  - TISA500 temporal
  - TISA500 nasal
- Per-image predictions averaged to eye level.
- Logistic regression risk model.

Result:

| Model | Threshold rule | AUROC | AUPRC | Sensitivity | Specificity | Balanced min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny angle-6 anatomy | Validation-balanced | 0.662 | 0.206 | 0.663 | 0.654 | 0.636 |

Decision: **Small success.**

This is the best balanced-min result in this experiment block and slightly
improves the previous ConvNeXt all-10 baseline. One fold reached 70/70
(`sensitivity 0.778`, `specificity 0.780`), but the mean 5-fold result still
does not meet stable 70/70.

## Experiment 4: Unfrozen ConvNeXt-Tiny

Run path:

```text
slit-project/paper2_runs/exp8_convnext_tiny_unfrozen_anatomy_stack_cv_complete476/
```

Method:

- ConvNeXt-Tiny backbone unfrozen.
- 10 AS-OCT anatomy targets.
- Learning rate `1e-4`.
- 4 epochs with early stopping.
- Per-image predictions averaged to eye level.
- Logistic regression risk model.

Result:

| Model | Threshold rule | AUROC | AUPRC | Sensitivity | Specificity | Balanced min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny all-10 anatomy, unfrozen | Validation-balanced | 0.684 | 0.186 | 0.679 | 0.668 | 0.633 |

Decision: **Mixed.**

Unfreezing improved AUROC from `0.655` to `0.684`, but balanced-min remained
similar to the target-focused frozen model. Training loss dropped rapidly while
validation loss often worsened after epoch 1, suggesting fast overfitting. This
is promising, but needs regularized fine-tuning before it can become the main
method.

## Overall Ranking After This Block

| Rank | Candidate | Status | Main reason |
| ---: | --- | --- | --- |
| 1 | Target-focused ConvNeXt angle-6 anatomy | Best current balanced result | Highest balanced-min: `0.636` |
| 2 | Unfrozen ConvNeXt all-10 anatomy | Promising but overfits | Highest AUROC: `0.684` |
| 3 | Original frozen ConvNeXt all-10 anatomy | Still strong baseline | Stable, simple, interpretable |
| 4 | Three-class risk model | Neutral | Same balanced-min as baseline, lower AUROC |
| 5 | View-aware aggregation | Failed | Worse than global mean aggregation |

## Current Conclusion

The best next model direction is:

1. Use the **target-focused angle-6 anatomy target set** as the new primary
   model variant.
2. Combine it with **carefully regularized unfrozen ConvNeXt fine-tuning**.
3. Do not continue view-aware feature expansion unless stronger per-image
   anatomy predictions are available.
4. Keep three-class modeling as a label-analysis tool, not the main classifier.

The target remains unmet:

- No candidate achieved stable mean sensitivity >=0.70 and specificity >=0.70
  in patient-level 5-fold CV.
- The best mean balanced-min is now `0.636`.

## Next Practical Experiment

Run a regularized unfrozen target-focused model:

```bash
conda run -n awg python slit-project/code/train_resnet50_anatomy_stack_cv.py \
  --outdir slit-project/paper2_runs/exp9_convnext_tiny_unfrozen_angle6_regularized_cv \
  --backbone convnext_tiny \
  --target-preset angle6 \
  --folds 5 \
  --epochs 6 \
  --patience 2 \
  --batch-size 32 \
  --num-workers 4 \
  --amp \
  --lr 5e-5 \
  --weight-decay 5e-4
```

This combines the two best signals from the block:

- focused angle/anterior-chamber targets
- some backbone adaptation to slit-lamp anatomy

## Experiment 5: Regularized Unfrozen Target-Focused ConvNeXt

Run path:

```text
slit-project/paper2_runs/exp9_convnext_tiny_unfrozen_angle6_regularized_cv/
```

Method:

- ConvNeXt-Tiny backbone unfrozen.
- Target-focused angle-6 anatomy targets:
  - ACD
  - lens vault
  - AOD500 temporal
  - AOD500 nasal
  - TISA500 temporal
  - TISA500 nasal
- Learning rate reduced to `5e-5`.
- Weight decay increased to `5e-4`.
- 6 maximum epochs, patience 2.
- Per-image predictions averaged to eye level.
- Logistic regression risk model.

Result:

| Model | Threshold rule | AUROC | AUPRC | Sensitivity | Specificity | Balanced min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny unfrozen angle-6 regularized | Validation-balanced | 0.737 | 0.235 | 0.707 | 0.722 | 0.674 |

Decision: **Best result so far / partial success.**

This is the first mean 5-fold result in this experiment series where sensitivity
and specificity both exceed `0.70` when thresholds are balanced inside each
validation fold. It is not yet a final locked model because:

- the balanced-min remains below `0.70` (`0.674`);
- fold 1 remains weak (`sensitivity 0.600`, `specificity 0.651`);
- train-derived thresholds remain poor because one fold chooses an overly
  conservative threshold and misses all positives.

Still, this run is the strongest current direction and should replace the frozen
angle-6 model as the leading candidate.

Updated ranking:

| Rank | Candidate | Status | Main reason |
| ---: | --- | --- | --- |
| 1 | Regularized unfrozen ConvNeXt angle-6 anatomy | Best current result | AUROC `0.737`, sensitivity `0.707`, specificity `0.722` |
| 2 | Target-focused frozen ConvNeXt angle-6 anatomy | Good fallback | Balanced-min `0.636`, less overfit |
| 3 | Unfrozen ConvNeXt all-10 anatomy | Promising but less balanced | AUROC `0.684`, balanced-min `0.633` |
| 4 | Original frozen ConvNeXt all-10 anatomy | Stable baseline | AUROC `0.655`, balanced-min `0.616` |
| 5 | Three-class risk model | Neutral | Did not improve over baseline |
| 6 | View-aware aggregation | Failed | Worse than global mean aggregation |

Next step after this result:

1. Treat `exp9` as the new leading model candidate.
2. Tune threshold calibration and train-derived threshold selection, because the
   validation-balanced result is strong but the train-derived threshold remains
   unstable.
3. Inspect fold 1 false negatives/false positives and grade-2/grade-3
   borderline cases.
4. Repeat `exp9` with a fixed 80/20 split and bootstrap CI for a manuscript-style
   internal validation table.
