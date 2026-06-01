# Glaucoma / Angle-Closure Screening Documentation

This folder contains the current Paper 2 documentation for slit-lamp image-based
angle-closure screening.

## Clinical Framing

The current project should be framed as angle-closure or narrow-angle referral
triage, not broad glaucoma diagnosis. The available endpoint is eye-level
gonioscopic Shaffer angle grade, supported by AS-OCT anterior segment biomarkers.

## Current Validation Method

The active manuscript method is now **patient-level train/validation only**, not
a separate held-out test split:

- Train: 80% of participants.
- Validation: 20% of participants.
- Stratification: angle-closure label at participant level.
- Leakage rule: both eyes and all images from the same participant stay in one
  split.
- Threshold selection and reported internal performance are based on validation.
- The paper should describe this as internal validation/model development, not
  locked-test evidence.

Patient-level cross-validation can still be reported as a robustness analysis,
but the main fixed-split workflow should not reserve a third test set unless an
external or truly locked cohort becomes available.

Primary binary label:

- Positive: Shaffer grade `0` or `1`.
- Negative: Shaffer grade `2`, `3`, or `4`.
- Missing, indeterminate, and `not seen` grades are excluded.

## Main Documents

- `MANUSCRIPT2_DRAFT.md`: current manuscript draft.
- `PAPER2_ANGLE_CLOSURE_FULL_DOCUMENTATION.md`: full technical and clinical documentation.
- `ANGLE_CLOSURE_STRATEGY_REVIEW.md`: current strategy review and model-selection notes.
- `AGENT_CLINICAL_PAPER_PLAN.md`: implementation and paper planning notes.
- `EXPERIMENT_SOLUTION_TREE.md`: ranked next-step solution candidates grounded in completed experiments and literature.

## Results

Key fixed-split and search summaries are in `results/`:

- `baseline_search_summary.md`
- `baseline_search_summary.csv`
- `best_model_next_experiments.md`
- `experiment_progress_chart.md`
- `experiment_progress_chart.svg`
- `experiment_progress_milestones.csv`
- `full_dataset_continuous_regression_cv_findings.md`
- `goal70_cv_findings.md`
- `resnet50_anatomy_stack_80_20_findings.md`
- `roi_shared_split_findings.md`
- `shaffer_grade_regression_80_20_findings.md`
- `split_manifest_summary.md`
- `split_manifest_summary.csv`

## Cross-Validation

Patient-level 5-fold validation summaries are in `cv/`:

- `image_only_multitask_5fold_README.md`
- `image_only_meta_risk_best_test_metrics_by_fold.csv`
- `image_only_meta_risk_best_test_metrics_summary.csv`
- `image_only_direct_probability_test_metrics_by_fold.csv`
- `image_only_direct_probability_test_metrics_summary.csv`
- `roi_local_aod_tisa_5fold_README.md`
- `roi_local_meta_risk_best_test_metrics_by_fold.csv`
- `roi_local_meta_risk_best_test_metrics_summary.csv`

## Current Conclusion

Neither the image-only multitask model nor the ROI local AOD/TISA model reached
stable sensitivity >=0.70 and specificity >=0.70 in patient-level 5-fold
validation. The defensible current claim is high-sensitivity angle-closure
referral triage, not a balanced open/closed angle classifier.

The newest fast 80/20 train-validation experiment found a better development
signal from a regression-only ResNet-50 anatomy stack. Excluding Shaffer grade
`2` as a borderline sensitivity analysis reached validation sensitivity `0.800`
and specificity `0.709` when the threshold was balanced internally on
validation. This is development evidence, not locked-test evidence.

The follow-up 5-fold continuous-regression experiments are documented in
`results/full_dataset_continuous_regression_cv_findings.md`. In short, the
10-parameter anatomy-regression stack remained the best signal, but the quick
frozen-backbone 5-fold runs did not stably reach sensitivity >=0.70 and
specificity >=0.70. Shaffer-grade regression alone was weaker, and the fixed
predicted-grade threshold `<=1.5` was not calibrated enough to use directly.

The newest ConvNeXt-Tiny anatomy-stack comparison modestly improved the strict
`0/1` versus `2/3/4` frozen-backbone CV result over ResNet-50, but still did not
reach stable 70/70. It should be treated as the current best strict-label
backbone signal, not as a final clinically sufficient model.

An attention-MIL version using ConvNeXt-Tiny image embeddings was also tested.
It did not improve over simple per-image anatomy prediction plus eye-level mean
aggregation.

The current best 5-fold candidate is a regularized unfrozen ConvNeXt-Tiny model
trained on the focused angle-6 anatomy target set. It reached mean AUROC `0.737`,
sensitivity `0.707`, and specificity `0.722` with validation-balanced fold
thresholds. This is the first mean 5-fold result above 70/70, but fold-level
instability and threshold calibration still need review before final manuscript
claims.
