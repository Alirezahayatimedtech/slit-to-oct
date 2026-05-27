# Glaucoma / Angle-Closure Screening Documentation

This folder contains the current Paper 2 documentation for slit-lamp image-based
angle-closure screening.

## Clinical Framing

The current project should be framed as angle-closure or narrow-angle referral
triage, not broad glaucoma diagnosis. The available endpoint is eye-level
gonioscopic Shaffer angle grade, supported by AS-OCT anterior segment biomarkers.

Primary binary label:

- Positive: Shaffer grade `0` or `1`.
- Negative: Shaffer grade `2`, `3`, or `4`.
- Missing, indeterminate, and `not seen` grades are excluded.

## Main Documents

- `MANUSCRIPT2_DRAFT.md`: current manuscript draft.
- `PAPER2_ANGLE_CLOSURE_FULL_DOCUMENTATION.md`: full technical and clinical documentation.
- `ANGLE_CLOSURE_STRATEGY_REVIEW.md`: current strategy review and model-selection notes.
- `AGENT_CLINICAL_PAPER_PLAN.md`: implementation and paper planning notes.

## Results

Key fixed-split and search summaries are in `results/`:

- `baseline_search_summary.md`
- `baseline_search_summary.csv`
- `goal70_cv_findings.md`
- `roi_shared_split_findings.md`
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

