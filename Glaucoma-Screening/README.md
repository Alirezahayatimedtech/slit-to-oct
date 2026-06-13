# Glaucoma / Angle-Closure Screening Documentation

This folder contains the current Paper 2 documentation for slit-lamp image-based
angle-closure screening.

## Clinical Framing

The current project should be framed as angle-closure or narrow-angle referral
triage, not broad glaucoma diagnosis. The available endpoint is eye-level
gonioscopic Shaffer angle grade, supported by AS-OCT anterior segment biomarkers.

The intended low-resource use case is:

> slit-lamp anterior-segment image(s) -> identify eyes that should be referred
> for formal gonioscopy / angle-closure evaluation.

The model should not be described as a complete glaucoma diagnostic system. It
cannot replace optic nerve assessment, intraocular pressure, visual fields, OCT
RNFL/GCC, or specialist examination. A future broader low-resource glaucoma
triage system may combine fundus-based optic nerve screening, slit-lamp-based
angle-closure risk, IOP, symptoms, age, and clinical risk factors.

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

Secondary and sensitivity labels:

- Broader referral endpoint: positive Shaffer `0/1/2` versus negative `3/4`.
- Borderline sensitivity analysis: Shaffer `0/1` versus `3/4`, excluding grade
  `2`.
- Optional auxiliary task: ordinal Shaffer grade `0` to `4`; this should support
  training and error analysis, not replace the primary binary endpoint.

## Main Documents

- `MANUSCRIPT2_DRAFT.md`: current manuscript draft.
- `METHODS_RESULTS_NPJ_STYLE.md`: manuscript-ready Methods/Results text and tables for the iterative model-development story.
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
- `exp9_to_exp14_80_80_followup.md`
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

## Updated Next-Step Plan

The next phase should test whether the performance ceiling is caused by weak
slit-lamp signal, limited AS-OCT numeric biomarkers, AS-OCT/gonioscopy
discordance, or noisy/borderline gonioscopy labels.

### Step 1: OCT-derived parameters -> gonioscopy baseline

First, test whether real measured AS-OCT parameters can predict gonioscopy before
using AS-OCT images or slit-lamp images.

Recommended models:

- Logistic regression with class weighting as the primary interpretable baseline.
- Random forest or gradient-boosted trees as nonlinear baselines.
- Single-parameter models for `ACD`, `LV`, `AOD500`, `AOD750`, `TISA500`,
  `TISA750`, `TIA500`, and `TIA750`.
- Focused angle-only feature sets versus all available AS-OCT parameters.

Required analyses:

- Patient-level 5-fold cross-validation.
- Primary endpoint: Shaffer `0/1` versus `2/3/4`.
- Secondary endpoint: Shaffer `0/1/2` versus `3/4`.
- Sensitivity endpoint: Shaffer `0/1` versus `3/4`, excluding grade `2`.
- Per-grade error analysis, especially grades `1`, `2`, and `3`.

Interpretation rule:

- If OCT parameters only reach moderate performance, this does not automatically
  mean gonioscopy grading is wrong. It may reflect expected discordance between
  static quantitative AS-OCT anatomy and dynamic clinical gonioscopy.
- If OCT parameters perform strongly, then the slit-lamp task should be treated
  as a problem of learning AS-OCT-relevant anatomy from low-cost images.

### Step 2: AS-OCT images -> gonioscopy upper-bound model

If the 10 AS-OCT images per eye are available, train an AS-OCT image-set model to
predict gonioscopy directly.

Recommended architecture:

```text
10 AS-OCT images per eye
        -> shared CNN / ConvNeXt / ViT encoder
        -> attention or DeepSets pooling
        -> eye-level closure probability
```

Purpose:

- Estimate the image-based anatomical upper bound for gonioscopy prediction.
- Test whether raw AS-OCT images contain information beyond the 10 numeric
  parameters.
- Generate an anatomical teacher signal for the slit-lamp model.

Important leakage rule:

- AS-OCT teacher models must be trained and evaluated with patient-level splits.
- Teacher predictions used for downstream slit-lamp training should be
  out-of-fold predictions, so the teacher never predicts a patient it saw during
  training.

### Step 3: AS-OCT teacher -> slit-lamp student

If AS-OCT images outperform OCT-derived parameters, use the AS-OCT image model as
an anatomical teacher. The final deployable model should still use slit-lamp
images only at inference.

Training target:

```text
slit-lamp image(s) -> gonioscopy label
                   -> AS-OCT teacher soft probability
```

Recommended loss:

```text
total loss = BCE(slit prediction, gonioscopy label)
           + lambda_1 * MSE/KL(slit prediction, AS-OCT teacher probability)
           + lambda_2 * Huber(predicted AS-OCT anatomy, measured AS-OCT targets)
```

Rationale:

- Gonioscopy labels are clinically meaningful but subjective and partly dynamic.
- AS-OCT image-teacher probabilities provide a continuous anatomical risk signal.
- The student can learn the hidden anatomical continuum from wide angle to
  borderline/narrow angle instead of relying only on hard binary labels.

Optional stronger version:

- Align slit-lamp and AS-OCT latent embeddings with cosine or contrastive loss.
- Use AS-OCT teacher confidence to down-weight discordant or low-confidence
  training examples.

### Step 4: Discordance and label-quality review

Create a review table comparing:

```text
gonioscopy Shaffer label
OCT-parameter risk
AS-OCT image-teacher risk
slit-lamp model risk
```

Priority cases for glaucoma-specialist review:

- Shaffer `0/1` but low AS-OCT risk.
- Shaffer `3/4` but high AS-OCT risk.
- Shaffer `2` with high or low AS-OCT risk.
- Recurrent false negatives among grade `1` eyes.
- Recurrent false positives among grade `2/3` eyes.

The goal is not to claim that gonioscopy is wrong, but to identify borderline,
low-quality, or physiologically discordant cases.

### Step 5: Slit-lamp final model and reporting

The final model should be reported as:

> slit-lamp image-based referral triage for gonioscopic angle closure.

Do not claim:

- Full glaucoma diagnosis.
- Detection of open-angle glaucoma.
- Replacement of gonioscopy.
- Locked-test clinical performance unless a truly locked or external cohort is
  available.

Preferred manuscript phrasing:

> We used AS-OCT as privileged anatomical supervision to train a slit-lamp-based
> model for gonioscopy-defined angle-closure referral triage.

### Immediate Implementation Checklist

1. Audit the tabular AS-OCT/gonioscopy file for missing angle labels, duplicated
   eye identifiers, missing OCT parameters, and inconsistent eye laterality.
2. Run OCT-parameter baselines for the three endpoint definitions above.
3. Build the AS-OCT image-set teacher with patient-level 5-fold validation.
4. Generate out-of-fold AS-OCT teacher probabilities for each eye.
5. Train slit-lamp student models with hard gonioscopy labels plus AS-OCT teacher
   soft labels.
6. Compare direct slit-lamp, anatomy-regression slit-lamp, AS-OCT-teacher
   distillation, and optional embedding-alignment variants.
7. Prepare a discordance table for clinical review before final manuscript
   interpretation.
