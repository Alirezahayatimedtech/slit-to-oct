# Paper 2: Gonioscopy-Defined Angle-Closure Screening

## Scope

This study evaluates angle-closure referral screening from slit-lamp
photographs and measured anterior-segment anatomy. It does not evaluate all
forms of glaucoma and does not contain an endpoint for glaucomatous optic
neuropathy.

The intended question is:

> Can slit-lamp photographs identify eyes that should undergo formal gonioscopy
> or further anterior-chamber-angle assessment?

## Endpoint

The primary binary endpoint uses the eye-level modified Shaffer grade in the
published release:

| Class | Shaffer grade |
| --- | --- |
| Angle closure | `0` or `1` |
| Non-closure | `2`, `3`, or `4` |
| Excluded | missing, indeterminate, or `not seen` |

The current release contains an eye-level grade, not four quadrant-specific
grades. Claims based on a quadrant closure rule are therefore unsupported.

Prespecified secondary analyses are:

- grades `0/1/2` versus `3/4`;
- grades `0/1` versus `3/4`, excluding grade `2`;
- per-grade error analysis, particularly at the grade `1/2` boundary.

## Corrected Analysis Cohort

| Measure | Count |
| --- | ---: |
| Participants | 286 |
| Eyes | 560 |
| Positive participants | 30 |
| Positive eyes | 55 |
| Negative eyes | 505 |

All 286 participants remain in one outer fold each. Both eyes and every image
from one participant are kept together. One released eye with a `not seen`
grade and one grade-valid eye without a usable linked slit image are not part of
the paired primary analysis.

## Validation Contract

- Patient-disjoint five-fold out-of-fold evaluation is the controlling internal
  design.
- Imputation, scaling, regularization selection, and operating thresholds are
  fitted from training data only.
- Paired comparisons use predictions for the same eye set.
- Confidence intervals resample participants to preserve fellow-eye dependence.
- A fixed 80/20 split may be used for rapid development, but it is not the
  controlling publication result.
- The study has no external validation cohort.

## Current Results

### Primary paired comparison

| Model | Inputs available at evaluation | AUROC (95% CI) |
| --- | --- | ---: |
| Frozen corrected slit-lamp model | Slit-lamp photographs | `0.659 (0.554-0.758)` |
| Nested measured-AS-OCT model | AS-OCT biometric values | `0.824 (0.758-0.886)` |

The paired AUROC difference was `0.166` (participant-bootstrap 95% CI
`0.049-0.287`). Measured anterior-segment anatomy therefore provided more
information for the reference endpoint than the available slit-lamp images.

### Parsimonious anatomy benchmark

The strongest compact measured-variable model used:

- anterior chamber depth;
- lens vault;
- mean AOD500;
- mean TISA500; and
- age.

It achieved AUROC `0.841` (`0.785-0.890`), sensitivity `0.800`, and specificity
`0.756` with nested thresholds. This point estimate did not reproduce as stable
80/80 performance across repeated patient-disjoint partitions.

### Negative results that constrain the claim

- Adding routine clinical variables to core measured anatomy did not improve
  AUROC.
- Adding the slit score to measured AS-OCT variables did not exceed AS-OCT
  alone.
- AS-OCT regression targets and teacher probabilities used only during training
  did not produce a supported improvement in slit-only inference.
- Errors concentrated near adjacent Shaffer grades, especially grades 1 and 2.

## Interpretation

The current paper should report a paired-modality and grade-boundary finding:

> In this single-centre cohort, measured anterior-segment anatomy contained
> substantially more information for gonioscopy-defined angle closure than the
> available slit-lamp photographs, while residual disagreement concentrated
> near the binary Shaffer-grade boundary.

The analysis does not support replacement of gonioscopy, clinical deployment,
or diagnosis of open-angle glaucoma.

## Reproducible Script Map

All scripts are under `../slit-project/code/`.

| Script | Purpose |
| --- | --- |
| `train_angle_grade_regression_cv.py` | Patient-disjoint Shaffer-grade regression |
| `train_resnet50_anatomy_stack_cv.py` | Image-to-anatomy regression and closure stack |
| `train_convnext_mil_anatomy_stack_cv.py` | Multi-view ConvNeXt anatomy stack |
| `evaluate_anatomy_stack_feature_experiments.py` | Measured/predicted feature ablations |
| `evaluate_oof_threshold_calibration.py` | Out-of-fold threshold analysis |

Input paths are command-line arguments. Controlled images and derived manifests
must remain local. Output directories should contain the full configuration,
split manifest, eye-level predictions, fold metrics, and software versions for
any reported run.

## Required Reporting

- Describe the endpoint as Shaffer `0/1` versus `2/3/4`.
- State whether each model uses slit images, clinical variables, measured
  AS-OCT values, or AS-OCT supervision only during training.
- Report participant and eye counts with the positive count.
- Report AUROC and AUPRC with participant-cluster confidence intervals.
- Report sensitivity and specificity only for thresholds selected without test
  leakage.
- Keep exploratory secondary endpoints separate from the primary result.

## Limitations

- Single-centre retrospective cohort.
- Only 30 positive participants and 55 positive eyes.
- Eye-level rather than quadrant-resolved gonioscopy.
- No formal masked multi-grader adjudication for the released grade.
- No external or prospective validation.
- Device- and acquisition-specific AS-OCT and slit-lamp protocols.

The published dataset and access instructions are documented in the repository
[root README](../README.md).
