# ROI Shared-Split Findings

Date: 2026-05-26

Target: strict Shaffer grade `0/1` angle closure versus grade `2/3/4` non-closure.

## Why This Was Needed

Separate nasal and temporal Van Herick models must use the same patient split before their predictions are merged. Earlier local-view combined diagnostic outputs were generated from independently split nasal and temporal runs; those outputs can leak a patient across train/validation/test after merging and must not be used as valid evidence.

The implementation now supports:

- `--split-csv` in `train_angle_closure_multitask.py`.
- Patient-overlap failure checks in `combine_local_view_angle_predictions.py`.
- Clinical late-fusion feature sets in `evaluate_angle_closure_meta_risk.py`.
- Shared split files in `paper2_runs/angle_closure_roi_local_shared_split/`.

## Corrected Shared Split

| Split | Participants | Eyes | Closed Eyes | Open Eyes |
| --- | ---: | ---: | ---: | ---: |
| Train | 156 | 294 | 27 | 267 |
| Validation | 34 | 67 | 6 | 61 |
| Test | 34 | 68 | 6 | 62 |

Patient overlap across train/validation/test: 0.

## Valid ROI Runs

| Run | Validation-Selected Model | Test AUROC | Test Sens | Test Spec | TP/FP/TN/FN |
| --- | --- | ---: | ---: | ---: | --- |
| `angle_closure_roi_local_combined_aod_tisa_sharedsplit` | temporal AOD/TISA + clinical metadata, RF depth 2 | 0.632 | 0.500 | 0.726 | 3/17/45/3 |
| `angle_closure_roi_local_combined_aod_tisa_acd_lv_sharedsplit` | temporal AOD/TISA, logistic regression | 0.540 | 0.333 | 0.661 | 2/21/41/4 |

Neither corrected candidate passed the pre-specified validation gate of sensitivity >=0.80 and specificity >=0.80.

## Oracle Signal Check

Measured AS-OCT true anatomy was evaluated on the same shared split as an upper-bound diagnostic check.

Best validation-selected true-anatomy row:

- Feature set: true AOD500/TISA500 nasal + temporal.
- Model: ExtraTrees depth 2.
- Validation: AUROC 0.751, sensitivity 0.833, specificity 0.754.
- Test: AUROC 0.823, sensitivity 0.833, specificity 0.710.

Interpretation: anatomical signal exists, but the current validation split is small and true anatomy itself does not clear 80/80 when the operating point is selected on validation only. Image-predicted biomarkers are still below this upper-bound signal.

## Decision

Do not run final 5-fold for these ROI candidates. The current reference baseline remains `angle_closure_image_only_multitask`, framed as high-sensitivity angle-closure referral triage rather than balanced >80/80 classification.
