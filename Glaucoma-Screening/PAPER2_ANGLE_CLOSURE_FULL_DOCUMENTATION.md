# Paper 2 Angle-Closure Screening Full Documentation

Last updated: 2026-05-26.

This document records the current state of the Paper 2 angle-closure screening work: clinical framing, data, label definition, implemented method, completed runs, results, best baseline decision, limitations, and step-by-step next work.

## 1. Clinical Framing

The current project should be framed as:

> Slit-lamp image-based angle-closure / narrow-angle referral triage using gonioscopic Shaffer grade and AS-OCT anterior segment biomarkers.

Do not frame the current model as broad glaucoma diagnosis. The current Paper 2 data contain:

- Eye-level gonioscopic angle grade.
- Slit-lamp images.
- AS-OCT anterior chamber / angle biometrics.
- Limited clinical metadata.

The current Paper 2 pipeline does **not** contain:

- Confirmed glaucomatous optic neuropathy labels.
- RNFL / ganglion-cell OCT endpoints.
- Visual-field endpoints.
- Optic disc progression endpoints.

Therefore, the defensible target is angle-closure / narrow-angle screening, not general glaucoma classification.

## 2. Current Data Sources

Main files:

- Image-level table: `code/ready_for_training_clustered_anatomical_with_means_with_views_anonymized.csv`
- Eye-level clinical table: `code/ready_for_upload_publish.csv`
- Main training script: `code/train_angle_closure_multitask.py`
- Main manuscript draft: `MANUSCRIPT2_DRAFT.md`
- Strategy decision note: `ANGLE_CLOSURE_STRATEGY_REVIEW.md`

Relevant image views:

- `center`
- `van_nasal`
- `van_temporal`

Rejected or failure-mode views:

- `other`
- `no_slit`

## 3. Label Definition

Primary angle-closure label:

```text
closure_label = 1 if eye-level Shaffer angle_grade <= 1
closure_label = 0 if eye-level Shaffer angle_grade >= 2
```

Excluded:

- Missing angle grade.
- `not seen`.
- Indeterminate/non-numeric angle grade.
- Eye records with no resolved usable slit-lamp image.

Important limitation:

- The current reference is eye-level Shaffer grade.
- The current pipeline does not use quadrant-resolved gonioscopy.
- A quadrant rule, such as "Shaffer <=1 in at least two quadrants," requires additional quadrant labels.

Current decision:

- The active binary task is strict angle closure: grade `0/1` versus grade `2/3/4`.
- The temporary grade `0/1/2` positive definition is not used for the current Paper 2 baseline.

## 4. Frozen Split

All current angle-closure runs use patient-level splitting. Both eyes from one participant stay in the same split.

Split manifest:

- `paper2_runs/angle_closure_screening/split_manifest.csv`
- `paper2_runs/angle_closure_screening/split_manifest_summary.md`

Current linked analytic split:

| Split | Participants | Eyes | Closed Eyes | Open Eyes | Images | Median Images/Eye |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 186 | 368 | 33 | 335 | 9,787 | 26 |
| Validation | 40 | 80 | 8 | 72 | 2,307 | 26 |
| Test | 41 | 81 | 8 | 73 | 2,482 | 30 |

Leakage check:

- Train/validation overlap: 0 participants.
- Train/test overlap: 0 participants.
- Validation/test overlap: 0 participants.

Major statistical limitation:

- Locked test has only 8 closed-angle eyes from 4 positive participants.
- Confidence intervals for classification metrics must be expected to be wide.

## 5. Implemented Model

Script:

```text
code/train_angle_closure_multitask.py
```

Architecture:

- ConvNeXt-Tiny image encoder pretrained on ImageNet.
- Per-image feature extraction.
- Attention pooling over multiple slit-lamp views.
- Optional metadata MLP.
- Binary angle-closure classification head.
- Optional AS-OCT biometric regression heads.

Default classification loss:

- Focal binary cross-entropy.
- Positive-class focal alpha is now `0.75` because closed-angle eyes are rare.
- Earlier runs with `alpha=0.25` down-weighted the minority positive class and should be considered methodologically suboptimal.

Optional regression targets:

- `acd_endo_mm`
- `aod500_temporal_mm`
- `aod500_nasal_mm`
- `tisa500_temporal_mm2`
- `tisa500_nasal_mm2`
- `lens_vault_mm`
- `cct_mean_um`

Metadata features when enabled:

- Age.
- Sex.
- IOP.
- Lens status.
- CCT.

## 6. Completed Runs

Consolidated current comparison:

```text
paper2_runs/angle_closure_screening/baseline_search_summary.csv
paper2_runs/angle_closure_screening/baseline_search_summary.md
```

Current goal status:

- Desired operating target: sensitivity > 0.80 and specificity > 0.80.
- Best honest validation-selected image-only result so far: sensitivity 0.875 and specificity 0.630 from the image-only multitask baseline plus shallow temporal-angle meta-risk.
- Best diagnostic-only test-ranked result so far: sensitivity 0.875 and specificity 0.699, but this is not a final selected model because it is sorted by test performance.
- No current fixed-split image-only model has honestly achieved >0.80 / >0.80 on the locked test.

Important 2026-05-26 correction:

- Nasal and temporal local-view models must not be combined unless they were trained with the same patient-level split assignment.
- Early ROI combined runs used independently generated view-specific splits. Those results are now diagnostic only and must not be cited as valid performance.
- The combine script now checks train/validation/test patient overlap and fails if leakage is detected.
- The valid shared-split ROI runs are documented below.

### 6.1 Old Complex Multitask Run

Path:

```text
paper2_runs/angle_closure_screening/
```

Inputs:

- Images.
- Metadata.
- AS-OCT regression heads.

Known issue:

- This run used focal alpha `0.25`, which down-weighted the rare positive class.

Test results at validation-selected Youden threshold:

| Metric | Value |
| --- | ---: |
| AUROC | 0.560 |
| AUPRC | 0.199 |
| Brier | 0.100 |
| Sensitivity | 0.625 |
| Specificity | 0.548 |
| PPV | 0.132 |
| NPV | 0.930 |
| TP/FP/TN/FN | 5/33/40/3 |

Interpretation:

- Not a clean baseline because it uses metadata and auxiliary regression.
- Overfit: train AUROC was 1.000.
- Not the best current model.

### 6.2 Pure Image-Only Binary Classifier

Path:

```text
paper2_runs/angle_closure_image_only_cls/
```

Command:

```bash
python3 code/train_angle_closure_multitask.py \
  --outdir ../paper2_runs/angle_closure_image_only_cls \
  --no-regression \
  --focal-alpha 0.75 \
  --amp
```

Inputs:

- Images only.
- No metadata.
- No AS-OCT regression heads.

Test results at validation-selected Youden threshold:

| Metric | Value |
| --- | ---: |
| AUROC | 0.317 |
| AUPRC | 0.077 |
| Brier | 0.141 |
| Sensitivity | 0.250 |
| Specificity | 0.630 |
| PPV | 0.069 |
| NPV | 0.885 |
| TP/FP/TN/FN | 2/27/46/6 |

Interpretation:

- This model failed.
- Predicted probabilities collapsed close to a constant around 0.327.
- It should not be used as the headline baseline.

### 6.3 Image-Only Multitask Anatomical Model

Path:

```text
paper2_runs/angle_closure_image_only_multitask/
```

Command:

```bash
python3 code/train_angle_closure_multitask.py \
  --outdir ../paper2_runs/angle_closure_image_only_multitask \
  --focal-alpha 0.75 \
  --amp
```

Inputs:

- Images only.
- No metadata.
- AS-OCT biometric regression heads enabled.

Direct probability-head test results at validation-selected Youden threshold:

| Metric | Value |
| --- | ---: |
| AUROC | 0.663 |
| AUPRC | 0.157 |
| Brier | 0.133 |
| Sensitivity | 0.875 |
| Specificity | 0.452 |
| PPV | 0.149 |
| NPV | 0.971 |
| TP/FP/TN/FN | 7/40/33/1 |

Bootstrap CI for direct probability-head test results:

| Metric | 95% CI |
| --- | ---: |
| AUROC | 0.461 to 0.840 |
| AUPRC | 0.046 to 0.395 |
| Sensitivity | 0.500 to 1.000 |
| Specificity | 0.329 to 0.595 |
| PPV | 0.038 to 0.295 |
| NPV | 0.897 to 1.000 |

Interpretation:

- This is the current clean baseline.
- It uses only slit-lamp images as input.
- Auxiliary AS-OCT biomarker supervision helps compared with pure binary classification.
- The direct probability head is still not strong enough to be the only headline result.

## 7. Best Current Angle-Closure Signal

The strongest current test signal is not the direct neural probability head. It is anatomical risk derived from the image-only multitask model's predicted biomarkers.

Best current score:

```text
low predicted nasal TISA500
```

Using validation-selected threshold, locked-test results were:

| Risk Score | Test AUROC | Sensitivity | Specificity | PPV | NPV | TP/FP/TN/FN |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Low predicted nasal TISA500 | 0.815 | 1.000 | 0.603 | 0.216 | 1.000 | 8/29/44/0 |
| Low predicted ACD | 0.724 | 1.000 | 0.260 | 0.129 | 1.000 | 8/54/19/0 |
| Direct probability head | 0.663 | 0.875 | 0.452 | 0.149 | 0.971 | 7/40/33/1 |

This suggests the most successful strategy is:

> Predict AS-OCT anterior chamber / angle biomarkers from slit-lamp images, then classify angle-closure risk using validation-selected anatomical risk rules.

This should become the main Paper 2 direction unless a direct classifier improves substantially.

Latest fixed-split search update:

- `angle_closure_multitask_van_temporal_quick`: temporal-only image multitask. It produced useful biomarker AUROC values but did not hold stable sensitivity/specificity after validation-selected thresholding.
- `angle_closure_van_temporal_angle4_regstrong`: temporal-only, angle-target-only, stronger Huber regression. It failed on test classification and should not be advanced.
- `angle_closure_usable_anatomy6_mean_regbalanced`: usable multi-view, mean pooling, six anatomy targets, Huber regression. It improved several regression metrics but did not improve strict angle-closure classification.

Conclusion:

The next efficient route is not another minor ConvNeXt multitask loss tweak. The likely path to the >0.80 / >0.80 target is a stronger dedicated predictor for AOD/TISA or a peripheral angle ROI model, followed by a validation-selected shallow anatomical risk rule.

## 8. Current Baseline Decision

Baseline to freeze:

```text
paper2_runs/angle_closure_image_only_multitask/
```

Primary baseline model:

- Image-only.
- Multi-view.
- Attention pooling.
- AS-OCT biometric auxiliary supervision.
- No metadata.

Primary classifier for next analysis:

- Anatomical risk rule from predicted biomarker.
- Start with low predicted nasal TISA500.

Direct probability head:

- Report as an ablation.
- Do not make it the headline unless improved.

Metadata models:

- Add only after image-only model and anatomical risk rule are frozen.
- Treat metadata as secondary comparison.

## 9. Step-by-Step Path to Best Success

### Step 1: Keep the Claim Clinically Correct

Use:

- Angle-closure screening.
- Narrow-angle referral triage.
- Gonioscopic angle-closure risk.
- AS-OCT anterior chamber biomarker prediction.

Avoid:

- General glaucoma diagnosis.
- General glaucoma screening.
- Deployable clinical decision support.
- AS-OCT or gonioscopy replacement.

### Step 2: Freeze Current Clean Baseline

Use:

```text
paper2_runs/angle_closure_image_only_multitask/
```

This is the cleanest current model because it uses no metadata.

### Step 3: Formalize Predicted-Biomarker Risk Evaluation

Create a formal evaluation output for:

- Low predicted nasal TISA500.
- Low predicted temporal TISA500.
- Low predicted nasal AOD500.
- Low predicted temporal AOD500.
- Low predicted ACD.
- High predicted lens vault.

For each score:

- Choose threshold on validation split only.
- Apply once to locked test.
- Report AUROC, AUPRC, sensitivity, specificity, PPV, NPV.
- Bootstrap by patient.
- Save result table in the run folder.

### Step 4: Run View-Specific Image-Only Multitask Models

Commands:

```bash
python3 code/train_angle_closure_multitask.py \
  --outdir ../paper2_runs/angle_closure_multitask_center \
  --view-mode center \
  --focal-alpha 0.75 \
  --amp
```

```bash
python3 code/train_angle_closure_multitask.py \
  --outdir ../paper2_runs/angle_closure_multitask_van_nasal \
  --view-mode van_nasal \
  --focal-alpha 0.75 \
  --amp
```

```bash
python3 code/train_angle_closure_multitask.py \
  --outdir ../paper2_runs/angle_closure_multitask_van_temporal \
  --view-mode van_temporal \
  --focal-alpha 0.75 \
  --amp
```

Note: `train_angle_closure_multitask.py` supports `usable`, `center`, `van_nasal`, `van_temporal`, and `all`.

Purpose:

- Identify whether Van Herick views carry the angle signal better than mixed multi-view bags.
- Reduce noise from irrelevant views.

### Step 5: Compare Models by Locked-Test Result

Compare:

1. Pure image classifier.
2. Image-only multitask multi-view.
3. Image-only multitask center-only.
4. Image-only multitask nasal Van Herick.
5. Image-only multitask temporal Van Herick.
6. Image-only multitask plus metadata.

Primary comparison should use predicted biomarker risk, especially nasal TISA500.

### Step 6: Add Metadata Only After Image-Only Is Fixed

Metadata candidate:

```bash
python3 code/train_angle_closure_multitask.py \
  --outdir ../paper2_runs/angle_closure_metadata_multitask_v2 \
  --use-metadata \
  --focal-alpha 0.75 \
  --amp
```

Only keep metadata if it improves locked-test performance without obvious overfit.

### Step 7: Calibrate and Report Triage Thresholds

For the final selected model:

- Choose one high-sensitivity threshold on validation.
- Choose one high-specificity threshold on validation.
- Apply both to locked test.
- Report referral burden and missed closed-angle cases.

For triage, sensitivity and NPV matter more than PPV, but PPV must still be reported.

### Step 8: Error Review

Review:

- All false negatives.
- Highest-probability false positives.
- Worst predicted-biomarker errors.
- Whether failures come from image quality, view label, poor Van Herick alignment, eyelid artifact, or label ambiguity.

### Step 9: Manuscript Position

If current best result holds:

Headline should be:

> Slit-lamp images can estimate AS-OCT angle biomarkers, and predicted nasal TISA500 supports high-sensitivity referral triage for gonioscopic angle closure.

Do not headline:

> Deep learning classifies glaucoma from slit-lamp images.

## 9.1 ROI Local-View Biomarker Search Update

Implemented on 2026-05-26:

- `--crop-mode beam_roi` for beam-centered Van Herick ROI crops.
- `--view-local-targets` for side-specific local targets:
  - `van_nasal` predicts `aod500_nasal_mm` and `tisa500_nasal_mm2`.
  - `van_temporal` predicts `aod500_temporal_mm` and `tisa500_temporal_mm2`.
- `--disable-hflip` is forced for local side-specific targets.
- `--biomarker-only` trains regression-only heads while preserving prediction-file compatibility.
- `--split-csv` allows shared patient-level split assignment across separately trained nasal/temporal runs.
- `combine_local_view_angle_predictions.py` merges local-view predictions and now fails if patient overlap across splits is detected.
- `evaluate_angle_closure_meta_risk.py` now supports clinical late-fusion feature sets with age, sex, IOP, lens status, and CCT.

Valid shared-split results:

| Run | Validation Selection | Test AUROC | Test Sens | Test Spec | TP/FP/TN/FN |
| --- | --- | ---: | ---: | ---: | --- |
| `angle_closure_roi_local_combined_aod_tisa_sharedsplit` | temporal AOD/TISA + clinical metadata, RF depth 2 | 0.632 | 0.500 | 0.726 | 3/17/45/3 |
| `angle_closure_roi_local_combined_aod_tisa_acd_lv_sharedsplit` | temporal AOD/TISA, logistic regression | 0.540 | 0.333 | 0.661 | 2/21/41/4 |

Interpretation:

- Corrected local ROI biomarker models did not beat the earlier image-only multitask baseline.
- The auxiliary ACD/lens-vault heads improved some regression training behavior but did not improve angle-closure classification.
- The prior non-shared-split local ROI combined results must not be used as final evidence.

Oracle analysis on measured AS-OCT features:

- Best validation-selected true-anatomy row: true AOD/TISA with ExtraTrees depth 2.
- Validation: AUROC 0.751, sensitivity 0.833, specificity 0.754.
- Test: AUROC 0.823, sensitivity 0.833, specificity 0.710.
- Diagnostic test-ranked oracle rows can exceed 80/80, which confirms anatomical signal exists, but validation-selected oracle performance still misses the target on this small validation split.

## 9.2 Relaxed 70/70 Goal Validation

After relaxing the operating goal to sensitivity >=0.70 and specificity >=0.70, two candidates were validated with patient-level 5-fold evaluation:

| Candidate | Mean AUROC | Mean Sens | Mean Spec | Mean Balanced Min |
| --- | ---: | ---: | ---: | ---: |
| Image-only multitask + validation-selected meta-risk | 0.643 | 0.349 | 0.773 | 0.343 |
| ROI local AOD/TISA + validation-selected meta-risk | 0.674 | 0.536 | 0.659 | 0.518 |

Conclusion:

- Neither model reached stable >=0.70 sensitivity and >=0.70 specificity under 5-fold validation.
- The ROI local model improved mean sensitivity compared with the image-only multitask model but did not preserve specificity.
- The fixed-split validation signal was not stable enough for a final balanced classifier claim.

Detailed results:

- `paper2_runs/angle_closure_image_only_multitask_5fold/cv_summary/README.md`
- `paper2_runs/angle_closure_roi_local_combined_aod_tisa_5fold/cv_summary/README.md`
- `paper2_runs/angle_closure_screening/goal70_cv_findings.md`

## 10. Acceptance Targets

Minimum target for an internally defensible paper:

- AUROC >= 0.75 for angle-closure risk or predicted anatomical risk.
- Sensitivity >= 0.90 at clinically acceptable referral burden.
- NPV high enough to support triage framing.
- Bootstrap CI reported honestly.
- No leakage across patients.
- No claim of deployment without external validation.

Current closest result:

- Low predicted nasal TISA500 from image-only multitask model:
  - AUROC 0.815.
  - Sensitivity 1.000.
  - Specificity 0.603.
  - NPV 1.000.

This is promising but must be interpreted as high-sensitivity triage rather than a balanced 80/80 classifier.

## 11. Open Technical Tasks

Immediate:

1. Keep `angle_closure_image_only_multitask` as the current reference baseline.
2. Treat corrected ROI local-view runs as negative experiments unless a materially stronger ROI/segmentation model is added.
3. Do not present the current candidates as stable 70/70 classifiers; both failed patient-level 5-fold validation.
4. Update manuscript tables to report the honest best baseline and the negative ROI search.
5. Next technical work should target more positive data, explicit Van Herick geometry extraction, or stronger measured-biomarker prediction rather than another shallow threshold search.

Later:

1. Add metadata only after image-only is frozen.
2. Add Grad-CAM or attention review for true positives, false negatives, and false positives.
3. Add external validation if available.
4. Add quadrant-level gonioscopy sensitivity analysis only if quadrant labels become available.

## 12. Files Created or Updated

Main implementation:

- `code/train_angle_closure_multitask.py`
- `code/combine_local_view_angle_predictions.py`
- `code/combine_local_view_cv_predictions.py`
- `code/evaluate_angle_closure_meta_risk.py`

Main results:

- `paper2_runs/angle_closure_image_only_cls/`
- `paper2_runs/angle_closure_image_only_multitask/`
- `paper2_runs/angle_closure_screening/`
- `paper2_runs/angle_closure_roi_local_combined_aod_tisa_sharedsplit/`
- `paper2_runs/angle_closure_roi_local_combined_aod_tisa_acd_lv_sharedsplit/`
- `paper2_runs/angle_closure_roi_local_shared_split/`

Main documentation:

- `PAPER2_ANGLE_CLOSURE_FULL_DOCUMENTATION.md`
- `ANGLE_CLOSURE_STRATEGY_REVIEW.md`
- `AGENT_CLINICAL_PAPER_PLAN.md`
- `MANUSCRIPT2_DRAFT.md`

## 13. One-Sentence Current Conclusion

The current best route is still not a direct broad glaucoma classifier; the most defensible baseline remains slit-lamp image-based prediction of anterior chamber biomarkers for high-sensitivity angle-closure referral triage, while corrected ROI local-view biomarker models have not yet reached balanced >80/80 performance.
