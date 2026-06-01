# Experiment Solution Tree for Angle-Closure Screening

Date: 2026-06-01.

## Purpose

This document summarizes the best candidate solutions for each step of the
Paper 2 experiment after reviewing our local documentation, completed runs, and
the relevant angle-closure screening literature.

Current target:

- Strict angle-closure / occludable-angle screening.
- Positive: eye-level Shaffer grade `0/1`.
- Negative: eye-level Shaffer grade `2/3/4`.
- Grade `2` remains the key borderline/noise group and should be handled as a
  sensitivity analysis unless re-graded clinically.

Current best baseline:

- ConvNeXt-Tiny image encoder.
- Regression-only prediction of 10 AS-OCT anatomy targets.
- Per-image predictions averaged to eye level.
- Logistic regression risk model on predicted anatomy.
- Patient-level 5-fold CV, strict label: AUROC `0.655`, sensitivity `0.657`,
  specificity `0.642` with validation-balanced thresholds.

Current negative result:

- ConvNeXt-Tiny attention-MIL over image embeddings did not improve over simple
  eye-level mean aggregation.

## Literature Anchors

The strongest external evidence supports anatomy-first modeling rather than a
generic whole-image classifier.

1. AS-OCT deep learning studies commonly use the same strict binary grouping we
   now use: Shaffer grades `0/1` as closed and `2/3/4` as open. One CHES AS-OCT
   study trained five-grade classifiers and summed grade probabilities into
   closed versus open classes, reporting AUC about `0.93` for gonioscopic angle
   closure with subject-level leakage prevention.
   Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC6888901/

2. ACD is a major angle-closure risk factor, but single-parameter ACD screening
   is not enough by itself. An anterior-segment photograph DL study notes that
   ACD screening has reported sensitivity around `76.4-83.0%` and specificity
   around `67.2-88.9%`, while also emphasizing that angle closure is
   heterogeneous and benefits from multiple AS-OCT measurements.
   Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC9931242/

3. Slit-lamp image geometry can predict anterior chamber depth. A MIDAS
   smartphone/slit-lamp study used Van Herick-like slit-image features and
   random-forest regression to estimate ACD from slit-lamp images, with test
   `R2 = 0.73`.
   Source: https://pubmed.ncbi.nlm.nih.gov/34198935/

4. Another slit-lamp AI study estimated ACD and angle-closure risk from
   anterior-segment images and reported sensitivity `0.943`, specificity
   `0.902`, and AUC `0.923` for angle-closure risk.
   Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC11505230/

5. Van Herick image-analysis work argues for explicitly measuring the ratio
   between peripheral anterior chamber depth and corneal thickness, because the
   classic Van Herick method is subjective. A 2023 algorithmic study computes
   this ratio and reports clinical agreement above `65%`, reaching `100%` for
   grade 4.
   Source: https://iris.unimore.it/bitstream/11380/1309566/5/IWASI23_Anterior_Chamber_Angle_Assessment__An_Advanced_Image_Analysis_Algorithm_for_Van_Herick_Classification.pdf

6. Reviews of angle assessment emphasize quantitative anterior segment
   parameters including AOD, TISA, ACD, ACW, lens vault, and Van Herick grading,
   with gonioscopy remaining the clinical reference standard.
   Source: https://www.mdpi.com/2077-0383/9/12/3814

7. Reporting should follow STARD-AI for AI diagnostic accuracy studies.
   Source: https://www.equator-network.org/reporting-guidelines/the-stard-ai-reporting-guideline-for-diagnostic-accuracy-studies-using-artificial-intelligence/

## Step 1: Clinical Endpoint and Claim

Best candidate:

- Keep the primary endpoint as strict Shaffer `0/1` versus `2/3/4`.
- Frame the study as angle-closure referral triage, not general glaucoma
  diagnosis.

Why:

- Literature-compatible AS-OCT DL work uses grade `0/1` versus `2/3/4`.
- Our dataset has angle grades and AS-OCT anterior segment biomarkers, not optic
  nerve/RNFL/visual-field glaucoma endpoints.

Secondary candidates:

- Grade `0/1` versus `3/4`, excluding grade `2`, as a sensitivity analysis.
- Continuous/ordinal grade prediction as a calibration or auxiliary task, not as
  the primary classifier.

Avoid:

- Calling the model a broad glaucoma classifier.
- Treating grade `2` as positive unless the paper explicitly changes the
  clinical target to "narrow-or-closed referral" and justifies that change.

## Step 2: Label Quality

Best candidate:

- Clinically re-review grade `1`, grade `2`, and model-discordant grade `3`
  cases before additional architecture search.

What to do:

- Create a review list with patient ID, eye, grade, view labels, predicted risk,
  predicted AOD/TISA/ACD/lens vault, and whether the case was FP/FN across folds.
- Ask the glaucoma specialists to classify each case as:
  - closed/occludable
  - open
  - borderline/indeterminate
  - poor image or poor gonioscopy confidence
- Freeze a cleaned label policy before the next full training run.

Why:

- In the complete-case cohort, grade `2` has 76 eyes, compared with only 36
  grade `0/1` positive eyes.
- Our strongest 80/20 signal appeared when grade `2` was excluded, but 5-fold CV
  showed that exclusion alone is not enough unless the clinical label policy is
  cleaner.

Secondary candidates:

- Keep grade `2` in the negative class but down-weight it during classification.
- Train a three-class clinical target: closed `0/1`, borderline `2`, open `3/4`;
  then collapse to binary only for referral-threshold analysis.

Avoid:

- Repeated threshold tuning on the same noisy grade `2` cases.
- More architecture trials before the label policy is frozen.

## Step 3: Split and Validation Strategy

Best candidate:

- Use patient-level 80/20 train/validation for fast model development.
- Use patient-level repeated 5-fold or 5x2 CV only after a candidate is chosen.
- Keep all eyes and all images from a participant in the same split.

Why:

- The positive class is too small for unstable three-way internal splits.
- The literature and our own documentation emphasize subject-level leakage
  prevention when multiple images or both eyes exist.

Secondary candidates:

- Repeated stratified group CV to reduce fold variance once compute budget is
  available.
- Bootstrap confidence intervals clustered by participant.

Avoid:

- Separate train/val/test inside this small internal dataset unless a truly
  locked or external test cohort is available.
- Combining nasal and temporal view models trained on independent splits.

## Step 4: Image/View Selection

Best candidate:

- Keep `center`, `van_nasal`, and `van_temporal` as usable views, but make the
  model view-aware at the risk-model stage.

Concrete implementation:

- Predict anatomy per image.
- Aggregate per eye by view:
  - mean predicted ACD/AOD/TISA/lens vault for `center`
  - mean predicted nasal AOD/TISA for `van_nasal`
  - mean predicted temporal AOD/TISA for `van_temporal`
  - standard deviation and image count per view
- Feed these view-aware summary features into logistic regression.

Why:

- Our attention-MIL experiment did not learn useful image weighting.
- Mean aggregation is currently stronger, but simple global means may dilute
  nasal/temporal Van Herick signal.

Secondary candidates:

- View-specific models for `van_nasal` and `van_temporal` with a shared split.
- Center-only ACD/lens vault predictor plus Van Herick-only AOD/TISA predictor.

Avoid:

- Treating all images as exchangeable without view labels.
- Training side-specific targets with horizontal flip unless nasal/temporal
  labels are swapped correctly.

## Step 5: Anatomical Targets

Best candidate:

- Keep multi-target anatomy prediction, but prioritize clinically strongest
  target groups:
  - AOD500 nasal/temporal
  - TISA500 nasal/temporal
  - ACD
  - lens vault
  - ACW
  - CCT as calibration/supporting anatomy, not main signal

Why:

- Our direct binary classifiers underperformed.
- The anatomy-stack approach is currently the best internal signal.
- Literature supports multi-parameter anterior segment modeling because no
  single anatomy measurement captures all angle-closure mechanisms.

Secondary candidates:

- Train two specialized heads:
  - central/anterior chamber geometry head: ACD, lens vault, ACW, CCT
  - angle head: nasal/temporal AOD500, TISA500, TIA500
- Add ordinal Shaffer grade as an auxiliary head only after anatomy heads are
  stable.

Avoid:

- Shaffer-grade regression as the primary model. Our 5-fold results were weak,
  and the predicted grade scale was poorly calibrated.

## Step 6: Model Architecture

Best candidate:

- ConvNeXt-Tiny anatomy regression remains the current best baseline.
- Next architecture change should be unfrozen fine-tuning, not a new backbone.

Concrete next run:

```bash
conda run -n awg python slit-project/code/train_resnet50_anatomy_stack_cv.py \
  --outdir slit-project/paper2_runs/convnext_tiny_anatomy_stack_80_20_unfrozen \
  --backbone convnext_tiny \
  --folds 1 \
  --epochs 12 \
  --patience 4 \
  --batch-size 32 \
  --num-workers 4 \
  --amp
```

Why:

- Frozen ConvNeXt-Tiny improved modestly over frozen ResNet-50.
- Attention-MIL was negative.
- The likely bottleneck is adaptation to slit-lamp anatomy, not a more complex
  pooling head.

Secondary candidates:

- ConvNeXt-Small only after ConvNeXt-Tiny unfrozen improves.
- EfficientNetV2-S or Swin-Tiny as one limited comparison if ConvNeXt-Tiny
  unfrozen fails.

Avoid:

- Broad backbone search.
- Direct probability-head models without anatomy supervision.
- More frozen-backbone MIL variants.

## Step 7: Multi-Image Eye-Level Aggregation

Best candidate:

- Use per-image anatomy prediction followed by eye-level mean aggregation as the
  baseline.
- Improve it with view-aware summary features instead of neural attention.

Why:

- Current mean aggregation beats attention-MIL:
  - ConvNeXt mean aggregation: AUROC `0.655`, balanced-min `0.616`
  - attention-MIL all validation images: AUROC `0.608`, balanced-min `0.578`
  - attention-MIL capped at 12 images: AUROC `0.575`, balanced-min `0.572`

Secondary candidates:

- Robust aggregation: median, trimmed mean, min AOD/TISA per view, and
  low-percentile AOD/TISA.
- Quality-weighted aggregation using image blur/exposure/beam-confidence scores.

Avoid:

- Learned attention pooling unless the backbone is unfrozen and the positive
  class is larger.

## Step 8: Van Herick Geometry Extractor

Best candidate:

- Build a deterministic or semi-deterministic Van Herick feature extractor:
  - detect limbus/slit beam
  - estimate corneal slit width
  - estimate peripheral anterior chamber dark gap
  - compute ACD/CT-like ratio
  - generate nasal and temporal ratios separately

Why:

- Literature supports the Van Herick ratio as the directly relevant slit-lamp
  geometry.
- Our DL models may not reliably infer this geometry from full resized images.
- A geometry feature can be combined with predicted AS-OCT anatomy in a shallow
  risk model.

Secondary candidates:

- Train a segmentation/keypoint model for corneal line, iris line, limbus, and
  beam center using a small manually annotated subset.
- Use weak supervision: automatically propose beam/limbus candidates, then
  manually correct 100-200 images.

Avoid:

- Another full-image classifier before testing explicit Van Herick geometry.

## Step 9: Risk Model and Thresholding

Best candidate:

- Use a shallow, transparent risk model on predicted anatomy and geometry:
  - logistic regression with class balancing
  - regularized logistic regression
  - monotonic/simple rules for sensitivity analysis

Threshold policy:

- Select threshold on validation only.
- Report:
  - Youden threshold
  - sensitivity at fixed specificity targets
  - specificity at fixed sensitivity targets
  - clinically chosen referral threshold

Why:

- PPV is expected to be low because positives are rare.
- The paper is stronger if the risk model is interpretable and thresholding is
  explicit.

Secondary candidates:

- Calibrated gradient boosting if logistic regression underfits, but only after
  feature set is frozen.
- Isotonic or Platt calibration on validation predictions.

Avoid:

- Test-ranked model selection.
- Reporting only one validation-balanced threshold without confidence intervals.

## Step 10: Metrics and Reporting

Best candidate:

- Follow STARD-AI.
- Report patient-clustered confidence intervals.
- Emphasize positive-eye and positive-participant counts.

Required metrics:

- AUROC and AUPRC.
- Sensitivity, specificity, PPV, NPV.
- Balanced-min of sensitivity/specificity.
- Confusion matrix.
- Calibration/Brier score.
- Decision curve / referral burden.
- Subgroups: age, sex, lens status, angle grade.

Why:

- The small positive class makes point estimates unstable.
- AI diagnostic accuracy papers require clear handling of split, reference
  standard, missingness, and threshold selection.

Avoid:

- Saying "the model classifies glaucoma."
- Presenting internally selected thresholds as external validation.

## Recommended Next Experiment Order

### Experiment A: View-Aware ConvNeXt Anatomy Stack

Highest-value model experiment.

Design:

- Use ConvNeXt-Tiny anatomy predictions.
- Aggregate predicted anatomy by view, not only globally by eye.
- Add robust summaries: mean, median, minimum, low percentile, standard
  deviation, count.
- Fit logistic regression on validation-selected features.

Expected benefit:

- Preserves the current best signal while adding anatomical/view structure.

### Experiment B: Unfrozen ConvNeXt-Tiny Anatomy Regression

Second priority.

Design:

- Same 10 anatomy targets.
- Unfreeze ConvNeXt-Tiny.
- Start with 80/20 only.
- Use conservative epochs/patience.

Expected benefit:

- May improve image-to-biomarker quality more than pooling changes.

### Experiment C: Van Herick Geometry Features

Third priority and strongest clinically motivated new feature.

Design:

- Extract beam/limbus/ACD-CT ratio features from `van_nasal` and
  `van_temporal`.
- Combine with predicted AOD/TISA/ACD/lens vault.

Expected benefit:

- Adds explicit slit-lamp geometry that the DL model may miss.

### Experiment D: Label Review and Three-Class Modeling

Highest-value clinical/data experiment.

Design:

- Review grade `1/2/3` discordant cases.
- Train/report three classes: closed, borderline, open.
- Collapse to binary referral as needed.

Expected benefit:

- Addresses the main bottleneck: borderline labels are large relative to the
  positive class.

## Current Stop Rules

Stop doing:

- More direct binary classifiers.
- More frozen-backbone attention-MIL variants.
- More broad backbone search.
- More grade-regression-only models.
- Grade-2 exclusion as a main claim without clinical re-review.

Continue doing:

- Anatomy-first modeling.
- ConvNeXt-Tiny as the main baseline.
- Mean or view-aware aggregation.
- Shallow transparent risk models.
- Conservative validation and STARD-AI reporting.

## Current Best Paper Claim

The safest current claim is:

> Slit-lamp images can provide a measurable anterior-segment anatomical signal
> for gonioscopic angle-closure referral triage, especially when used to predict
> AS-OCT-derived angle biomarkers. However, with the current small positive
> cohort and borderline grade-2 label burden, the model has not yet reached
> stable balanced sensitivity and specificity above 70% in patient-level
> cross-validation.

