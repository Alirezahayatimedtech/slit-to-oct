# Angle-Closure Screening Strategy Review

Last updated: 2026-05-26.

## Current Update

The label is locked back to the literature-compatible strict angle-closure definition:

- Positive class: eye-level Shaffer grade `0` or `1`.
- Negative class: eye-level Shaffer grade `2`, `3`, or `4`.
- Missing, `not seen`, or indeterminate grades are excluded.

The temporary grade `0/1/2` positive experiment should be ignored for Paper 2. It is not the current clinical target and did not improve the anatomical upper-bound signal.

No more routine 5-fold runs should be used during model search. The current search uses the fixed patient-level train/validation/test split; 5-fold validation is reserved only for the final selected approach.

Current goal status:

- Target goal: sensitivity > 0.80 and specificity > 0.80 for strict angle closure.
- Best validation-selected image-only result so far: sensitivity 0.875, specificity 0.630 from the image-only multitask model plus shallow temporal-angle meta-risk.
- Best diagnostic-only test row so far: sensitivity 0.875, specificity 0.699 from the same baseline, but this row is sorted by test performance and must not be used as a final selected model without a new locked evaluation.
- The >0.80 / >0.80 goal has not been reached honestly on the current locked test.
- New ROI local-view experiments were run on 2026-05-26. The corrected shared-split versions did not reach the target; the best shared-split ROI AOD/TISA + clinical metadata candidate had validation sensitivity 0.833/specificity 0.738 and test sensitivity 0.500/specificity 0.726.
- Earlier combined nasal/temporal ROI outputs created before the shared split correction are diagnostic only. They merged models trained with independently generated splits, which can leak patients across split files.
- Relaxing the target to >=0.70 sensitivity and >=0.70 specificity still did not produce a stable model in 5-fold validation. Image-only multitask + meta-risk averaged sensitivity 0.349/specificity 0.773; ROI local AOD/TISA + meta-risk averaged sensitivity 0.536/specificity 0.659.

Consolidated run summary:

- `paper2_runs/angle_closure_screening/baseline_search_summary.csv`
- `paper2_runs/angle_closure_screening/baseline_search_summary.md`
- `paper2_runs/angle_closure_screening/goal70_cv_findings.md`

## Current Clinical Target

The current dataset supports angle-closure / narrow-angle referral triage, not broad glaucoma diagnosis. The available labels are eye-level Shaffer angle grade and AS-OCT anterior segment biometrics. There are no optic nerve, RNFL, visual-field, or confirmed glaucomatous optic neuropathy endpoints in the current Paper 2 pipeline.

Primary label currently implemented:

- `closure_label = 1` when eye-level Shaffer `angle_grade <= 1`.
- Missing, `not seen`, or indeterminate grades are excluded.
- Current linked split: 529 eyes from 267 participants.
- Locked test set: 81 eyes, 8 closed-angle eyes from 4 positive participants.

The small number of positive test participants is the main statistical limitation.

## Method Audit

What is correct:

- Patient-level splitting is enforced; both eyes from a participant stay in the same split.
- The clinical and image tables merge correctly by anonymized participant and eye identifiers.
- View filtering uses usable slit-lamp views: `center`, `van_nasal`, `van_temporal`.
- The manuscript correctly states that the reference standard is eye-level Shaffer grade, not quadrant-resolved gonioscopy.

What needed correction:

- The original focal-loss default used `alpha=0.25`, which down-weighted the rare positive class. The code default is now `alpha=0.75`.
- High-specificity threshold selection could write `Infinity` when no finite threshold reached the target. The helper now returns a finite threshold.
- Local nasal/temporal view models must be trained with the same external patient split CSV before their prediction files are combined. The combine script now checks for patient overlap and fails fast if leakage is detected.
- Biomarker-only training now skips batches with no available regression target mask, which can happen in side-specific AOD/TISA training.

What remains weak:

- Direct binary classification overfits or collapses because positives are rare.
- The direct probability head is less reliable than anatomical risk derived from predicted AS-OCT biomarkers.
- Report confidence intervals prominently; do not overstate any single locked-test estimate.

## Baseline Decision

Best current baseline for the paper:

**Image-only multitask anatomical model**  
Path: `paper2_runs/angle_closure_image_only_multitask/`

Rationale:

- Uses slit-lamp images only; no metadata shortcut.
- Learns AS-OCT anatomy as auxiliary supervision.
- Has the best current direct angle-closure test AUROC among clean model runs.
- More importantly, its predicted anatomical biomarkers classify angle closure better than the direct probability head.

Direct classifier comparison at validation-selected Youden threshold:

| Run | Test AUROC | Sensitivity | Specificity | PPV | NPV | TP/FP/TN/FN |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Pure image classifier | 0.317 | 0.250 | 0.630 | 0.069 | 0.885 | 2/27/46/6 |
| Image-only multitask | 0.663 | 0.875 | 0.452 | 0.149 | 0.971 | 7/40/33/1 |
| Old image+metadata multitask | 0.560 | 0.625 | 0.548 | 0.132 | 0.930 | 5/33/40/3 |

Best anatomical risk signal from the image-only multitask run:

| Risk Score | Test AUROC | Sensitivity | Specificity | PPV | NPV | TP/FP/TN/FN |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Low predicted nasal TISA500 | 0.815 | 1.000 | 0.603 | 0.216 | 1.000 | 8/29/44/0 |
| Low predicted ACD | 0.724 | 1.000 | 0.260 | 0.129 | 1.000 | 8/54/19/0 |
| Direct probability head | 0.663 | 0.875 | 0.452 | 0.149 | 0.971 | 7/40/33/1 |

Conclusion:

The most defensible baseline is not "direct glaucoma classifier." It is:

> Slit-lamp image-only multitask prediction of AS-OCT anterior chamber biomarkers, followed by a validation-selected anatomical risk rule for gonioscopic angle-closure referral triage.

After the latest fixed-split experiments, this remains the best baseline. Two attempted improvements did not beat it:

- `angle_closure_van_temporal_angle4_regstrong`: temporal Van Herick only, angle biomarkers only, stronger Huber regression. It failed on test classification.
- `angle_closure_usable_anatomy6_mean_regbalanced`: usable multi-view, mean pooling, six anatomy targets, Huber loss. It improved some regression metrics but did not improve angle-closure classification.
- `angle_closure_roi_local_combined_aod_tisa_sharedsplit`: valid shared-split beam-ROI local nasal/temporal AOD/TISA biomarker model. Best validation-selected shallow model used temporal AOD/TISA plus clinical metadata. Test result: AUROC 0.632, sensitivity 0.500, specificity 0.726.
- `angle_closure_roi_local_combined_aod_tisa_acd_lv_sharedsplit`: same ROI local model plus auxiliary ACD/lens-vault heads. Test result: AUROC 0.540, sensitivity 0.333, specificity 0.661.

This means the next efficient path is not another small multitask-loss tweak. The bottleneck is stronger image-to-angle-biomarker prediction and better use of peripheral angle crops/views.

Oracle check:

- On the corrected shared split, measured AS-OCT true AOD/TISA selected on validation reached validation sensitivity 0.833 and specificity 0.754, still short of 80/80.
- The same true-anatomy model achieved test sensitivity 0.833 and specificity 0.710 when selected by validation.
- A test-ranked true-anatomy/clinical row can exceed 80/80, but that is diagnostic only and confirms that anatomical signal exists; it is not a selectable model result.

## Step-by-Step Approach

### Step 1: Lock the Claim

Use this wording:

- Angle-closure / narrow-angle referral triage.
- Gonioscopic angle-closure screening.
- Slit-lamp-to-AS-OCT anatomical biomarker prediction.

Avoid:

- General glaucoma diagnosis.
- Deployable glaucoma screening.
- AS-OCT or gonioscopy replacement.

### Step 2: Freeze the Clean Baseline

Baseline model:

```bash
python3 code/train_angle_closure_multitask.py \
  --outdir ../paper2_runs/angle_closure_image_only_multitask \
  --focal-alpha 0.75 \
  --amp
```

This is image-only input with auxiliary AS-OCT regression heads. It is the current reference model.

### Step 3: Make Anatomical Risk the Primary Classifier

Evaluate angle closure using predicted biomarkers from the image-only multitask model:

- Primary anatomical score: low predicted nasal TISA500.
- Secondary scores: low predicted ACD, low predicted nasal AOD500, high predicted lens vault.
- Thresholds must be selected on validation only and applied once to the locked test set.

Report the direct neural probability head as an ablation, not the main claim, unless it improves.

### Step 4: Add View-Specific Baselines

Run the same image-only multitask setup separately for:

- `center`
- `van_nasal`
- `van_temporal`
- `usable` multi-view

Goal:

- Determine whether Van Herick views carry most of the angle signal.
- Reduce noisy bags if center views dilute peripheral angle information.

### Step 5: Improve the Loss Without Adding Metadata

Try only focused changes:

- Reduce or remove direct classification head weight if biomarker-derived risk is stronger.
- Train biometric-only first, then fit a simple validation-calibrated risk rule.
- Compare focal BCE vs class-weighted BCE only for the direct head.
- Keep patient-level split fixed.

Do not start broad architecture search yet.

Update after fixed-split search:

- Simple loss reweighting and target-subset changes have already been tested and did not reach the >0.80 / >0.80 goal.
- Next image-only work should move to dedicated single-target or small-target biomarker predictors for AOD/TISA, ideally with peripheral/Van Herick ROI emphasis and prediction CSVs that can feed the meta-risk evaluator.
- Do not keep spending compute on minor ConvNeXt multitask variants unless they change the information available to the model.

### Step 6: Add Metadata Only After Image-Only Is Frozen

Metadata model should be a secondary comparison:

- age
- sex
- IOP
- lens status
- CCT

Accept metadata only if it improves locked-test AUROC/sensitivity-specificity tradeoff without obvious overfit.

### Step 7: Report Statistics Conservatively

Always report:

- AUROC/AUPRC.
- Sensitivity, specificity, PPV, NPV.
- Patient-cluster bootstrap CIs.
- Number of positive eyes and positive participants.
- Calibration and decision curve.

For the current split, emphasize that locked test has only 8 positive eyes from 4 positive participants.

### Step 8: What Would Make This Strong

Minimum success target for the current internal paper:

- AUROC >= 0.75 for angle closure or anatomical risk.
- Sensitivity >= 0.90 at a clinically acceptable referral burden.
- NPV high enough for triage framing.
- Clear statement that PPV is limited because prevalence is low.

Current closest result:

- Low predicted nasal TISA500 from the image-only multitask model: AUROC 0.815, sensitivity 1.000, specificity 0.603 on the locked test set.

This should be validated with bootstrapping and view-specific reruns before becoming the headline result.

Current practical next steps:

1. Keep `angle_closure_image_only_multitask` as the reference baseline.
2. Use `evaluate_angle_closure_meta_risk.py` for all shallow risk-rule comparisons.
3. For any local-view combined experiment, generate and reuse a single patient split CSV across nasal and temporal training.
4. Do not advance the current image-only or ROI local candidates as stable 70/70 classifiers; both failed patient-level 5-fold validation.
5. The next model change should improve the underlying clinical signal: more positive cases, explicit Van Herick geometry extraction, or a stronger dedicated predictor for measured ACD/lens-vault/AOD/TISA. Do not spend more time on minor threshold or ConvNeXt head tweaks.
