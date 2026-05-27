# Agent Plan: Clinical Slit-Lamp-to-AS-OCT Paper

This note is for Codex/agent use inside `slit-project`. It documents the current project state, the intended second paper, and the next work to run. Treat it as the local planning anchor for the clinical triage validation paper after the Scientific Data/Data Descriptor paper.

## Decision lock (read first)

Paper 2 is a clinical triage validation paper, not a general glaucoma diagnosis paper and not a pure ML benchmark. The target claim is:

> Slit-lamp photographs can estimate AS-OCT-derived ACD and anterior chamber / angle biomarkers to support angle-closure or narrow-angle referral triage.

Update after angle-closure screening runs on 2026-05-25:

- The best current angle-closure baseline is the image-only multitask anatomical model in `paper2_runs/angle_closure_image_only_multitask/`.
- Pure image-only binary classification collapsed to near-constant probabilities and should not be the headline model.
- The direct probability head from the image-only multitask model is usable only as an ablation; predicted anatomical biomarkers are stronger for angle-closure triage.
- Current strongest test signal is low predicted nasal TISA500 from the image-only multitask model: AUROC 0.815, sensitivity 1.000, specificity 0.603 at the validation-selected Youden threshold.
- The active label is strict Shaffer grade `0/1` positive versus grade `2/3/4` negative. The temporary grade `0/1/2` positive branch is ignored.
- Do not run 5-fold validation during routine search. Use the fixed patient-level split; reserve 5-fold only for the final selected approach.
- Additional fixed-split attempts did not beat the baseline:
  - `angle_closure_van_temporal_angle4_regstrong`
  - `angle_closure_usable_anatomy6_mean_regbalanced`
- Best honest validation-selected image-only meta-risk result so far: sensitivity 0.875, specificity 0.630.
- Best diagnostic-only test-ranked image-only row so far: sensitivity 0.875, specificity 0.699; do not use this as a final selected model without fresh locked evaluation.
- Keep the paper framed as angle-closure / narrow-angle referral triage. Do not call this general glaucoma classification unless optic nerve/RNFL/visual-field glaucoma labels are added.
- Detailed strategy note: `ANGLE_CLOSURE_STRATEGY_REVIEW.md`.
- Consolidated comparison table: `paper2_runs/angle_closure_screening/baseline_search_summary.md`.

Current route:

- Build a TVST / IOVS-ready clinical validation paper first.
- Treat npj Digital Medicine as a conditional upgrade path only if a true external validation cohort, site/device shift, or prospective validation set becomes available.
- Do not submit this as an npj Digital Medicine "clinical translation" paper if all evidence remains internal held-out validation plus stress tests.
- Keep all wording in the triage/referral-support frame. Do not claim diagnosis, treatment recommendation, CDSS, or general glaucoma screening.

## Core framing

Working title, internal:

> Clinical validation of slit-lamp-to-AS-OCT prediction for low-resource angle-closure referral triage

Use safer clinical wording:

> Slit-lamp photographs can estimate AS-OCT-derived anterior chamber biomarkers to triage eyes for gonioscopy, AS-OCT, or ophthalmology referral.

Do not frame this as a general glaucoma diagnostic tool. The defensible clinical problem is primary angle-closure disease / narrow-angle risk, because the available labels are anterior chamber and angle biometrics rather than optic nerve or visual-field outcomes.

The journal strategy is:

- Paper 1: Scientific Data Data Descriptor for the paired dataset and release structure.
- Paper 2: TVST / IOVS-ready clinical validation and referral-triage paper, focused on slit-lamp-to-AS-OCT ACD / angle biomarker prediction.
- Paper 2 upgrade path: npj Digital Medicine only after external/site/device/prospective validation is available and locked-in.
- Paper 3: TVST / IOVS / methods-style multimodal benchmark paper, focused on fusion and anterior chamber biometry.

## What I learned from this folder

Current assets:

- `data/center_roi_images/data_trainval_set.csv`: 476 eye-level rows, 243 participants, paired AS-OCT biometrics.
- `data/center_roi_images/data_test_set.csv`: 86 eye-level rows, 43 participants, held-out test-style split.
- `code/ready_for_training_clustered_anatomical_with_means_with_views_anonymized.csv`: 15,912 slit-lamp image rows, 555 eye records, 283 participants.
- View labels in that table: `van_nasal` 6,144; `van_temporal` 5,383; `center` 3,769; `other` 357; `no_slit` 259.
- `code/ready_for_upload_publish.csv`: publish-oriented table, 531 eye records, 268 participants, 71 columns.
- `labeling_readme.md`: active-learning and manual view-labeling documentation.
- `NOTES_MODELS.md`: previous model plan and result notes.
- `code/fusion_acd_baseline.py`: strongest reusable training script for ACD; supports grouped splitting, `--external-test-csv`, `--cv-folds`, tabular features, prediction CSV output, and attention output.
- `code/eval_suite.py`: generic regression metric helper, but it does not yet include cluster bootstrap, Bland-Altman, CCC, decision curves, calibration, or classification operating points.
- `code/worst_case_audit.py`: useful for error-case review and image-row exports.
- `attention_maps/`: existing attention map examples.

Important existing ACD results from current CSV/log inspection:

- `runs/cv_A0_img_only/cv_metrics.csv`: 5-fold validation mean MAE 0.2085 mm, RMSE 0.2698 mm, Pearson r 0.7817.
- `runs/cv_A3_img_iop/cv_metrics.csv`: 5-fold validation mean MAE 0.2179 mm, RMSE 0.2801 mm, Pearson r 0.7637.
- `runs/cv_acd_best_optuna/cv_metrics.csv`: 5-fold validation mean MAE 0.2240 mm, RMSE 0.2908 mm, Pearson r 0.7455.
- `runs/preds_A0_img_only_sep_test_e50_p6/test_predictions.csv`: held-out test n=85, MAE 0.2541 mm, RMSE 0.3604 mm, Pearson r 0.6397, bias -0.0827 mm.
- `runs/preds_A2_img_age_sex_sep_test_e50_p6/test_predictions.csv`: held-out test n=85, MAE 0.2643 mm, RMSE 0.3559 mm, Pearson r 0.6960, bias -0.1366 mm.
- `runs/preds_A9_img_all_tab_separate_test_20260216_165215/test_predictions.csv`: held-out test n=84, MAE 0.3334 mm, RMSE 0.4254 mm, Pearson r 0.6605, bias -0.0142 mm.

Paper 2 evidence package generated on 2026-04-14:

- Script: `code/paper2_clinical_eval.py`
- Split manifest: `paper2_runs/split_manifest.csv`
- Split summary: `paper2_runs/split_manifest_summary.md`
- Clinical evaluation outputs: `paper2_runs/clinical_eval/`
- Participant-level split check: train/validation participants = 243, test participants = 43, participant overlap = 0, combo overlap = 0.
- A0 image-only held-out evaluation with participant-cluster bootstrap: n=85 eyes, 43 participants, MAE 0.2541 mm (95% CI 0.1964 to 0.3181), RMSE 0.3604 mm (95% CI 0.2762 to 0.4453), bias -0.0827 mm (95% CI -0.1771 to -0.0019), Pearson r 0.6397 (95% CI 0.3776 to 0.7944), CCC 0.5379 (95% CI 0.3264 to 0.6800).
- A2 image+age/sex held-out evaluation with participant-cluster bootstrap: n=85 eyes, 43 participants, MAE 0.2643 mm (95% CI 0.2115 to 0.3181), RMSE 0.3559 mm (95% CI 0.2816 to 0.4281), bias -0.1366 mm (95% CI -0.2173 to -0.0549), Pearson r 0.6960 (95% CI 0.4958 to 0.8280), CCC 0.5815 (95% CI 0.4026 to 0.7153).
- Manuscript draft updated: `MANUSCRIPT2_DRAFT.md`

Interpretation:

- The strongest current evidence is that slit-lamp images contain measurable ACD signal, but held-out test performance is not yet a finished clinical-translation package.
- Tabular additions are not consistently improving held-out ACD performance in the inspected runs.
- The folder already supports patient-level / participant-level thinking, but the next paper needs a formal split-leakage audit and participant-clustered uncertainty.
- The first clinical evaluation package now exists, but the manuscript still needs literature-justified thresholds, worst-case image review, final reference insertion, and external validation before any npj-level clinical-translation claim.

## Paper 2 research question

Primary question:

> Can slit-lamp images estimate AS-OCT-derived ACD and triage shallow anterior chamber / narrow-angle risk in a way that is clinically useful for low-resource referral workflows?

Primary endpoint:

- Continuous ACD[Endo.] prediction from slit-lamp images.

Secondary endpoints:

- Shallow ACD classification at literature-justified and sensitivity-analysis thresholds.
- Narrow-angle risk classification from AS-OCT-derived angle metrics if usable thresholds are available.
- AOD500 / AOD750 / TISA500 / TISA750 / TIA500-derived tasks if the label quality and missingness are acceptable.
- Coverage after QC gating: performance when poor-quality or non-slit images are rejected.

Clinical action:

- Low risk: routine follow-up.
- Intermediate risk or high uncertainty: repeat image or obtain AS-OCT/gonioscopy.
- High risk: refer for ophthalmology/gonioscopy/AS-OCT.

## Required methodological upgrades

Before claiming clinical translation, add these pieces:

- Split integrity audit: participant-level disjoint train/validation/test; both eyes from a participant must remain in the same split for inference-level claims.
- Clustered uncertainty: cluster bootstrap by participant for MAE, RMSE, Pearson r, AUC, sensitivity, specificity, and calibration metrics.
- Agreement analysis for ACD: Bland-Altman bias and limits of agreement, plus concordance correlation coefficient.
- Screening analysis: convert ACD / angle metrics into triage classifications and report operating points, especially high-sensitivity thresholds.
- Decision-curve analysis: estimate net benefit across plausible referral thresholds.
- Calibration and uncertainty: calibration curve / Brier score for classification and prediction intervals or uncertainty bins for ACD.
- QC gate: classify `center`, `van_nasal`, `van_temporal` as usable views; treat `other` and `no_slit` as reject or separate failure modes; report safety-coverage tradeoff.
- Subgroup reporting: age, sex, baseline ACD strata, view type, number of views, and available device/acquisition metadata.
- XAI sanity: do not rely on heatmaps alone; pair attention maps with worst-case audit and perturbation/sanity checks where possible.
- Reproducibility: fixed split manifest, public derived labels/splits/eval scripts, controlled-access image path, and explicit code availability statement.

## Exact next work to run

Do not start with more architecture search. First make the evidence package reviewer-proof.

1. Freeze the participant-level split manifest before more training.

Expected manifest output, when this is implemented:

```text
paper2_runs/split_manifest.csv
paper2_runs/split_manifest_summary.md
```

Minimum fields:

- participant ID
- eye ID / eye side
- split name
- image count
- usable view count
- ACD availability
- angle-label availability

Acceptance criterion: train/validation/test must be participant-disjoint. Both eyes from one participant must remain in the same inference split.

2. Create a paper-2 run folder:

```bash
mkdir -p paper2_runs
```

3. Verify split integrity:

```bash
python3 - <<'PY'
import pandas as pd
trainval = pd.read_csv('data/center_roi_images/data_trainval_set.csv')
test = pd.read_csv('data/center_roi_images/data_test_set.csv')
tv = set(trainval['patient_id'].dropna().astype(int))
te = set(test['patient_id'].dropna().astype(int))
print('trainval participants', len(tv))
print('test participants', len(te))
print('overlap participants', sorted(tv & te)[:20], 'n=', len(tv & te))
PY
```

Expected acceptance criterion: overlap participants must be zero.

Local check on 2026-04-14: trainval participants = 243, test participants = 43, overlap participants = 0; trainval rows = 476, test rows = 86.

4. Freeze the ACD image-only baseline as the clinical reference model:

```bash
cd code
conda activate awg
python3 fusion_acd_baseline.py \
  --source-csv ../data/center_roi_images/data_trainval_set.csv \
  --external-test-csv ../data/center_roi_images/data_test_set.csv \
  --method mil \
  --backbone resnet50 \
  --epochs 50 \
  --patience 6 \
  --loss huber \
  --huber-delta 1.0 \
  --mixup-alpha 0.2 \
  --freeze-epochs 1 \
  --unfreeze-lr-factor 0.5 \
  --preds-outdir ../paper2_runs/acd_a0_image_only/preds \
  --checkpoint ../paper2_runs/acd_a0_image_only/model.pth \
  --scaler-path ../paper2_runs/acd_a0_image_only/scaler.npz \
  --test-scatter ../paper2_runs/acd_a0_image_only/test_scatter.png
```

5. Run one lean tabular comparison, not a wide ablation:

```bash
cd code
conda activate awg
python3 fusion_acd_baseline.py \
  --source-csv ../data/center_roi_images/data_trainval_set.csv \
  --external-test-csv ../data/center_roi_images/data_test_set.csv \
  --method mil \
  --backbone resnet50 \
  --use-age \
  --use-sex \
  --epochs 50 \
  --patience 6 \
  --loss huber \
  --huber-delta 1.0 \
  --mixup-alpha 0.2 \
  --freeze-epochs 1 \
  --unfreeze-lr-factor 0.5 \
  --preds-outdir ../paper2_runs/acd_a2_age_sex/preds \
  --checkpoint ../paper2_runs/acd_a2_age_sex/model.pth \
  --scaler-path ../paper2_runs/acd_a2_age_sex/scaler.npz \
  --test-scatter ../paper2_runs/acd_a2_age_sex/test_scatter.png
```

6. Build a clinical evaluation script next:

Suggested new script:

```text
code/paper2_clinical_eval.py
```

Inputs:

- prediction CSV with `combo_key`, `true_ACD[Endo.]`, `pred_ACD[Endo.]`
- source metadata CSV with participant ID, eye, age, sex, ACD, AOD/TISA/TIA if used
- optional view-quality table

Outputs:

- `metrics_cluster_bootstrap.csv`
- `bland_altman_acd.csv`
- `triage_threshold_metrics.csv`
- `decision_curve.csv`
- `subgroup_metrics.csv`
- `qc_coverage_metrics.csv`
- `paper2_results_summary.md`

Required analyses in that script:

- participant-clustered bootstrap confidence intervals;
- Bland-Altman bias and limits of agreement;
- concordance correlation coefficient;
- ACD triage threshold metrics;
- calibration and Brier score for binary triage endpoints;
- decision curve analysis across referral thresholds;
- QC coverage / refuse-to-predict behavior;
- subgroup metrics by age, sex, eye, view label, and baseline ACD strata.

7. Run worst-case audit on the frozen baseline:

```bash
cd code
conda activate awg
python3 worst_case_audit.py \
  --preds ../paper2_runs/acd_a0_image_only/preds/test_predictions.csv \
  --target-col 'ACD[Endo.]' \
  --topk 20 \
  --view-filter center \
  --outdir ../paper2_runs/acd_a0_image_only/worst_cases \
  --row-images
```

8. Decide external validation path before selecting npj as the submission target:

- Best: obtain a second device/site/prospective cohort and run locked inference.
- Acceptable interim: run a device/view/domain-shift stress test and explicitly frame npj as preliminary until external validation exists.
- Do not overclaim "clinical translation" if only the current internal held-out test split is available.

## Paper 3 benchmark track

Working title:

> Benchmarking multimodal fusion on a paired slit-lamp/AS-OCT dataset for anterior chamber depth and angle parameters

Minimum benchmark matrix:

- Slit-lamp only.
- AS-OCT-derived/tabular metadata only.
- Slit-lamp + age/sex.
- Slit-lamp + IOP or other clinical metadata only if useful.
- Slit-lamp + AS-OCT labels/fusion, if image-level AS-OCT inputs are prepared.
- Missing-modality stress test.
- ACD, AOD/TISA/TIA endpoint comparison.

This is better for TVST / IOVS / methods venues than npj unless it includes clinical workflow and validation.

Keep Paper 3 separate from Paper 2:

- Paper 2 asks whether a slit-lamp-only workflow can support clinical triage against AS-OCT-derived biomarkers.
- Paper 3 asks how much multimodal fusion improves anterior segment biomarker prediction and benchmarking.
- Do not mix Paper 3's broad fusion matrix into Paper 2 until the Paper 2 clinical evaluation package is stable.

## Manuscript skeleton for Paper 2

Introduction:

- Burden and clinical logic of angle-closure risk.
- Slit-lamp is common and lower-cost; AS-OCT is quantitative but less available.
- Paired dataset enables AS-OCT-derived supervision from slit-lamp images.

Methods:

- Dataset, participants, eye-level records, image counts, pairing, and governance.
- Endpoint definitions: ACD primary; shallow/narrow risk secondary.
- Participant-level splitting and inter-eye correlation handling.
- Model stack: image-only baseline, age/sex comparison, optional foundation model later.
- QC gate and reject policy.
- Statistical analysis: cluster bootstrap, agreement analysis, triage metrics, decision curves.

Results:

- Dataset flow and split table.
- ACD regression with clustered confidence intervals.
- Bland-Altman and CCC.
- Triage operating points.
- QC safety-coverage analysis.
- Subgroup/failure analysis.

Discussion:

- What is clinically promising.
- Why it is a triage aid, not diagnosis.
- Generalization and external validation gap.
- Low-resource workflow path.
- Data/code availability and controlled-access imaging.

## Do not overclaim

Use these phrases:

- "referral triage"
- "angle-closure / narrow-angle risk"
- "AS-OCT-derived anterior chamber biomarkers"
- "decision-support evidence is preliminary without external validation"

Avoid these phrases unless future evidence directly supports them:

- "diagnoses glaucoma"
- "treatment recommendation"
- "clinical decision support system"
- "deployable screening tool"
- "npj-ready clinical translation"
- "general glaucoma screening"

If the only validation remains the current internal held-out test set, frame the paper as TVST/IOVS clinical validation or biomarker prediction, not as deployment-ready digital medicine.

## Hard stop criteria (prominent)

Do not submit as npj Digital Medicine clinical translation if:

- participant-level split leakage exists;
- no uncertainty or cluster bootstrap is reported;
- only AUROC/correlation is reported;
- there is no QC/refuse-to-predict behavior;
- narrow-angle labels are not defensibly defined;
- there is no external validation or clearly labeled external-validation plan.

If those blockers remain, aim for TVST/IOVS validation or a benchmark paper first, then return to npj after external/site/device validation.
