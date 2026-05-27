# Manuscript 2 Draft: Clinical Slit-Lamp-to-AS-OCT Triage Validation

Draft status: working draft for internal use.

Target route: TVST / IOVS-ready clinical validation paper first. npj Digital Medicine is a conditional upgrade path only if true external validation, device/site shift validation, or prospective validation becomes available.

Claim boundary: This manuscript is about referral triage for angle-closure / narrow-angle risk using AS-OCT-derived anterior chamber biomarkers. It is not a general glaucoma diagnostic paper, treatment recommendation paper, or clinical decision support system claim.

## Working Title

Clinical Validation of Slit-Lamp-to-AS-OCT Prediction for Low-Resource Angle-Closure Referral Triage

Alternative titles:

- Slit-Lamp Prediction of AS-OCT-Derived Anterior Chamber Depth for Angle-Closure Referral Triage
- Estimating AS-OCT-Derived Anterior Chamber Biomarkers from Slit-Lamp Images for Low-Resource Triage
- Slit-Lamp-to-AS-OCT Anterior Chamber Biometry as a Referral-Triage Tool for Narrow-Angle Risk

Short title:

Slit-Lamp-to-AS-OCT Triage Validation

## Target Journal Strategy

Primary realistic target:

- TVST or IOVS, framed as a clinically oriented validation and biomarker prediction paper.

Conditional upgrade target:

- npj Digital Medicine only if an external validation cohort, prospective validation set, or convincing device/site shift evaluation is added before submission.

Do not submit as npj clinical translation if the evidence package is limited to internal held-out validation and stress testing.

## Structured Abstract Draft

### Background

Primary angle-closure disease and narrow anterior chamber angle risk require timely identification, but quantitative AS-OCT imaging is not always available in low-resource or community settings. Slit-lamp photography is more accessible, but conventional assessment is qualitative and operator dependent. We evaluated whether slit-lamp photographs can estimate AS-OCT-derived anterior chamber biomarkers and support referral triage for angle-closure / narrow-angle risk.

### Methods

We used a paired multimodal anterior segment imaging dataset containing slit-lamp photographs, eye-level gonioscopic Shaffer angle grades, AS-OCT images, and linked biometric metadata. The primary endpoint was gonioscopic angle closure, defined as an eye-level Shaffer grade of 0 or 1. Eyes with missing, not-seen, or indeterminate angle grades were excluded from the primary analysis. Secondary endpoints included AS-OCT biometric regression for anterior chamber and angle parameters. Models were trained and evaluated with participant-level split separation to avoid leakage between eyes or repeated images from the same participant. The clinical evaluation plan included classification metrics, validation-selected operating points, participant-clustered bootstrap confidence intervals, calibration, decision curve analysis, quality-control gating, subgroup performance, and biometric regression metrics.

### Results

In the current local project state, the train/validation table contains 476 eye-level rows from 243 participants and the held-out test table contains 86 eye-level rows from 43 participants, with no participant overlap in the inspected split. A larger image-level training table contains 15,912 slit-lamp image rows from 283 participants and includes view labels for center, nasal, temporal, other, and no-slit images. Preliminary existing runs show that an image-only ACD model achieves cross-validation mean MAE of 0.2085 mm and held-out test MAE of 0.2541 mm, but these results do not yet include the final clinical evaluation package required for submission. Final results will report participant-clustered confidence intervals, agreement statistics, triage performance, QC-gated performance, and subgroup analyses.

### Conclusions

Slit-lamp photographs appear to contain measurable signal for angle-closure / narrow-angle screening and AS-OCT biometric estimation. The intended clinical role is referral triage, not standalone glaucoma diagnosis. External validation remains required before making deployment-ready or npj-level clinical translation claims.

## Keywords

Slit-lamp photography; anterior segment optical coherence tomography; anterior chamber depth; angle closure; narrow angle; referral triage; ophthalmic artificial intelligence; low-resource screening; clinical validation; AS-OCT.

## Key Points

- Question: Can slit-lamp photographs screen for eye-level gonioscopic angle closure and estimate AS-OCT-derived anterior chamber biomarkers to support referral triage?
- Findings: Preliminary local runs show measurable ACD prediction signal, and the implemented Paper 2 pipeline now supports eye-level angle-closure classification with participant-level splitting, QC gating, calibration, bootstrap uncertainty, decision curves, and subgroup analysis.
- Meaning: The model should be framed as a triage-support approach for identifying eyes needing repeat imaging, gonioscopy, AS-OCT, or ophthalmology referral, not as a diagnosis or treatment tool.

## Main Text Draft

## 1. Introduction

Primary angle-closure disease remains an important cause of preventable visual morbidity, particularly in populations where access to specialist ophthalmic imaging is limited. Quantitative anterior segment imaging can help characterize anterior chamber depth and angle anatomy, but AS-OCT systems are less available than slit-lamp examination in many clinical and community workflows. As a result, there is a practical need for methods that can extract more quantitative triage information from lower-cost and more widely available anterior segment imaging.

Slit-lamp photography is widely used, but conventional interpretation of anterior chamber configuration is qualitative and dependent on acquisition quality and examiner experience. AS-OCT provides quantitative anterior segment metrics, including ACD, AOD, TISA, and TIA, that can be used to characterize anterior chamber anatomy. A paired slit-lamp and AS-OCT dataset creates a cross-modality learning setting: slit-lamp images can be supervised by AS-OCT-derived biometric labels, making it possible to test whether lower-resource imaging contains clinically useful signal for referral triage.

The goal of this study is not to develop a general glaucoma diagnostic system. The available labels are eye-level anterior chamber angle grade and anterior segment biometrics, not optic nerve, retinal nerve fiber layer, or visual field outcomes. Therefore, the defensible clinical use case is angle-closure / narrow-angle referral triage: identifying eyes that may require repeat imaging, gonioscopy, AS-OCT, or ophthalmology referral.

We evaluated slit-lamp-based screening for eye-level gonioscopic angle closure as the primary task, with secondary multitask prediction of AS-OCT anterior chamber biomarkers. This structure is intended to support a TVST / IOVS-ready clinical validation manuscript now, with npj Digital Medicine considered only if external validation becomes available.

## 2. Methods

### 2.1 Study Design and Reporting

This is a retrospective paired-imaging model development and validation study using slit-lamp photographs, eye-level gonioscopic angle grades, and AS-OCT-derived anterior segment biomarkers. The analytic goal is to evaluate whether slit-lamp images can screen for gonioscopic angle closure and estimate anterior chamber biomarkers to support referral triage for narrow-angle / angle-closure risk.

Reporting should follow TRIPOD+AI for prediction model reporting and CLAIM for imaging AI completeness. The manuscript should also include a transparent data/code availability statement and a controlled-access imaging data pathway if imaging cannot be publicly released.

### 2.2 Dataset and Participants

The local project contains multiple analytic tables:

- `data/center_roi_images/data_trainval_set.csv`: 476 eye-level rows from 243 participants.
- `data/center_roi_images/data_test_set.csv`: 86 eye-level rows from 43 participants.
- `code/ready_for_training_clustered_anatomical_with_means_with_views_anonymized.csv`: 15,912 slit-lamp image rows from 283 participants.
- `code/ready_for_upload_publish.csv`: 531 publication-oriented eye-level records from 268 participants.

The train/validation and held-out test split inspected on 2026-04-14 had no participant overlap: 243 train/validation participants and 43 test participants, overlap n = 0. Final manuscript counts must be regenerated from the frozen split manifest before submission.

### 2.3 Imaging and Label Sources

Slit-lamp image records include image paths, eye labels, acquisition metadata where available, and view labels. The current view-label distribution in the image-level table is:

- `van_nasal`: 6,144 images.
- `van_temporal`: 5,383 images.
- `center`: 3,769 images.
- `other`: 357 images.
- `no_slit`: 259 images.

The primary reference label is the eye-level gonioscopic Shaffer angle grade recorded in the cleaned clinical table. Angle closure was defined as an eye-level Shaffer angle grade of 0 or 1. Eyes marked as not seen, missing, or indeterminate were excluded from the primary analysis. The current implementation does not use quadrant-resolved gonioscopy; a quadrant-based rule can be added as a sensitivity analysis if superior, inferior, nasal, and temporal grades become available.

AS-OCT-derived labels include ACD[Endo.] and multiple angle-related metrics, including AOD250/500/750, ARA250/500/750, TISA250/500/750, and TIA250/500/750 for nasal and temporal sides in the cleaned tables. These biomarkers are secondary multitask regression targets.

### 2.4 Primary and Secondary Endpoints

Primary endpoint:

- Binary gonioscopic angle-closure screening from slit-lamp images, where angle closure is defined as eye-level Shaffer grade 0 or 1.

Secondary endpoints:

- AS-OCT biometric regression for ACD[Endo.], AOD500, TISA500, lens vault, and central corneal thickness.
- QC-gated performance after rejecting poor-quality or non-slit images.
- Subgroup performance by age, sex, eye side, view label, baseline ACD strata, and acquisition/device metadata if available.

The primary operating threshold is selected and evaluated on the validation split using the Youden index. A high-specificity threshold targeting 95% specificity on the validation split is reported as a prespecified secondary operating point. Because the current cohort is small and closed-angle cases are rare, the fixed-split analysis uses patient-level train/validation separation only rather than reserving a small third internal test partition.

### 2.5 Model Inputs and Baselines

The first manuscript version should avoid broad architecture search and focus on a reviewer-proof clinical evaluation package.

Initial baseline models:

- S0: single-view center slit-lamp image MIL model.
- M0: multi-view MIL model using usable slit-lamp views (`center`, `van_nasal`, `van_temporal`).
- M1: multi-view MIL model plus clinical metadata.

The implemented model uses a ConvNeXt-Tiny image encoder pretrained on ImageNet, attention pooling across slit-lamp views, an optional metadata MLP for age, sex, intraocular pressure, lens status, and CCT, a binary classification head for angle closure, and secondary regression heads for AS-OCT biometrics. Classification is trained with focal binary cross-entropy, and regression heads are trained with masked MSE for available biometric labels. The final comparison should focus on the prespecified single-view vs multi-view vs multi-view-plus-metadata matrix, rather than broad architecture search.

### 2.6 Split Strategy and Leakage Control

All model development and validation must use participant-level separation. Both eyes and all repeated images from the same participant must remain in the same split for inference-level claims.

The active fixed-split manuscript workflow uses an 80/20 participant-level train/validation split, stratified by participant-level angle-closure status. This validation set is the internal validation cohort for threshold selection and performance reporting. Patient-level cross-validation can be reported as a robustness analysis, but no separate internal test set is reserved in the current primary method.

Required output before final analysis:

- `paper2_runs/angle_closure_screening/split_manifest.csv`
- `paper2_runs/angle_closure_screening/split_manifest_summary.md`

The implemented manifest includes participant ID, eye side, split name, angle grade, closure label, image count, usable view count, AS-OCT target availability, and clinical metadata fields. Final train/validation counts must be regenerated from the frozen 80/20 manifest before manuscript submission.

### 2.7 Statistical Analysis

Classification / triage performance should include:

- AUROC and AUPRC.
- Sensitivity, specificity, PPV, and NPV at prespecified referral thresholds.
- Patient-level bootstrap 95% confidence intervals using 2,000 resamples for AUROC, sensitivity, specificity, PPV, NPV, and accuracy.
- Sensitivity at the validation-selected threshold targeting 95% specificity.
- Probability calibration and Brier score.
- Decision curve analysis across referral thresholds.
- Model-variant comparisons using the DeLong test for paired AUCs and the Wilcoxon signed-rank test for paired continuous absolute ACD errors.
- Referral utility summaries reporting the proportion of open-angle eyes spared from unnecessary gonioscopy or AS-OCT and the number of missed angle-closure cases at the selected threshold.

Regression performance for secondary AS-OCT biomarker heads should include MAE, RMSE, bias, Pearson correlation, and R2 for each target with available labels.

All uncertainty estimates should account for participant-level clustering. Bootstrap resampling should resample participants, not individual eyes or image rows. The fixed-split analysis should be described as internal validation/model development unless a truly external or prospectively locked test cohort becomes available. Two-sided P values below 0.05 will be considered statistically significant, with exploratory wording for small subgroup and model-comparison analyses.

### 2.8 Quality-Control Gate and Refuse-to-Predict Policy

The primary QC gate should treat the following as usable views:

- `center`
- `van_nasal`
- `van_temporal`

The following should be rejected or analyzed as separate failure modes:

- `other`
- `no_slit`

The manuscript should report coverage and performance before and after QC gating. This is essential for a low-resource workflow because poor image quality is a likely deployment failure mode.

### 2.9 Subgroup and Fairness Analysis

Minimum subgroup analyses:

- Age younger than 60 years vs 60 years or older.
- Sex.
- Lens status: phakic vs pseudophakic when available.
- Eye side.
- View label.
- Baseline ACD strata.
- Number of available images per eye.

Add device, site, acquisition date, or temporal shift analysis if the metadata supports it. If subgroup sample sizes are small, report this explicitly and avoid strong fairness claims.

### 2.10 Explainability and Error Analysis

Do not rely on heatmaps alone. The XAI and failure analysis package should include:

- Worst-case audit of the largest absolute errors.
- Review of whether failures are associated with poor view label, limited image count, extreme ACD, or acquisition artifacts.
- Attention maps or saliency examples only as supportive figures.
- Perturbation or sanity-check analysis if feasible.

Optional interpretability analysis:

- Generate Grad-CAM maps for a random subset of locked test-set images, sampled separately from true positives, false positives, true negatives, and false negatives at the prespecified screening threshold.
- Compute the center of mass of each activation map and measure its distance from the limbus or Van Herick zone when limbal localization is available.
- Test whether attention is concentrated in the peripheral anterior chamber and correlate attention distribution with predicted angle-closure probability or the regression-derived risk score using Spearman rank correlation.

If limbus coordinates are not available or not reliable, report Grad-CAM as qualitative face-validity evidence only and do not claim mechanistic interpretability.

### 2.11 Data and Code Availability

The manuscript should state that de-identified derived tabular labels, split definitions, evaluation code, and baseline scripts will be made available as permitted. Imaging data can remain controlled-access under a DUA if required by ethics and privacy governance, but the access route must be clear.

Minimum reproducibility package:

- Frozen split manifest.
- Derived labels used in analysis.
- Prediction CSVs for final models.
- Clinical evaluation script.
- Figure/table generation scripts.
- Data dictionary.

Draft availability language:

> The de-identified tabular data and documentation are openly available on Zenodo (DOI: [insert]). Slit-lamp and AS-OCT images are available via controlled access on Synapse under a Data Use Agreement. All model code, training configurations, and evaluation scripts are publicly accessible at [GitHub repository URL]. The study was pre-registered on the Open Science Framework (OSF; [registration DOI]) before the final test set evaluation.

## 3. Results Draft

### 3.1 Cohort and Imaging Flow

Generated evidence package:

- Split manifest: `paper2_runs/angle_closure_screening/split_manifest.csv`
- Split summary: `paper2_runs/angle_closure_screening/split_manifest_summary.md`
- Clinical evaluation outputs: `paper2_runs/angle_closure_screening/clinical_eval/`

Draft result text:

> The angle-closure screening manifest included 529 linked analytic eyes from 267 participants after excluding missing or not-seen eye-level angle grades and requiring at least one usable slit-lamp image. The patient-level split contained 368 training eyes from 186 participants, 80 validation eyes from 40 participants, and 81 locked test eyes from 41 participants. There was no participant overlap across train, validation, and test splits.

Note: the broader image-level table contains 15,912 slit-lamp image rows from 283 participants; the manifest-linked count is lower because it is restricted to analytic eyes with valid angle labels and resolved image paths.

### 3.2 Angle-Closure Screening Performance

Generated by `code/train_angle_closure_multitask.py` after full model training:

- `classification_metrics.csv`
- `test_bootstrap_ci_youden.csv`
- `test_calibration.png`
- `test_decision_curve.csv`
- `test_subgroup_metrics.csv`

Draft result text:

> On the locked internal test set, the final multi-view model will be evaluated for eye-level gonioscopic angle closure using AUROC, AUPRC, Brier score, sensitivity, specificity, PPV, and NPV. The primary operating point will be selected on the validation split using the Youden index and applied once to the test split. A second operating point targeting 95% specificity on the validation split will be reported as a high-specificity screening threshold.

Current baseline decision:

- Pure image-only binary classifier: test AUROC 0.317, sensitivity 0.250, specificity 0.630. This model collapsed toward nearly constant probabilities and should not be the headline baseline.
- Image-only multitask anatomical model: test AUROC 0.663 for the direct probability head, sensitivity 0.875, specificity 0.452. This is the current clean model baseline because it uses image input only and learns AS-OCT anatomy as auxiliary supervision.
- Anatomical risk from the image-only multitask model is stronger than the direct probability head. Low predicted nasal TISA500 achieved test AUROC 0.815, sensitivity 1.000, specificity 0.603 at the validation-selected threshold. This should be treated as the leading angle-closure triage strategy pending bootstrap confirmation and view-specific reruns.

Updated fixed-split search note:

- The active label is strict angle closure: Shaffer grade 0/1 positive versus grade 2/3/4 negative. The temporary grade 0/1/2 positive definition is not used.
- The best honest validation-selected image-only meta-risk result currently reaches sensitivity 0.875 and specificity 0.630 on the locked test.
- The best diagnostic-only test-ranked image-only result reaches sensitivity 0.875 and specificity 0.699, but this is not a final selected model because it is sorted by test performance.
- Two targeted image-only improvements, temporal angle-only stronger regression and mean-pooled six-anatomy multitask training, did not improve locked-test classification.
- The current evidence supports angle-closure referral triage and biomarker prediction, but the >0.80 sensitivity / >0.80 specificity goal has not yet been reached honestly on this fixed split.

### 3.3 Secondary AS-OCT Biometric Regression Performance

Generated from existing prediction CSVs using `code/paper2_clinical_eval.py` with 1,000 participant-cluster bootstrap iterations:

| Model | n eyes | n participants | MAE mm (95% CI) | RMSE mm (95% CI) | Bias mm (95% CI) | Pearson r (95% CI) | CCC (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A0 image-only | 85 | 43 | 0.2541 (0.1964 to 0.3181) | 0.3604 (0.2762 to 0.4453) | -0.0827 (-0.1771 to -0.0019) | 0.6397 (0.3776 to 0.7944) | 0.5379 (0.3264 to 0.6800) |
| A2 image + age/sex | 85 | 43 | 0.2643 (0.2115 to 0.3181) | 0.3559 (0.2816 to 0.4281) | -0.1366 (-0.2173 to -0.0549) | 0.6960 (0.4958 to 0.8280) | 0.5815 (0.4026 to 0.7153) |

Draft result text:

> On the held-out internal test set, the A0 image-only model achieved MAE 0.2541 mm (95% CI 0.1964 to 0.3181), RMSE 0.3604 mm (95% CI 0.2762 to 0.4453), Pearson r 0.6397 (95% CI 0.3776 to 0.7944), and CCC 0.5379 (95% CI 0.3264 to 0.6800). The A2 image plus age/sex model achieved similar MAE 0.2643 mm (95% CI 0.2115 to 0.3181), RMSE 0.3559 mm (95% CI 0.2816 to 0.4281), Pearson r 0.6960 (95% CI 0.4958 to 0.8280), and CCC 0.5815 (95% CI 0.4026 to 0.7153).

Interpretation note: A2 improves correlation and CCC but has slightly higher MAE and more negative bias than A0. Do not overstate A2 as globally superior.

### 3.4 Agreement with AS-OCT-Derived ACD

Generated Bland-Altman outputs:

- A0 image-only: bias -0.0827 mm; limits of agreement -0.7743 to 0.6089 mm; CCC 0.5379.
- A2 image + age/sex: bias -0.1366 mm; limits of agreement -0.7845 to 0.5114 mm; CCC 0.5815.

Draft result text:

> Bland-Altman analysis showed a mean prediction bias of -0.0827 mm for the A0 image-only model, with limits of agreement from -0.7743 to 0.6089 mm. The A2 image plus age/sex model showed a mean bias of -0.1366 mm, with limits of agreement from -0.7845 to 0.5114 mm.

Do not write "clinically interchangeable with AS-OCT" unless agreement is strong and the claim is defensible.

### 3.5 Legacy ACD-Derived Triage Performance

Generated triage outputs:

For shallow ACD threshold 2.4 mm:

- A0 fixed predicted-ACD cutoff 2.4 mm: AUROC 0.8219, AUPRC 0.5849, sensitivity 0.4286, specificity 0.9718, PPV 0.7500, NPV 0.8961.
- A2 fixed predicted-ACD cutoff 2.4 mm: AUROC 0.8320, AUPRC 0.6018, sensitivity 0.5000, specificity 0.9014, PPV 0.5000, NPV 0.9014.

High-sensitivity operating point examples for shallow ACD threshold 2.4 mm:

- A0 at target sensitivity 0.90 used predicted ACD cutoff 2.7732 mm: sensitivity 0.9286, specificity 0.4648, PPV 0.2549, NPV 0.9706.
- A2 at target sensitivity 0.90 used predicted ACD cutoff 2.5464 mm: sensitivity 0.9286, specificity 0.7183, PPV 0.3939, NPV 0.9808.

Draft result text:

> For the 2.4 mm shallow-ACD threshold, discrimination based on negative predicted ACD was moderate, with AUROC 0.8219 for A0 and 0.8320 for A2. Using the same 2.4 mm cutoff on predicted ACD gave high specificity but limited sensitivity. When optimizing for high-sensitivity triage, A2 reached sensitivity 0.9286 and specificity 0.7183 at a predicted ACD cutoff of 2.5464 mm, with NPV 0.9808.

This section must make the clinical action explicit:

- Low predicted risk: routine follow-up.
- Intermediate risk or high uncertainty: repeat image or obtain AS-OCT/gonioscopy.
- High predicted risk: refer for ophthalmology/gonioscopy/AS-OCT.

### 3.6 Calibration and Decision Curve Analysis

Generated outputs:

- `test_calibration_bins.csv` contains probability calibration bins for the angle-closure classifier.
- `test_decision_curve.csv` contains probability-threshold decision curve analysis for the angle-closure classifier.

Draft result text:

> The angle-closure classifier outputs a probability score, allowing Brier score estimation, calibration plots, and decision curve analysis across referral thresholds. Calibration and net benefit should be interpreted as internal-validation evidence until external or prospective validation is available.

### 3.7 QC-Gated Performance

Generated QC outputs:

- A0 all held-out eyes: n = 85, MAE 0.2541 mm, RMSE 0.3604 mm.
- A0 with at least one usable view: n = 83, coverage 97.65%, MAE 0.2535 mm, RMSE 0.3621 mm.
- A2 all held-out eyes: n = 85, MAE 0.2643 mm, RMSE 0.3559 mm.
- A2 with at least one usable view: n = 83, coverage 97.65%, MAE 0.2654 mm, RMSE 0.3579 mm.

Draft result text:

> The simple view-label QC gate retained 83 of 85 held-out eyes (97.65%). QC gating produced minimal change in regression performance, suggesting that nearly all held-out eyes already had at least one usable view. The two eyes without a usable view should be treated as a failure-mode subset rather than used for stable subgroup performance estimates.

This section should present both safety and coverage. A model that improves performance by rejecting too many cases may not be practical for low-resource triage.

### 3.8 Subgroup and Failure Analysis

Generated failure analysis:

- Worst A0 case: `6_R`, true ACD 4.240 mm, predicted 2.8366 mm, error -1.4034 mm.
- Worst A2 case should be read from the legacy ACD evidence package; new angle-closure false-positive and false-negative reviews should be generated from `paper2_runs/angle_closure_screening/clinical_eval/test_predictions.csv`.
- A0 errors were higher in `acd_gt_3_0` and `acd_le_2_4` strata than in the middle ACD stratum, suggesting weaker performance at anatomical extremes.

Draft result text:

> Subgroup analysis suggested that errors were lower in the middle ACD stratum than at the shallow and deep ACD extremes. The largest A0 error occurred in eye `6_R`, where true ACD was 4.240 mm and predicted ACD was 2.8366 mm. These failures should be reviewed against the underlying images before making clinical workflow claims.

Avoid strong fairness claims if subgroup counts are small.

## 4. Discussion Draft

### Principal Findings

This study evaluates whether slit-lamp photographs can estimate AS-OCT-derived anterior chamber biomarkers for angle-closure / narrow-angle referral triage. Preliminary local results indicate that slit-lamp images contain measurable signal for ACD prediction, but final clinical interpretation requires participant-clustered uncertainty, agreement analysis, QC-gated evaluation, calibration, decision curve analysis, and subgroup reporting.

### Clinical Meaning

The intended use is not to replace AS-OCT or gonioscopy. Instead, the model is designed as a triage aid for settings where slit-lamp imaging is available but quantitative AS-OCT is not. A clinically realistic workflow would use model output to identify eyes that should receive repeat imaging, gonioscopy, AS-OCT, or ophthalmology referral.

### Relationship to Prior Work

Prior work has motivated ACD and angle biomarkers as clinically relevant for angle-closure risk assessment. The present study contributes a paired slit-lamp-to-AS-OCT evaluation using AS-OCT-derived biometric supervision and a clinical validation analysis plan. Final manuscript citations should be inserted for:

- Angle-closure burden and screening relevance.
- ACD and angle metrics as risk markers.
- Prior slit-lamp or smartphone anterior segment prediction work.
- Reporting standards for AI prediction models and imaging AI.
- Decision curve analysis and agreement analysis methodology.

### Safety and Reliability

Safety depends on more than test-set correlation. The manuscript should emphasize QC gating, uncertainty, calibration, high-sensitivity operating points, and subgroup failure analysis. If uncertainty is high or image quality is poor, the correct model behavior is to refuse prediction or recommend repeat imaging / confirmatory assessment.

### Limitations

Key limitations to state clearly:

- External validation is not yet available in the current local project state.
- The held-out test set is internal and may not capture device, site, population, or prospective workflow shift.
- A formal multi-reader clinician comparison is not included in the main analysis. If a small reader pilot is performed, it should be reported as exploratory supplementary evidence only.
- The model screens for eye-level angle closure and predicts AS-OCT-derived biomarkers; it does not diagnose glaucomatous optic neuropathy or visual field loss.
- The reference standard in the current implementation is an eye-level Shaffer angle grade, not quadrant-resolved gonioscopy. A quadrant-based closure definition requires additional quadrant labels.
- Subgroup analysis may be limited by sample size.
- Imaging data access may require controlled access under a DUA.

### Future Work

Future work should obtain an external validation cohort, ideally from a different site, device, or prospective low-resource workflow. After locked external validation, the manuscript can be reframed for a higher-threshold digital medicine venue. A formal multi-reader comparison with clinicians should be designed as a separate powered study; any small pilot reader study, such as two glaucoma specialists reading 30 test eyes, should be labeled exploratory and placed in the Supplementary Material. Multimodal fusion and foundation-model experiments should be handled as Paper 3 unless they directly improve the clinical triage evidence package.

## 5. Figures and Tables

### Figure 1. Clinical Triage Workflow

Slit-lamp image acquisition -> QC gate -> ACD / angle biomarker prediction -> risk stratum -> repeat imaging / AS-OCT / gonioscopy / ophthalmology referral.

### Figure 2. Dataset and Split Flow

Participants -> eyes -> slit-lamp images -> AS-OCT labels -> train/validation/test split -> final analytic cohorts.

### Figure 3. ACD Prediction and Agreement

Scatter plot of predicted vs true ACD, Bland-Altman plot, and error distribution.

### Figure 4. Triage Performance

ROC, precision-recall curve, calibration curve, and decision curve analysis for shallow-ACD / narrow-angle triage.

### Figure 5. QC and Failure Analysis

Performance by view label, QC-gated coverage, and representative worst-case errors.

### Table 1. Cohort Characteristics

Participant and eye-level characteristics by split: age, sex, eye side, ACD, angle metrics, image counts, view-label distribution.

### Table 2. Model Performance for ACD Prediction

Single center view vs multi-view vs multi-view+metadata, with MAE, RMSE, Pearson r, bias, CCC, and participant-clustered confidence intervals.

### Table 3. Triage Classification Performance

Thresholds, sensitivity, specificity, PPV, NPV, AUROC, AUPRC, patient-level bootstrap confidence intervals, calibration, referral rate, open-angle eyes spared, and missed angle-closure cases.

### Table 4. QC-Gated and Subgroup Performance

Performance by QC status, view label, age strata, sex, eye side, baseline ACD strata, and image-count strata.

## 6. Reporting Checklist Mapping

TRIPOD+AI:

- Define intended use and clinical setting.
- Define predictors, outcome, and prediction timepoint.
- Report participant flow and missing data.
- Report model development, validation, and performance.
- Report limitations and generalizability constraints.

CLAIM:

- Define imaging data source, preprocessing, labels, split strategy, and leakage control.
- Report architecture, training, validation, and evaluation metrics.
- Include data/code availability and failure analysis.

Manuscript-specific safety checklist:

- Participant-level split verified.
- Cluster bootstrap implemented.
- QC gate implemented.
- Regression-output calibration bins generated; probability calibration remains pending.
- ACD-cutoff utility curve generated; calibrated-probability decision curve analysis remains pending.
- No diagnosis/treatment/CDSS claims.
- External validation status explicitly stated.

## 7. Data and Code Availability Draft

Draft text:

> De-identified derived tabular labels, split definitions, prediction outputs, and evaluation scripts will be made available as permitted by institutional governance. Imaging data will be available through a controlled-access pathway under a Data Use Agreement because the dataset contains clinical ophthalmic images. The code required to reproduce the reported evaluation, including split generation, model inference, clinical metrics, and figure/table generation, will be released with documentation.

## 8. Ethics and Governance Draft

Draft text:

> The dataset was de-identified before analysis. Imaging access and downstream reuse are governed by the approved data access pathway and Data Use Agreement. This study evaluates research software for biomarker prediction and referral-triage analysis; it is not intended for autonomous diagnosis, treatment recommendation, or direct clinical deployment without further external and prospective validation.

## 9. Author Contributions Placeholder

- Conceptualization: [names]
- Data curation: [names]
- Methodology: [names]
- Software: [names]
- Formal analysis: [names]
- Clinical interpretation: [names]
- Writing - original draft: [names]
- Writing - review and editing: [names]
- Supervision: [names]

## 10. Immediate Next Tasks

Completed in the first local evidence package:

1. Implemented `code/train_angle_closure_multitask.py`.
2. Generated `paper2_runs/angle_closure_screening/split_manifest.csv` and `split_manifest_summary.md`.
3. Added eye-level Shaffer grade 0/1 closure labeling with exclusion of missing or not-seen grades.
4. Added patient-level 70/15/15 stratified splitting with leakage checks.
5. Added validation-selected thresholds, calibration, decision curve, bootstrap CI, subgroup, and biometric-regression outputs.
6. Updated the Methods framing to angle-closure / narrow-angle referral triage rather than general glaucoma diagnosis.

Remaining:

1. Freeze the image-only multitask anatomical model as the current baseline.
2. Add formal evaluation of predicted biomarker-derived angle risk, starting with low predicted nasal TISA500.
3. Run view-specific image-only multitask baselines for center, van_nasal, van_temporal, and usable multi-view.
4. Review test-set false positives, false negatives, and worst biometric-regression errors.
5. Insert final references and complete STARD-AI / TRIPOD+AI / CLAIM checklists.
6. Add quadrant-level gonioscopy sensitivity analysis only if quadrant labels become available.
7. Obtain external/site/device/prospective validation before any npj Digital Medicine clinical-translation claim.

## 11. Do Not Overclaim

Use:

- referral triage
- angle-closure / narrow-angle risk
- AS-OCT-derived anterior chamber biomarkers
- preliminary clinical validation
- requires external validation before deployment

Avoid:

- diagnosis of glaucoma
- treatment recommendation
- clinical decision support system
- deployable screening tool
- general glaucoma screening
- AS-OCT replacement
