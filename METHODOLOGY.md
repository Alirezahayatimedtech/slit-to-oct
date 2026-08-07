# Methodology and Reproducibility

This document separates the published dataset methods from the ongoing
angle-closure analysis. The peer-reviewed article is the authoritative source
for the data descriptor. The Paper 2 section records the current analysis
contract and claim limits.

## 1. Published Dataset

### 1.1 Study design

The dataset was collected in a cross-sectional, single-centre study at a general
ophthalmology outpatient clinic between December 2024 and November 2025.
Participants were adults able to provide consent and complete imaging. Corneal
pathology that prevented adequate imaging, prior intraocular or ocular-surface
surgery, confounding anterior-segment pathology, and inability to obtain usable
images were exclusion criteria.

The study received institutional ethics approval
(`IR.TUMS.FARABIH.REC.1403.058`). Written informed consent covered research use
of de-identified data. The controlled-access model for images addresses the
residual re-identification risk of ocular imaging.

### 1.2 Clinical examination

The same-session examination included visual acuity, slit-lamp biomicroscopy,
Goldmann applanation tonometry, fundus examination, and gonioscopy with a Zeiss
four-mirror lens. Two trained specialists performed gonioscopy in dim
illumination using the modified Shaffer classification. The release contains
the final clinical eye-level grade. It was not designed as a masked
inter-observer agreement study.

### 1.3 Slit-lamp photography

Photographs were acquired with a Canon EOS 1300D attached to a Topcon SL-D8
slit-lamp microscope. The protocol included central focal illumination and
nasal and temporal Van Herick views. Captures were repeated when motion blur,
defocus, eyelid artefact, or poor alignment was present.

Released slit-lamp files are 8-bit RGB JPEG images at 2592 x 1728 pixels. The
controlled release contains 15,582 photographs.

### 1.4 AS-OCT acquisition

AS-OCT was performed in a darkened room with a swept-source CASIA 2 device
(Tomey, Japan; 30,000 A-scans/s; 1,310 nm wavelength; 16 mm scan width; 10 um
axial resolution). The quantitative anterior-segment protocol acquired 16
evenly spaced radial B-scans over 360 degrees per eye. Scans were reviewed for
signal quality, blinking, major decentration, and segmentation failure.

The controlled release contains 9,092 exported AS-OCT JPEG images. The open
table includes central and angle measurements such as ACD, CCT, lens vault,
lens thickness, ACW, ATA, AOD, ARA, TISA, and TIA.

### 1.5 Release structure

The final release contains:

| Component | Count |
| --- | ---: |
| Participants | 286 |
| Eye-level records | 562 |
| Left eyes | 282 |
| Right eyes | 280 |
| Participants contributing both eyes | 276 |
| Slit-lamp photographs | 15,582 |
| AS-OCT images | 9,092 |

The public table has one row per `patient_id` and `Eye`. Image names encode the
same non-meaningful participant and eye identifiers. Direct identifiers and
visit timestamps are not released.

- Open table and dictionary: https://doi.org/10.5281/zenodo.21006557
- Controlled images: https://doi.org/10.5281/zenodo.18432418

The data dictionary supplied with the open release is the authority for column
names, units, encodings, and missing values.

### 1.6 Release validation

Quality control covers:

- required-column and data-type checks;
- uniqueness of each patient-eye record;
- allowed values for eye laterality and Shaffer grade;
- broad plausibility checks for age, IOP, cup-to-disc ratio, CCT, and ACD;
- missingness summaries for key fields;
- linkage consistency across tabular, slit-lamp, and AS-OCT identifiers.

Run the public structural checks with:

```bash
python slit-project/code/validate_release.py --data /path/to/data.csv
```

The validator does not alter source data.

### 1.7 ACD technical-validation baseline

The published baseline predicts AS-OCT-derived anterior chamber depth from
multiple slit-lamp photographs. It is a dataset sanity check, not a clinical
model.

- Input unit: multiple photographs grouped by eye.
- Leakage control: all eyes and images from one participant stay in one split.
- Encoder: ImageNet-pretrained convolutional backbone.
- Aggregation: attention-based multiple-instance pooling.
- Target preprocessing: train-set standardization.
- Optimization: mean-squared error with AdamW.
- Evaluation: MAE, RMSE, and Pearson correlation in millimetres.

The article reports five-fold validation MAE `0.21 +/- 0.01`, RMSE
`0.27 +/- 0.02`, and Pearson `r = 0.78 +/- 0.02`. The held-out test included 85
eyes and produced MAE `0.25`, RMSE `0.36`, and Pearson `r = 0.64`. These results
show cross-modal signal; they do not establish clinical utility.

## 2. Angle-Closure Screening Study

### 2.1 Clinical question

Can slit-lamp photographs identify eyes requiring formal assessment for
gonioscopy-defined angle closure, and how does their information compare with
measured AS-OCT anatomy?

This is not an analysis of glaucomatous optic neuropathy. The dataset does not
provide adjudicated optic-nerve, RNFL/GCC, or visual-field endpoints required
for a general glaucoma diagnostic claim.

### 2.2 Reference endpoint

The primary label is derived from the released eye-level Shaffer grade:

```text
closure_label = 1  when angle is 0 or 1
closure_label = 0  when angle is 2, 3, or 4
exclude missing, indeterminate, and not-seen grades
```

Quadrant-resolved closure cannot be inferred from this eye-level field.

Secondary analyses may evaluate `0/1/2` versus `3/4`, `0/1` versus `3/4` after
excluding grade 2, and errors for each individual grade. These analyses do not
replace the primary endpoint.

### 2.3 Analysis cohort and splitting

The corrected paired analysis contains 560 eyes from all 286 participants, with
55 positive eyes from 30 positive participants. Every split is participant
disjoint. All photographs and both eyes from a participant remain together.

For model comparison:

- outer evaluation uses patient-disjoint folds;
- preprocessing and model selection are fitted inside the training partition;
- thresholds are selected from training or inner out-of-fold predictions;
- uncertainty is estimated with participant-cluster bootstrap resampling;
- the same corrected eye set is used for paired model comparisons.

### 2.4 Model families

1. **Slit-only:** single-view or multi-view image models using central and Van
   Herick photographs.
2. **Measured anatomy:** regularized models using ACD, lens vault, AOD500,
   TISA500, CCT, and related AS-OCT measurements.
3. **Clinical fusion:** age, sex, IOP, lens status, and CCT added to an image or
   anatomy representation.
4. **Privileged supervision:** AS-OCT targets or teacher scores available during
   training but not slit-only inference.

Inputs available at evaluation must be stated for every reported model.

### 2.5 Current evidence

On the same 560 eyes, the frozen slit-lamp own-eye score achieved AUROC `0.659`
(participant-bootstrap 95% CI `0.554-0.758`). The nested measured-AS-OCT
own-eye model achieved AUROC `0.824` (`0.758-0.886`). The paired difference was
`0.166` (`0.049-0.287`).

A parsimonious model using ACD, lens vault, mean AOD500, mean TISA500, and age
reached AUROC `0.841` (`0.785-0.890`), sensitivity `0.800`, and specificity
`0.756` at nested operating thresholds. Repeated validation did not establish
stable simultaneous 80% sensitivity and 80% specificity.

The defensible result is a modality-information gap and a useful internal
AS-OCT benchmark. It is not evidence that slit-lamp screening can replace
gonioscopy.

## 3. Shared Reproducibility Requirements

- Use the published Zenodo version as the source of truth.
- Never join records from different identifier namespaces without a validated
  crosswalk.
- Preserve anatomical laterality when mapping nasal and temporal measurements.
- Keep data preprocessing inside the training partition.
- Save patient-level split manifests and verify zero participant overlap.
- Report counts at participant, eye, and image levels.
- Separate exploratory model selection from final evaluation.
- Do not commit controlled images, patient-level derived files, or model
  weights.

## 4. References

- Published data descriptor: https://doi.org/10.1038/s41597-026-07992-9
- Open tabular release: https://doi.org/10.5281/zenodo.21006557
- Controlled imaging release: https://doi.org/10.5281/zenodo.18432418
