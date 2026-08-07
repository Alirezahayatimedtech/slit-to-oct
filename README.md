# Paired Slit-Lamp and AS-OCT Research

This repository supports two studies built from the same paired anterior-segment
dataset:

1. a published data descriptor for the dataset and its access model; and
2. an ongoing internal-validation study of gonioscopy-defined angle-closure
   screening.

The two studies answer different questions. The data descriptor documents and
validates the resource. The screening study evaluates whether slit-lamp images
or measured anterior-segment anatomy can distinguish Shaffer grade `0/1` from
`2/3/4`.

## Study 1: Published Data Descriptor

**A multimodal paired slit-lamp photography and anterior segment optical
coherence tomography dataset for anterior chamber biometry and cross-modality
modelling** was published in *Scientific Data* on 1 August 2026.

- Article: https://doi.org/10.1038/s41597-026-07992-9
- Participants: 286
- Eye-level records: 562
- Slit-lamp photographs: 15,582
- AS-OCT images: 9,092

The release links slit-lamp photographs, CASIA 2 AS-OCT images, eye-level
clinical information, gonioscopic Shaffer grades, and quantitative
anterior-segment measurements through non-meaningful `patient_id` and `eye`
keys.

### Data access

| Resource | Access | DOI |
| --- | --- | --- |
| Tabular data, data dictionary, and documentation | Open | https://doi.org/10.5281/zenodo.21006557 |
| Slit-lamp and AS-OCT images | Controlled under a Data Use Agreement | https://doi.org/10.5281/zenodo.18432418 |

Images are not stored in Git. Researchers must use the controlled-access
workflow on Zenodo. Do not redistribute approved imaging files or attempt
participant re-identification.

The release structure, acquisition protocol, linkage, quality control, and ACD
technical-validation baseline are summarized in [METHODOLOGY.md](METHODOLOGY.md).

## Study 2: Angle-Closure Screening

The second study concerns **angle-closure referral screening**, not diagnosis of
all forms of glaucoma. Its reference endpoint is the eye-level gonioscopic
Shaffer grade:

- positive: grade `0` or `1`;
- negative: grade `2`, `3`, or `4`;
- excluded: missing, indeterminate, or `not seen` grades.

The corrected internally validated cohort contains 560 eyes from 286
participants, including 55 positive eyes from 30 participants. Current evidence
shows that measured AS-OCT anatomy contains substantially more discriminatory
information than the available slit-lamp photographs. This is an internal
paired-modality result, not external validation or a deployment claim.

Methods, current results, limitations, and the script map are in
[Glaucoma-Screening/README.md](Glaucoma-Screening/README.md).

## Repository Map

```text
.
|-- README.md                         project and data-access entry point
|-- METHODOLOGY.md                    shared dataset and modelling methods
|-- slit-project/
|   |-- labeling_readme.md            slit-image view annotation guide
|   `-- code/                         release QC and modelling scripts
`-- Glaucoma-Screening/
    `-- README.md                     Paper 2 endpoint, validation, and results
```

Only source code and compact documentation are tracked. Raw data, model weights,
predictions, checkpoints, local paths, and generated figures remain outside Git.

## Code Map

### Dataset and cross-modality work

- `slit-project/code/validate_release.py`: validates the public tabular release
  against the published cohort structure.
- `slit-project/code/label_gui.py`: manual slit-image view annotation utility.
- `slit-project/code/fusion_acd_center_baseline.py`: multi-view ACD regression
  baseline with participant-level splitting.
- `slit-project/code/train_multitarget_fusion.py`: multi-target slit-to-AS-OCT
  regression experiments.

### Angle-closure work

- `slit-project/code/train_angle_grade_regression_cv.py`: patient-disjoint
  Shaffer-grade regression.
- `slit-project/code/train_resnet50_anatomy_stack_cv.py`: anatomy-regression
  stack and angle-closure evaluation.
- `slit-project/code/train_convnext_mil_anatomy_stack_cv.py`: multi-view
  ConvNeXt anatomy stack.
- `slit-project/code/evaluate_anatomy_stack_feature_experiments.py`: shallow
  feature-set ablations.
- `slit-project/code/evaluate_oof_threshold_calibration.py`: out-of-fold
  operating-point analysis.

## Validate the Open Release

Download `data.csv` from the open Zenodo record, then run:

```bash
python slit-project/code/validate_release.py --data /path/to/data.csv
```

The validator checks row counts, participant counts, patient-eye uniqueness,
laterality, Shaffer-grade distribution, and required columns against the
published release.

Training scripts require an approved local image directory and a prepared image
manifest. Pass local paths through each script's command-line arguments. Never
commit controlled images, derived patient-level manifests, or model outputs.

## Reproducibility Rules

- Split by participant, not image or eye. Both eyes and all repeated images from
  one participant must remain in one partition.
- Fit imputation, normalization, feature selection, and operating thresholds on
  training data only.
- Report the exact endpoint, release version, exclusions, and participant/eye
  counts.
- Resample participants, not individual eyes, when estimating uncertainty.
- Keep slit-only and AS-OCT-assisted models clearly separated.

## Citation

```bibtex
@article{tabatabaei2026slitoct,
  title   = {A multimodal paired slit-lamp photography and anterior segment
             optical coherence tomography dataset for anterior chamber
             biometry and cross-modality modelling},
  author  = {Tabatabaei, Seyed Mehdi and Hayati, Alireza and Safizadeh, Mona
             and Vahedian, Zakieh and Ahmadi, Amin and Arian, Roya and Keane,
             Pearse A. and Tayebi, Fereshteh},
  journal = {Scientific Data},
  year    = {2026},
  doi     = {10.1038/s41597-026-07992-9}
}
```

Dataset users should also cite the specific Zenodo record version they used.
