# slit-to-oct

A collection of experiments predicting OCT-derived anterior segment metrics from slit-lamp images. Includes:
- Data prep scripts to map slit-lamp images to CASIA OCT biometrics and cluster/label views.
- Multi-view regression models (early/late fusion, MIL) for targets like CCT, AOD/TISA/TIA, ACD.
- Utilities for anatomical normalization (nasal/temporal), evaluation, and visualization (attention maps, scatter plots).
- Slit-lamp labeling + active-learning workflow documented in `slit-project/labeling_readme.md`.

## Main Project Areas

- `slit-project/`: slit-lamp to AS-OCT modeling scripts and local experiment outputs.
- `Glaucoma-Screening/`: Paper 2 angle-closure screening documentation, validation summaries, and progress tracking.
- `METHODOLOGY.md`: slit-lamp/OCT data integration and modeling methodology.

## Current Paper 2 Focus

The active paper work is slit-lamp image-based angle-closure referral triage.
The primary endpoint is eye-level Shaffer grade `0/1` versus `2/3/4`, with
patient-level splitting to avoid leakage.

Key tracking documents:

- `Glaucoma-Screening/README.md`
- `Glaucoma-Screening/results/experiment_progress_chart.md`
- `Glaucoma-Screening/results/best_model_next_experiments.md`
- `Glaucoma-Screening/EXPERIMENT_SOLUTION_TREE.md`

## Repository Scope

This repository is now intended to track only the slit-lamp/OCT and glaucoma
screening work. RETFound age-prediction, OSD679 reproducibility bundles, and
other non-slit local experiments should stay outside this Git history or remain
ignored local files.
