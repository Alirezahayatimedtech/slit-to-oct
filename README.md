# slit-to-oct

A collection of experiments predicting OCT-derived anterior segment metrics from slit-lamp images. Includes:
- Data prep scripts to map slit-lamp images to CASIA OCT biometrics and cluster/label views.
- Multi-view regression models (early/late fusion, MIL) for targets like CCT, AOD/TISA/TIA, ACD.
- Utilities for anatomical normalization (nasal/temporal), evaluation, and visualization (attention maps, scatter plots).
- Slit-lamp labeling + active-learning workflow documented in `slit-project/labeling_readme.md`.

This repo currently hosts the submodule `RETFound_MAE` for related MAE work; main code and data are kept locally/offline for now.
