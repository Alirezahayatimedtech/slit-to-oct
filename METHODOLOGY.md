## Methodology

### 3. Data Acquisition and Preprocessing
#### 3.1 Data Sources
This retrospective dataset comprises multimodal anterior segment data collected at Farabi Eye Hospital. The dataset includes:
- **Slit-Lamp Photography (2D RGB):** Standard anterior segment examinations captured under multiple illumination/view settings, including:
  - Central optical section / pupil-containing views (center, paracenter, and marginal pupil variants but still within the pupil region).
  - Peripheral Van Herick technique views (left and right peripheral/limbal views).
- **Anterior Segment OCT (AS-OCT):** Quantitative biometric parameters exported from CASIA2 (Tomey Corp, Nagoya, Japan) as device-generated tabular outputs. Ground-truth parameters used:
  - **Central/Global parameters (center-view related):** Anterior Chamber Depth (ACD), Central Corneal Thickness (CCT), Lens Vault (LV), Lens Thickness (LT), Angle-to-Angle distance (ATA).
  - **Peripheral/Angle parameters (Van Herick view related):** Angle Opening Distance (AOD250/500/750), Trabecular-Iris Space Area (TISA250/500/750), Trabecular-Iris Angle (TIA500), Angle Recess Area (ARA250/500).

#### 3.2 Data Integration and Patient/Eye Matching
Inconsistent linkage between manually named slit-lamp image files and automated OCT exports reduced usable data. This is a common workflow issue in clinical datasets, since slit-lamp photography and AS-OCT acquisition are often performed by different operators at different times, and naming typos/inconsistencies can occur. A hierarchical multi-stage linkage pipeline was used:
- **Deterministic Matching:** Records with a unique 9-digit patient identifier linked directly.
- **Probabilistic Matching:** For records without reliable IDs, fuzzy string matching (Levenshtein distance) associated slit-lamp entries and OCT records; similarity threshold 80% to accept candidates.
- **Human-in-the-Loop Verification:** Candidates in the 60–80% similarity range were manually reviewed.

This yielded a consolidated dataset of 332 patients and 15,912 slit-lamp images paired with valid OCT-derived ground-truth parameters.

### 4. View Categorization and Anatomical Normalization — Active Learning Labeling
We used a 2-stage labeling scheme to make view labels consistent across eyes:

1) **Image-space view (`View_Image`)** — manually assigned per image using a custom keyboard-driven GUI (`slit-project/code/label_gui.py`):
   - `center`, `left`, `right`, `no_slit`, `other`

2) **Anatomical view (`View_Label`)** — derived from `View_Image` using eye laterality (`eye_clean`):
   - `center` → `center`
   - `no_slit` → `no_slit`
   - `other` → `other`
   - For **OD**: `left` → `van_temporal`, `right` → `van_nasal`
   - For **OS**: `left` → `van_nasal`, `right` → `van_temporal`

All labels were stored in `slit-project/code/labels_output.csv` (columns: `Image_Path`, `View_Image`, `View_Label`, `eye_clean`). During labeling, skipped images were left blank to avoid accidental class assignment; the final dataset was fully labeled (15,912 images).

To scale annotation from an initial seed set to the full dataset, we used an iterative **active learning + semi-supervised** loop (`slit-project/code/active_learning_view_pipeline.py`):
- Train a lightweight view classifier (EfficientNet-B0) on the current labeled set (stratified train/val split).
- Score the unlabeled pool and compute prediction confidence and ambiguity via the margin (top1 − top2).
- Accept pseudo-labels when `pred_conf ≥ 0.9` and `margin ≥ 0.15`; export the remaining lowest-confidence / lowest-margin images for manual review.
- Manually review and correct both “low-confidence” and sampled “high-confidence” predictions, update `labels_output.csv`, and repeat.

In early iterations, we scored small subsets of unlabeled images (e.g., 666 per round) to keep review manageable; as label coverage increased, we expanded to scoring the full remaining unlabeled pool. In the final iteration, 5,529 images remained unlabeled; the classifier produced 5,307 high-confidence pseudo-labels and flagged 222 low-confidence examples for manual correction.

Final label distribution (all 15,912 images): `center` 3,769; `van_nasal` 6,144; `van_temporal` 5,383; `no_slit` 259; `other` 357. For reproducibility, the exact workflow and commands are documented in `slit-project/labeling_readme.md`.

### 5. Model Training (Multi-View Regression)
After view labeling, we trained regression models to predict OCT-derived targets from slit-lamp images, using grouped splits at the patient/eye level to avoid leakage. Our main implementation for ACD uses a multi-view ResNet-50 with fusion or MIL attention pooling (`slit-project/code/fusion_acd_center_baseline.py`).

#### 5.1 View-Stratified Training Sets
We used `View_Label` to select the appropriate subset of images for each target family:
- **Center-view targets** (e.g., ACD): trained on images with `View_Label == center`.
- **Peripheral targets** (e.g., nasal/temporal angle parameters): trained on images with `View_Label ∈ {van_nasal, van_temporal}`.

#### 5.2 Multi-View Input and Grouping
Multiple slit-lamp images often exist per patient/eye. We grouped images into a “bag” using a patient/eye key extracted from the filename prefix (`patient_eye_*`). Each bag contains up to `MAX_VIEWS` images (randomly subsampled if more are available). Targets were aggregated per bag by taking the mean across rows (OCT targets are constant per eye in practice).

#### 5.3 Model Architecture (ResNet-50 Fusion / MIL)
We used an ImageNet-pretrained ResNet-50 backbone with two alternative ways to combine multi-view information:
- **Fusion**: early fusion averages the per-view image tensors before the backbone; late fusion averages per-view predictions.
- **MIL attention pooling**: extracts per-view feature vectors, learns attention weights over views, and computes a weighted bag representation before the regression head.

The regression head outputs one or more continuous targets (e.g., ACD only).

#### 5.4 Training Protocol and Evaluation
- **Split**: GroupShuffleSplit by patient/eye (test fraction 0.15; validation fraction 0.15 of remaining).
- **Target scaling**: StandardScaler fit on train targets; training loss computed in z-space; metrics reported in both z-space and raw units (inverse transformed).
- **Loss/metrics**: MSE loss on standardized targets; report MSE/MAE/RMSE (raw and z-scored) and Pearson r.
- **Optimization**: AdamW (lr 5e-4, weight decay 5e-3) with cosine annealing (eta_min 1e-6); gradient clipping (1.0); up to 40 epochs.
- **Augmentation**: mild geometric/photometric augmentations (resize + center crop, small rotation/affine, color jitter, occasional blur) and ImageNet normalization.
- **Regularization**: mixup (α=0.2), EMA weight averaging (decay 0.999), and early stopping (min_delta 0.005; patience configured per run).

### References
- Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions, and reversals. Soviet Physics Doklady, 10(8), 707–710.
- Deng, J., et al. (2009). ImageNet: A large-scale hierarchical image database. CVPR, 248–255.
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Berkeley Symposium, 281–297.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
