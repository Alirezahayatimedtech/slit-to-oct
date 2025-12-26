## Methodology

### 3. Data Acquisition and Preprocessing
#### 3.1 Data Sources
This retrospective dataset comprises multimodal anterior segment data collected from [Insert Hospital/Clinic Name]. The dataset includes:
- **Slit-Lamp Photography (2D RGB):** Standard anterior segment examinations captured under multiple illumination/view settings, including:
  - Central optical section / pupil-containing views (center, paracenter, and marginal pupil variants but still within the pupil region).
  - Peripheral Van Herick technique views (left and right peripheral/limbal views).
- **Anterior Segment OCT (AS-OCT):** Quantitative biometric parameters exported from CASIA2 (Tomey Corp, Nagoya, Japan) as device-generated tabular outputs. Ground-truth parameters used:
  - **Central/Global parameters (center-view related):** Anterior Chamber Depth (ACD), Central Corneal Thickness (CCT), Lens Vault (LV), Lens Thickness (LT), Angle-to-Angle distance (ATA).
  - **Peripheral/Angle parameters (Van Herick view related):** Angle Opening Distance (AOD250/500/750), Trabecular-Iris Space Area (TISA250/500/750), Trabecular-Iris Angle (TIA500), Angle Recess Area (ARA250/500).

#### 3.2 Data Integration and Patient/Eye Matching
Inconsistent linkage between manually named slit-lamp image files and automated OCT exports reduced usable data. A hierarchical multi-stage linkage pipeline was used:
- **Deterministic Matching:** Records with a unique 9-digit patient identifier linked directly.
- **Probabilistic Matching:** For records without reliable IDs, fuzzy string matching (Levenshtein distance) associated slit-lamp entries and OCT records; similarity threshold 80% to accept candidates.
- **Human-in-the-Loop Verification:** Candidates in the 60–80% similarity range were manually reviewed.

This yielded a consolidated dataset of 332 patients and 15,912 slit-lamp images paired with valid OCT-derived ground-truth parameters.

### 4. Baseline Benchmark Model (First-Step Results)
Before view labeling and ROI standardization, a minimal baseline quantified predictability of OCT parameters from raw slit-lamp images:
- **Input:** Original slit-lamp images with standard resizing only (no beam/angle localization; no view-specific ROIs), processed one image at a time.
- **Labeling:** All images from the same eye share the same OCT-derived ground truth (label duplication at the image level).
- **Backbone/Loss:** ResNet-50 (ImageNet-pretrained) with Huber loss for robustness to outliers.
- **Training:** Loss computed per image (no train-time view aggregation); effective batch increased via gradient accumulation when memory-limited.
- **Evaluation:** Primary reporting is per-image metrics (MAE/MSE/RMSE, z-scored and raw, Pearson r) to match the training regime. A secondary, practical view can be obtained by averaging predictions per eye at test time to see the gain from naive aggregation.

This image-level baseline on view-mixed inputs provides a conservative reference prior to view-aware modeling.

### 5. View Categorization (Center vs Peripheral; Left vs Right) — Semi-Supervised Labeling
Views were categorized into three classes: Center; Van Herick – Left; Van Herick – Right. Workflow:
- **Seed Manual Annotation:** ~1,400 images labeled.
- **Supervised View Classifier:** Trained with grouped splits by eye_clean to avoid leakage.
- **Pseudo-Labeling:** Applied to unlabeled pool; high-confidence predictions accepted.
- **Iterative Spot-Checking:** Ambiguous cases reviewed; model refreshed.
- **Final Tables:** Separate curated tables for center-related targets (ACD, CCT, LV, LT, ATA) and peripheral targets (AOD*, TISA*, TIA500, ARA*).

### 6. ROI Cropping from Original High-Resolution Slit-Lamp Images (Center Class)
- **Problem:** Center-class images vary (center, paracenter, marginal pupil); beam not always centered.
- **Beam Localization:** CLAHE, top-hat filtering, vertical-line enhancement, column-wise projection to find beam_x.
- **ROI:** Beam-centered crop:
  - ROI width fraction: 0.45 of original width.
  - Vertical crop: y_top_frac = 0.12, y_bot_frac = 0.92.
- **Resizing:** ROIs resized to 448×448 (or 384×384).
- **QC:** Debug overlays (beam line + ROI box) for 50 samples before full run.
- **Note:** off_center_score computed/stored but not used for pruning.

### 7. Predictive Modeling for Central Biometry (ACD/CCT/LV/LT/ATA)
#### 7.1 Backbone Selection (10-Epoch Screening)
Multiple backbones (ResNet, EfficientNet, Swin, etc.) trained for 10 epochs with identical grouped splits and augmentation; ConvNeXt-Tiny showed best generalization and stability and was selected.

#### 7.2 Regression Model
- **Input:** Beam-centered ROI crops, resized to 448×448 (or 384×384 if needed).
- **Backbone:** ConvNeXt-Tiny (ImageNet-pretrained).
- **Head:** Regression layer per target.

#### 7.3 Training Protocol and Split
- Grouped splits by eye_clean (patient/eye) to prevent leakage.
- Loss: Huber (tuned delta in later runs).
- Mild geometric/photometric augmentations; gradient clipping.
- Metrics: MSE and MAE (z-scored and raw units).
- Hyperparameter search: Optuna sweeps (lr, weight decay, batch size 4/8) on grouped validation MSE to choose a reasonable baseline configuration before longer training runs.

### References
- Levenshtein, V. I. (1966). Binary codes capable of correcting deletions, insertions, and reversals. Soviet Physics Doklady, 10(8), 707–710.
- Deng, J., et al. (2009). ImageNet: A large-scale hierarchical image database. CVPR, 248–255.
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. Berkeley Symposium, 281–297.
- Liu, Z., et al. (2022). A ConvNet for the 2020s (ConvNeXt). CVPR.
