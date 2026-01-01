## Methodology

### 3. Data Acquisition and Preprocessing
#### 3.1 Data Sources
This retrospective dataset comprises multimodal anterior segment data collected at Farabi Eye Hospital. The dataset includes:
- **Slit-Lamp Photography (2D RGB):** Standard anterior segment examinations captured under multiple illumination/view settings, including:
  - Central optical section / pupil-containing views (center, paracenter, and marginal pupil variants but still within the pupil region).
  - Peripheral Van Herick technique views (left and right peripheral/limbal views).
- **Anterior Segment OCT (AS-OCT):** Quantitative biometric parameters exported from Tomey CASIA2 (Tomey Corp, Nagoya, Japan), a 1310‑nm swept‑source AS‑OCT platform. CASIA2-derived anterior segment biometry and lens measurements have demonstrated good repeatability/reproducibility in prior validation studies (Xu et al., 2017; Shoji et al., 2017). The use of 1310‑nm swept‑source OCT is also well-established for long-range, deep-penetration anterior eye imaging (Su et al., 2015). Ground-truth parameters used:
  - **Central/Global parameters (center-view related):** Anterior Chamber Depth (ACD), Central Corneal Thickness (CCT), Lens Vault (LV), Lens Thickness (LT), Angle-to-Angle distance (ATA).
  - **Peripheral/Angle parameters (Van Herick view related):** Angle Opening Distance (AOD250/500/750), Trabecular-Iris Space Area (TISA250/500/750), Trabecular-Iris Angle (TIA500), Angle Recess Area (ARA250/500).

#### 3.2 Data Integration and Patient/Eye Matching
Inconsistent linkage between manually named slit-lamp image files and automated OCT exports reduced usable data. This is a common workflow issue in clinical datasets, since slit-lamp photography and AS-OCT acquisition are often performed by different operators at different times, and naming typos/inconsistencies can occur. A hierarchical multi-stage linkage pipeline was used; deterministic matching prioritized high precision, while probabilistic matching recovered additional valid pairs by tolerating minor string differences:
- **Deterministic Matching:** Records with a unique 9-digit patient identifier linked directly.
- **Probabilistic Matching:** For records without reliable IDs, fuzzy string matching using Levenshtein edit distance (Levenshtein, 1966) associated slit-lamp entries and OCT records; similarity threshold 80% to accept candidates.
- **Human-in-the-Loop Verification:** Candidates in the 60–80% similarity range were manually reviewed.

This yielded a consolidated dataset of 332 patients and 15,912 slit-lamp images paired with valid OCT-derived ground-truth parameters.

### 4. View Categorization and Anatomical Normalization — Active Learning Labeling
Anatomical view labeling of slit-lamp photographs is a common bottleneck due to large image volumes and the need for expert review. To reduce manual burden while maintaining label quality, we used a semi-supervised, human-in-the-loop active learning workflow (Settles, 2009) that iteratively combined model-driven pseudo-labeling (Lee, 2013) with targeted manual verification and correction.

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
- Train a lightweight view classifier (EfficientNet-B0) on the current labeled set (stratified train/val split) (Tan & Le, 2019).
- Score the unlabeled pool and compute prediction confidence and ambiguity via the margin (top1 − top2), a standard uncertainty-sampling strategy in active learning (Scheffer et al., 2001; Settles, 2009).
- Accept pseudo-labels when `pred_conf ≥ 0.9` and `margin ≥ 0.15` to expand the training set with high-confidence predictions (Lee, 2013); export the remaining lowest-confidence / lowest-margin images for manual review.
- Manually review and correct both “low-confidence” and sampled “high-confidence” predictions, update `labels_output.csv`, and repeat.

In early iterations, we scored small subsets of unlabeled images (e.g., 666 per round) to keep review manageable; as label coverage increased, we expanded to scoring the full remaining unlabeled pool. In the final iteration, 5,529 images remained unlabeled; the classifier produced 5,307 high-confidence pseudo-labels and flagged 222 low-confidence examples for manual correction.

Final label distribution (all 15,912 images): `center` 3,769; `van_nasal` 6,144; `van_temporal` 5,383; `no_slit` 259; `other` 357. For reproducibility, the exact workflow and commands are documented in `slit-project/labeling_readme.md`.

### 5. Model Training (Multi-View Regression)
After view labeling, we trained regression models to predict OCT-derived targets from slit-lamp images, using grouped splits at the patient/eye level to avoid leakage. Our main implementation for ACD uses a multi-view ResNet-50 (He et al., 2016) with attention-based multiple-instance learning (MIL) pooling (Ilse et al., 2018) (`slit-project/code/fusion_acd_center_baseline.py`).

#### 5.1 View-Stratified Training Sets
We used `View_Label` to select the appropriate subset of images for each target family:
- **Center-view targets** (e.g., ACD): trained on images with `View_Label == center`.
- **Peripheral targets** (e.g., nasal/temporal angle parameters): trained on images with `View_Label ∈ {van_nasal, van_temporal}`.

#### 5.2 Multi-View Input and Grouping
Multiple slit-lamp images often exist per patient/eye. We grouped images into a “bag” using a patient/eye key extracted from the filename prefix (`patient_eye_*`). Each bag contains up to `MAX_VIEWS` images (randomly subsampled if more are available). Targets were aggregated per bag by taking the mean across rows (OCT targets are constant per eye in practice).

#### 5.3 Model Architecture (ResNet-50 MIL Attention Pooling)
In clinical settings, multiple slit-lamp photographs are often captured per eye across different viewpoints/illumination, while the OCT-derived target is defined at the eye level. To leverage all available images without discarding data, we used multiple-instance learning (MIL) with attention pooling (Ilse et al., 2018):
- Each view is encoded by a shared ImageNet-pretrained ResNet-50 feature extractor (He et al., 2016; Deng et al., 2009).
- An attention module assigns a weight to each view; a weighted sum yields a bag-level representation.
- A regression head predicts the target(s) from the bag representation.
- This allows the model to prioritize informative views and down-weight low-quality or off-target images without manual pre-selection.

The regression head outputs one or more continuous targets (e.g., ACD only).

#### 5.4 Training Protocol and Evaluation
- **Split**: GroupShuffleSplit by patient/eye to prevent patient leakage (all images from the same eye are confined to a single split); test fraction 0.15 and validation fraction 0.15 of the remaining data.
- **Target scaling**: StandardScaler fit on train targets; training loss computed in z-space; metrics reported in both z-space and raw units (inverse transformed).
- **Loss/metrics**: MSE loss on standardized targets; report MSE/MAE/RMSE (raw and z-scored) and Pearson r.
- **Optimization**: AdamW (lr 5e-4, weight decay 5e-3) with cosine annealing (eta_min 1e-6) (Loshchilov & Hutter, 2019); gradient clipping (1.0); up to 40 epochs.
- **Augmentation**: mild geometric/photometric augmentations (resize + center crop, small rotation/affine, color jitter, occasional blur) and ImageNet normalization (Deng et al., 2009); augmented samples were visually inspected to ensure transformations preserved anatomy and did not introduce artifacts.
- **Regularization**: mixup (α=0.2) (Zhang et al., 2018), EMA weight averaging (decay 0.999) (Tarvainen & Valpola, 2017), and early stopping (min_delta 0.005; patience configured per run).

### References
- Levenshtein, V. I. (1966). "Binary codes capable of correcting deletions, insertions, and reversals." Soviet Physics Doklady, 10(8), 707–710.
- Deng, J., Dong, W., Socher, R., Li, L.‑J., Li, K., & Fei-Fei, L. (2009). "ImageNet: A large-scale hierarchical image database." IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
- Su, J. P., Li, Y., Tang, M., Liu, L., Pechauer, A. D., Huang, D., & Liu, G. (2015). "Imaging the anterior eye with dynamic-focus swept-source optical coherence tomography." Journal of Biomedical Optics, 20(12), 126002. (doi:10.1117/1.JBO.20.12.126002)
- Xu, B. Y., Mai, D. D., Penteado, R. C., Saunders, L., & Weinreb, R. N. (2017). "Reproducibility and Agreement of Anterior Segment Parameter Measurements Obtained Using the CASIA2 and Spectralis OCT2 Optical Coherence Tomography Devices." Journal of Glaucoma, 26(11), 974–979. (doi:10.1097/IJG.0000000000000788)
- Shoji, T., Kato, N., Ishikawa, S., Ibuki, H., Yamada, N., Kimura, I., & Shinoda, K. (2017). "In vivo crystalline lens measurements with novel swept-source optical coherent tomography: an investigation on variability of measurement." BMJ Open Ophthalmology, 1(1), e000058. (doi:10.1136/bmjophth-2016-000058)
- Settles, B. (2009). "Active Learning Literature Survey." University of Wisconsin–Madison, Computer Sciences Technical Report 1648.
- Scheffer, T., Decomain, C., & Wrobel, S. (2001). "Active Hidden Markov Models for Information Extraction." International Symposium on Intelligent Data Analysis (IDA).
- Lee, D.-H. (2013). "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks." ICML Workshop on Challenges in Representation Learning.
- Tan, M., & Le, Q. V. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." International Conference on Machine Learning (ICML).
- Ilse, M., Tomczak, J. M., & Welling, M. (2018). "Attention-based Deep Multiple Instance Learning." International Conference on Machine Learning (ICML).
- Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay Regularization." International Conference on Learning Representations (ICLR).
- Zhang, H., Cissé, M., Dauphin, Y. N., & Lopez-Paz, D. (2018). "mixup: Beyond Empirical Risk Minimization." International Conference on Learning Representations (ICLR).
- Tarvainen, A., & Valpola, H. (2017). "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results." Advances in Neural Information Processing Systems (NeurIPS).
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
