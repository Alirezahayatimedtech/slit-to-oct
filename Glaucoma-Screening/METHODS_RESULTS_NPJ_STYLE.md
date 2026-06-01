# Manuscript-Ready Methods and Results: Iterative Slit-Lamp Angle-Closure Screening

Draft purpose: Methods/Results text for a publication framing that emphasizes
transparent model development, clinically motivated problem solving, and the
limitations that remain before external clinical deployment.

Target style: npj Digital Medicine compatible, with conservative language. This
document describes **internal model development and patient-level
cross-validation**, not external validation.

## Proposed Title

Anatomy-supervised slit-lamp image learning for gonioscopic angle-closure
screening: iterative model development and internal validation

## Methods

### Study Design and Clinical Framing

We conducted a retrospective paired-imaging model development study to evaluate
whether slit-lamp photographs can support screening for gonioscopic
angle-closure risk. The intended clinical use case was referral triage for
eyes that may require confirmatory gonioscopy or anterior-segment optical
coherence tomography (AS-OCT), rather than standalone glaucoma diagnosis. This
distinction was prespecified because the available reference labels were
eye-level Shaffer angle grades and AS-OCT anterior segment biomarkers, not optic
nerve, retinal nerve fiber layer, or visual-field endpoints.

The primary binary endpoint was gonioscopic angle closure, defined as an
eye-level Shaffer grade of `0` or `1`. Eyes with grades `2`, `3`, or `4` were
treated as non-closed for the primary strict-label analysis. Eyes marked as
missing, indeterminate, or not seen were excluded. A grade-2-excluded analysis
was conducted as a sensitivity analysis to evaluate the impact of clinically
borderline labels.

### Dataset and Eligibility

The final complete-case anatomy-supervised analysis included slit-lamp images
from eyes with valid Shaffer grades, usable image paths, usable view labels, and
complete selected AS-OCT biomarker labels. All repeated images from the same
eye were retained for image-level model training, but performance was evaluated
at the eye level.

Table 1 summarizes the primary complete-case cohort used for the current
ConvNeXt anatomy-stack experiments.

**Table 1. Complete-case cohort for anatomy-supervised angle-closure modeling**

| Quantity | Count |
| --- | ---: |
| Participants | 258 |
| Eyes | 476 |
| Slit-lamp image rows after filtering | 13,080 |
| Positive eyes, Shaffer 0/1 | 36 |
| Negative eyes, Shaffer 2/3/4 | 440 |

**Table 2. Eye-level Shaffer grade distribution**

| Shaffer grade | Eyes | Primary strict-label class |
| ---: | ---: | --- |
| 0 | 12 | Closed |
| 1 | 24 | Closed |
| 2 | 76 | Non-closed, borderline sensitivity group |
| 3 | 317 | Non-closed |
| 4 | 47 | Non-closed |

### Image Views and Quality Filtering

Image records included view labels assigned during the slit-lamp data curation
workflow. For the current modeling experiments, images were restricted to
usable labeled views:

- `center`
- `van_nasal`
- `van_temporal`

Images labeled `other` or `no_slit` were excluded from the primary
anatomy-supervised model because they were not expected to contain reliable
anterior chamber or Van Herick information. This filtering step was intended to
reduce off-target supervision while retaining multiple clinically relevant
views per eye.

### Model Development Strategy

The model development process followed an iterative error-driven strategy. We
began with direct image-level binary classification, but shifted toward
anatomy-supervised learning after early experiments showed that direct
classification was weak and unstable. The final leading model used slit-lamp
images to predict AS-OCT-derived anterior chamber biomarkers, then used those
predicted biomarkers to estimate angle-closure risk.

The best-performing model family used a ConvNeXt image encoder pretrained on
ImageNet. Each slit-lamp image was processed independently to predict continuous
AS-OCT anatomical targets. Image-level predictions were then averaged within
each eye to create an eye-level predicted anatomy representation. A shallow
logistic regression model with class balancing was trained on the predicted
eye-level anatomy to estimate the probability of Shaffer-defined angle closure.

The final target-focused model predicted six angle-relevant AS-OCT biomarkers:

- anterior chamber depth, endothelium to lens surface (`ACD[Endo.]`)
- lens vault
- temporal AOD500
- nasal AOD500
- temporal TISA500
- nasal TISA500

This six-target configuration was selected because earlier experiments showed
that angle-focused targets improved the balance of sensitivity and specificity
relative to broader target sets.

### Neural Network Training

The leading model used ConvNeXt-Tiny with the backbone unfrozen. Training used
standard ImageNet normalization and online augmentation consisting of resizing,
small random rotation, and mild color jitter. Continuous anatomical targets were
standardized using training-fold statistics, and the model was optimized using
a Smooth L1 regression loss. The final regularized ConvNeXt-Tiny configuration
used AdamW with learning rate `5e-5`, weight decay `5e-4`, mixed-precision
training, a maximum of 6 epochs, and early stopping with patience of 2 epochs.
To limit imbalance from eyes with many repeated photographs, training sampled
up to 12 images per eye per fold; validation used all available images.

ConvNeXt-Small was evaluated as a bounded capacity test using the same
angle-focused target set with learning rate `3e-5` and weight decay `5e-4`.

### Validation Design

All validation used participant-level separation. Both eyes and all slit-lamp
images from the same participant were kept within the same fold. The main
robustness analysis used patient-level 5-fold cross-validation. Within each
fold, the image model was trained on the training participants, predicted
anatomical values for training and validation images, and a logistic regression
risk model was fitted using training-fold eye-level predicted anatomy.

Two threshold-selection approaches were evaluated:

1. **Fold-internal validation-balanced threshold**: selected within the held-out
   validation fold to maximize the minimum of sensitivity and specificity. This
   reflects an optimistic development diagnostic and was used to measure the
   upper bound of separability within each fold.
2. **Other-fold out-of-fold threshold**: for each held-out fold, thresholds were
   selected using predictions from the other four folds, then applied to the
   held-out fold. This tested threshold transfer and calibration stability.

The primary performance metrics were AUROC, AUPRC, sensitivity, specificity,
positive predictive value, negative predictive value, accuracy, and the minimum
of sensitivity and specificity. Because closed-angle cases were rare, AUPRC and
threshold stability were interpreted alongside AUROC.

### Iterative Experiments

Table 3 summarizes the model development sequence and the rationale for each
step. These experiments were not treated as independent confirmatory tests;
they were used to identify the most defensible modeling strategy.

**Table 3. Iterative model-development experiments**

| Step | Experiment | Main question | Key decision |
| ---: | --- | --- | --- |
| 1 | Direct whole-image binary classifier | Can slit-lamp images directly classify Shaffer 0/1 vs 2/3/4? | Direct classification was weak; abandon as primary route. |
| 2 | Multitask image model and shallow anatomical risk | Does anatomy-related supervision improve risk signal? | Predicted anatomy carried more signal than direct probability. |
| 3 | Weighted Shaffer-grade regression | Does continuous grade prediction use more label information? | Continuous grade prediction improved ranking but needed calibration. |
| 4 | ResNet-50 10-biomarker anatomy stack | Are AS-OCT biomarkers a useful intermediate representation? | Anatomy-stack regression became the main model family. |
| 5 | Frozen ConvNeXt-Tiny all-10 anatomy stack | Does the signal survive patient-level 5-fold validation? | Signal persisted but was weaker under stricter validation. |
| 6 | Frozen ConvNeXt-Tiny angle-6 targets | Does focusing on angle-related targets reduce noise? | Angle-focused targets improved balanced performance. |
| 7 | Unfrozen ConvNeXt-Tiny all-10 | Does fine-tuning improve the image representation? | Fine-tuning improved AUROC but overfit without stronger regularization. |
| 8 | Regularized unfrozen ConvNeXt-Tiny angle-6 | Can focused targets plus regularized fine-tuning improve balance? | Became the leading strict-label model. |
| 9 | Grade-2-excluded sensitivity analysis | Is borderline grade 2 a major source of label noise? | Mean sensitivity/specificity improved, but fold stability did not. |
| 10 | ConvNeXt-Small capacity test | Does a larger backbone improve performance? | Larger backbone did not improve AUROC or specificity. |
| 11 | Out-of-fold threshold calibration | Does threshold selection transfer across folds? | Threshold transfer remained unstable. |

## Results

### Direct Classification Was Insufficient

The initial whole-image binary classifier performed poorly, with AUROC `0.317`,
sensitivity `0.250`, and specificity `0.630` in the fixed-split diagnostic
analysis. This result indicated that direct binary supervision was not a robust
starting point for the small positive class and clinically noisy angle labels.
Subsequent experiments therefore shifted from direct classification toward
anatomy-supervised learning.

### Anatomy-Supervised Learning Improved the Signal

Adding anatomy-related supervision produced the first meaningful performance
increase. A multitask image model followed by shallow anatomical risk modeling
achieved AUROC `0.702`, sensitivity `0.875`, and specificity `0.630` in a
fixed-split validation-selected analysis. A weighted Shaffer-grade regression
model reached AUROC `0.755`, sensitivity `0.700`, and specificity `0.755` on an
80/20 validation split. A ResNet-50 anatomy stack predicting 10 AS-OCT
biomarkers achieved AUROC `0.764`, sensitivity `0.714`, and specificity `0.721`
in the same 80/20 development setting.

When evaluated with patient-level 5-fold cross-validation, performance was
lower, indicating that the fixed-split development results were optimistic. The
first rigorous ConvNeXt-Tiny all-10 anatomy stack achieved mean AUROC `0.655`,
sensitivity `0.657`, and specificity `0.642`. This supported the presence of a
reproducible signal but showed that the model was not yet clinically balanced.

**Table 4. Development trajectory across selected milestone experiments**

| Experiment | Validation design | AUROC | Sensitivity | Specificity | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| Direct whole-image classifier | Fixed split diagnostic | 0.317 | 0.250 | 0.630 | Direct class head failed. |
| Multitask anatomical risk | Fixed split validation-selected | 0.702 | 0.875 | 0.630 | Anatomy-related outputs improved sensitivity. |
| Weighted Shaffer-grade regression | 80/20 validation | 0.755 | 0.700 | 0.755 | Continuous grade prediction improved ranking. |
| ResNet-50 10-biomarker anatomy stack | 80/20 validation | 0.764 | 0.714 | 0.721 | Best quick fixed-split anatomy signal. |
| Frozen ConvNeXt-Tiny all-10 anatomy | 5-fold CV | 0.655 | 0.657 | 0.642 | Signal persisted under stricter validation. |
| Frozen ConvNeXt-Tiny angle-6 anatomy | 5-fold CV | 0.662 | 0.663 | 0.654 | Target focusing modestly improved balance. |
| Unfrozen ConvNeXt-Tiny all-10 anatomy | 5-fold CV | 0.684 | 0.679 | 0.668 | Fine-tuning improved AUROC but overfit. |
| Regularized unfrozen ConvNeXt-Tiny angle-6 | 5-fold CV | 0.737 | 0.707 | 0.722 | Best current strict-label model. |

### Target-Focused Regularized ConvNeXt-Tiny Was the Best Strict-Label Model

The best strict-label model was the regularized unfrozen ConvNeXt-Tiny trained
on six angle-focused AS-OCT biomarkers. Using fold-internal
validation-balanced thresholds, this model achieved mean AUROC `0.737`, AUPRC
`0.235`, sensitivity `0.707`, specificity `0.722`, and balanced-min `0.674`.
Two of five folds reached simultaneous sensitivity and specificity of at least
`0.70`.

**Table 5. Patient-level 5-fold performance of the leading strict-label model**

| Model | Threshold rule | AUROC | AUPRC | Sensitivity | Specificity | Balanced min | 70/70 folds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny angle-6 anatomy stack | Fold-internal validation-balanced | 0.737 | 0.235 | 0.707 | 0.722 | 0.674 | 2/5 |
| ConvNeXt-Tiny angle-6 anatomy stack | Threshold from other OOF folds | 0.737 | 0.235 | 0.667 | 0.697 | 0.614 | Not stable |

The gap between fold-internal thresholding and other-fold thresholding showed
that threshold transfer was unstable. Thus, although the model learned a useful
ranking signal, calibration and operating-point selection remain unresolved.

### Grade-2 Exclusion Improved Mean Sensitivity and Specificity but Did Not Solve Stability

Because grade `2` represents a clinically borderline angle category, we repeated
the leading ConvNeXt-Tiny angle-6 experiment after excluding all grade-2 eyes.
The binary task became grade `0/1` versus grade `3/4`. This clean-label
sensitivity analysis included 400 eyes: 36 closed and 364 open.

Using fold-internal validation-balanced thresholds, grade-2 exclusion achieved
AUROC `0.726`, sensitivity `0.722`, specificity `0.750`, and balanced-min
`0.655`. Although mean sensitivity and specificity improved, AUROC decreased
relative to the strict-label exp9 model, and only two of five folds reached
`70/70`.

**Table 6. Strict-label model versus grade-2-excluded sensitivity analysis**

| Analysis | Eyes | AUROC | AUPRC | Sensitivity | Specificity | Balanced min | 70/70 folds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Strict: grade 0/1 vs 2/3/4 | 476 | 0.737 | 0.235 | 0.707 | 0.722 | 0.674 | 2/5 |
| Grade-2 excluded: 0/1 vs 3/4 | 400 | 0.726 | 0.227 | 0.722 | 0.750 | 0.655 | 2/5 |

This result suggests that grade `2` contributes to label ambiguity but is not
the only performance bottleneck. The remaining errors after grade-2 exclusion
included false negatives among grade `0/1` eyes and false positives among grade
`3` eyes, indicating the need for clinical label review and stricter image/view
quality analysis.

### Larger Backbone Capacity Did Not Improve Performance

To test whether model capacity limited performance, we repeated the leading
angle-6 anatomy approach using ConvNeXt-Small. The larger backbone did not
improve the result. ConvNeXt-Small achieved AUROC `0.708`, sensitivity `0.758`,
specificity `0.655`, and balanced-min `0.655` with fold-internal
validation-balanced thresholds. AUROC and specificity were lower than the
ConvNeXt-Tiny model.

**Table 7. Backbone capacity test**

| Backbone | Targets | AUROC | AUPRC | Sensitivity | Specificity | Balanced min |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny | Angle-6 | 0.737 | 0.235 | 0.707 | 0.722 | 0.674 |
| ConvNeXt-Small | Angle-6 | 0.708 | 0.189 | 0.758 | 0.655 | 0.655 |

These findings argue against continued broad architecture scaling in the current
dataset. The limiting factors appear more consistent with label ambiguity,
calibration instability, and image/view quality than with insufficient backbone
capacity.

### Error Analysis and Remaining Bottlenecks

Across the best strict-label model, false negatives were concentrated among
Shaffer grade `0/1` eyes, particularly grade `1`, while false positives were
common among grade `2` and grade `3` eyes. After excluding grade `2`, many false
positives remained grade `3`. This pattern suggests three non-exclusive
explanations: some non-closed eyes may have narrow-looking anterior chamber
features in slit-lamp images; image quality or view labeling may corrupt
eye-level averaging; and eye-level Shaffer labels may not fully capture the
region photographed by slit-lamp images.

**Table 8. Development conclusions and next actions**

| Finding | Evidence | Next action |
| --- | --- | --- |
| Direct classification is weak | AUROC 0.317 in the first image-only classifier | Do not use plain binary image classification as the main method. |
| Anatomy supervision is the strongest signal | ResNet and ConvNeXt anatomy stacks outperform direct classification | Keep anatomy-regression stack as the central method. |
| Angle-focused targets help | Angle-6 improved balanced performance over all-10 frozen ConvNeXt | Use ACD, lens vault, AOD500, and TISA500 as primary targets. |
| Regularized fine-tuning helps | Exp9 improved AUROC to 0.737 | Keep ConvNeXt-Tiny fine-tuning with conservative regularization. |
| Grade 2 is ambiguous but not the only issue | Grade-2 exclusion improved sens/spec but did not improve stability | Treat grade-2 exclusion as sensitivity analysis and perform clinical review. |
| Threshold transfer is unstable | Other-fold thresholds degraded balanced-min | Improve calibration only after label/image cleanup. |
| Larger backbone is not better | ConvNeXt-Small reduced AUROC and specificity | Stop architecture scaling for now. |

## Publication-Oriented Interpretation

The experiments show a clear development trajectory: direct classification was
insufficient, anatomy-supervised learning improved the signal, angle-focused
targets and regularized fine-tuning produced the best strict-label performance,
and subsequent stress tests identified label ambiguity and threshold stability
as the main remaining bottlenecks. This is a useful problem-solving narrative
for publication because it explains why the final model is anatomy-supervised
rather than a generic image classifier.

However, the current evidence should be described as internal development and
patient-level cross-validation. The model has not yet reached stable `80/80`
performance, and no external validation cohort has been tested. A conservative
conclusion is:

> Slit-lamp photographs contain measurable signal for angle-closure referral
> triage when supervised by AS-OCT anterior chamber anatomy. The best current
> model achieved mean AUROC 0.737 with sensitivity 0.707 and specificity 0.722
> in patient-level 5-fold validation, but performance remains limited by label
> ambiguity, calibration instability, and image-quality variation. These results
> support anatomy-supervised model development and motivate prospective label
> review and external validation before clinical deployment claims.

## Recommended Next Manuscript Figure/Table Package

For an NPJ-style submission package, include:

1. Study flow diagram: records to complete-case eyes to image rows.
2. Model schematic: slit-lamp images to predicted AS-OCT anatomy to shallow
   angle-closure risk.
3. Experiment progression figure: AUROC versus experiment milestone.
4. Main performance table: strict-label exp9 with fold-level and mean metrics.
5. Sensitivity table: grade-2 excluded analysis.
6. Calibration/threshold table: fold-internal versus other-fold thresholding.
7. Error-analysis table: false negatives/false positives by Shaffer grade.
8. Limitations paragraph: internal validation only, rare positive class,
   eye-level rather than quadrant-resolved gonioscopy, no external validation.
