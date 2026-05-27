# Goal 70/70 Cross-Validation Findings

Date: 2026-05-26

Target: strict Shaffer grade `0/1` angle closure versus grade `2/3/4` non-closure.

Relaxed operating goal: sensitivity >= 0.70 and specificity >= 0.70.

## Models Validated

Two candidates were tested with patient-level 5-fold validation because they were the only serious candidates under the relaxed goal:

1. `angle_closure_image_only_multitask_5fold`
   - Image-only ConvNeXt multitask model.
   - Shallow meta-risk selected inside each validation fold.

2. `angle_closure_roi_local_combined_aod_tisa_5fold`
   - Shared-split nasal/temporal Van Herick beam-ROI biomarker models.
   - Local AOD500/TISA500 predictions merged per eye.
   - Shallow meta-risk selected inside each validation fold.
   - Patient-overlap checks passed for all folds.

## 5-Fold Test Summary

| Candidate | Mean AUROC | Mean Sens | Mean Spec | Mean PPV | Mean NPV | Mean Balanced Min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Image-only multitask + meta-risk | 0.643 | 0.349 | 0.773 | 0.126 | 0.922 | 0.343 |
| ROI local AOD/TISA + meta-risk | 0.674 | 0.536 | 0.659 | 0.157 | 0.933 | 0.518 |

## Interpretation

Neither candidate met the relaxed >=0.70 sensitivity and >=0.70 specificity target under patient-level 5-fold validation.

The ROI local model improved mean sensitivity over the image-only multitask model, but specificity dropped below 0.70 and performance varied substantially by fold. The fixed-split >70 validation signal was therefore not stable.

## Decision

Do not advance these two candidates as final >70/70 models.

For the current paper, the defensible claim remains high-sensitivity angle-closure referral triage, not balanced open/closed angle classification. To reach a stable 70/70 classifier, the next change should address data/label signal rather than another small architecture or threshold tweak.

Most efficient next options:

1. Add more positive angle-closure eyes or use repeated cross-validation with more positive cases per validation fold.
2. Train a direct model to predict the measured strongest anatomy signal, especially true ACD/lens-vault and true AOD/TISA, then calibrate a risk model.
3. Build an explicit Van Herick geometry extractor from the slit beam and limbal corneal/anterior-chamber width instead of relying only on ConvNeXt features.
4. Review false negatives and label ambiguity, especially grade 1 versus grade 2 eyes.
