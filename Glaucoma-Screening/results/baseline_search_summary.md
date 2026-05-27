# Angle-Closure Baseline Search Summary

Target label: Shaffer grade 0/1 positive; grades 2/3/4 negative; missing/not-seen excluded. Fixed patient-level train/validation/test split; no folds during model search.

## 2026-05-26 Correction

The first local nasal/temporal ROI combined runs were diagnostic only because nasal and temporal models had been split independently. That can leak a participant across train/validation/test when the two view-specific prediction files are merged. The combine script now raises an error if patient overlap is detected, and valid local-view runs must use the same `--split-csv`.

Corrected shared-split ROI results did **not** reach the >0.80 sensitivity / >0.80 specificity target. The best corrected shared-split candidate was local ROI AOD/TISA plus clinical metadata, validation-selected on the shared validation set: validation sensitivity 0.833, specificity 0.738; test sensitivity 0.500, specificity 0.726. The auxiliary ACD/lens-vault version was worse.

Measured AS-OCT biomarker oracle analysis showed that even true AOD/TISA selected on this small validation set did not reach 80/80 on validation, although true anatomy can separate the test split better. This suggests the immediate bottleneck is both small positive validation count and image-to-biomarker prediction quality, not another minor classifier-head tweak.

| Run | Classifier | Selection | AUROC | Sens | Spec | PPV | NPV | TP/FP/TN/FN |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| old_metadata_multitask_alpha025 | direct_probability | threshold_from_validation_youden | 0.560 | 0.625 | 0.548 | 0.132 | 0.930 | 5/33/40/3 |
| pure_image_classifier | direct_probability | threshold_from_validation_youden | 0.317 | 0.250 | 0.630 | 0.069 | 0.885 | 2/27/46/6 |
| image_only_multitask_baseline | direct_probability | threshold_from_validation_youden | 0.663 | 0.875 | 0.452 | 0.149 | 0.971 | 7/40/33/1 |
| image_only_multitask_baseline | best_shallow_meta_risk_by_validation | candidate_and_threshold_selected_on_validation | 0.702 | 0.875 | 0.630 | 0.206 | 0.979 | 7/27/46/1 |
| image_only_multitask_baseline | best_shallow_meta_risk_test_diagnostic | diagnostic_only_sorted_by_test_not_for_final_selection | 0.757 | 0.875 | 0.699 | 0.241 | 0.981 | 7/22/51/1 |
| van_temporal_multitask_quick | direct_probability | threshold_from_validation_youden | 0.649 | 1.000 | 0.404 | 0.150 | 1.000 | 6/34/23/0 |
| van_temporal_multitask_quick | best_single_predicted_biomarker | biomarker_threshold_from_validation_youden | 0.800 | 0.667 | 0.754 | 0.222 | 0.956 | 4/14/43/2 |
| van_temporal_multitask_quick | best_shallow_meta_risk_by_validation | candidate_and_threshold_selected_on_validation | 0.804 | 0.500 | 0.789 | 0.200 | 0.938 | 3/12/45/3 |
| van_temporal_multitask_quick | best_shallow_meta_risk_test_diagnostic | diagnostic_only_sorted_by_test_not_for_final_selection | 0.787 | 0.833 | 0.702 | 0.227 | 0.976 | 5/17/40/1 |
| van_temporal_angle4_regstrong_failed | direct_probability | threshold_from_validation_youden | 0.548 | 0.500 | 0.509 | 0.097 | 0.906 | 3/28/29/3 |
| van_temporal_angle4_regstrong_failed | best_single_predicted_biomarker | biomarker_threshold_from_validation_youden | 0.623 | 0.500 | 0.579 | 0.111 | 0.917 | 3/24/33/3 |
| van_temporal_angle4_regstrong_failed | best_shallow_meta_risk_by_validation | candidate_and_threshold_selected_on_validation | 0.599 | 0.500 | 0.649 | 0.130 | 0.925 | 3/20/37/3 |
| van_temporal_angle4_regstrong_failed | best_shallow_meta_risk_test_diagnostic | diagnostic_only_sorted_by_test_not_for_final_selection | 0.608 | 0.500 | 0.649 | 0.130 | 0.925 | 3/20/37/3 |
| usable_anatomy6_mean_failed | direct_probability | threshold_from_validation_youden | 0.646 | 0.625 | 0.452 | 0.111 | 0.917 | 5/40/33/3 |
| usable_anatomy6_mean_failed | best_single_predicted_biomarker | biomarker_threshold_from_validation_youden | 0.712 | 0.625 | 0.644 | 0.161 | 0.940 | 5/26/47/3 |
| usable_anatomy6_mean_failed | best_shallow_meta_risk_by_validation | candidate_and_threshold_selected_on_validation | 0.689 | 0.500 | 0.658 | 0.138 | 0.923 | 4/25/48/4 |
| usable_anatomy6_mean_failed | best_shallow_meta_risk_test_diagnostic | diagnostic_only_sorted_by_test_not_for_final_selection | 0.707 | 0.625 | 0.671 | 0.172 | 0.942 | 5/24/49/3 |
| roi_local_aod_tisa_sharedsplit | best_shallow_meta_risk_by_validation | shared_split_candidate_selected_on_validation | 0.632 | 0.500 | 0.726 | 0.150 | 0.938 | 3/17/45/3 |
| roi_local_aod_tisa_acd_lv_sharedsplit | best_shallow_meta_risk_by_validation | shared_split_candidate_selected_on_validation | 0.540 | 0.333 | 0.661 | 0.087 | 0.911 | 2/21/41/4 |
