# Image-Only Multitask 5-Fold Meta-Risk Summary

Candidate selection is performed inside each fold validation set. Reported metrics below are for held-out fold test sets.

## Validation-Selected Meta-Risk Test Metrics by Fold

| fold | feature_set | model | threshold_name | n | positives | auroc | sensitivity | specificity | ppv | npv | balanced_min | tp | fp | tn | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | all_pred_biomarkers | logreg_balanced | balanced_from_val | 107 | 10 | 0.613 | 0.000 | 0.835 | 0.000 | 0.890 | 0.000 | 0 | 16 | 81 | 10 |
| 2 | nasal_angle | logreg_balanced | youden_from_val | 107 | 10 | 0.707 | 0.700 | 0.670 | 0.179 | 0.956 | 0.670 | 7 | 32 | 65 | 3 |
| 3 | compact_nasal_anatomy | extratrees_depth2 | youden_from_val | 105 | 10 | 0.731 | 0.300 | 0.789 | 0.130 | 0.915 | 0.300 | 3 | 20 | 75 | 7 |
| 4 | prob_plus_all_pred | logreg_balanced | youden_from_val | 105 | 9 | 0.481 | 0.444 | 0.854 | 0.222 | 0.943 | 0.444 | 4 | 14 | 82 | 5 |
| 5 | nasal_angle | extratrees_depth2 | youden_from_val | 105 | 10 | 0.682 | 0.300 | 0.716 | 0.100 | 0.907 | 0.300 | 3 | 27 | 68 | 7 |

## Meta-Risk Summary

| metric | folds | mean | std | min | max |
| --- | --- | --- | --- | --- | --- |
| auroc | 5 | 0.643 | 0.100 | 0.481 | 0.731 |
| auprc | 5 | 0.234 | 0.082 | 0.121 | 0.341 |
| sensitivity | 5 | 0.349 | 0.254 | 0.000 | 0.700 |
| specificity | 5 | 0.773 | 0.078 | 0.670 | 0.854 |
| ppv | 5 | 0.126 | 0.085 | 0.000 | 0.222 |
| npv | 5 | 0.922 | 0.027 | 0.890 | 0.956 |
| accuracy | 5 | 0.734 | 0.061 | 0.673 | 0.819 |
| balanced_min | 5 | 0.343 | 0.244 | 0.000 | 0.670 |

## Direct Probability Test Metrics by Fold

| fold | n | positives | auroc | sensitivity | specificity | ppv | npv | tp | fp | tn | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | 107.000 | 10.000 | 0.609 | 0.300 | 0.670 | 0.086 | 0.903 | 3.000 | 32.000 | 65.000 | 7.000 |
| 2.000 | 107.000 | 10.000 | 0.597 | 0.200 | 0.845 | 0.118 | 0.911 | 2.000 | 15.000 | 82.000 | 8.000 |
| 3.000 | 105.000 | 10.000 | 0.694 | 0.900 | 0.484 | 0.155 | 0.979 | 9.000 | 49.000 | 46.000 | 1.000 |
| 4.000 | 105.000 | 9.000 | 0.519 | 0.444 | 0.781 | 0.160 | 0.938 | 4.000 | 21.000 | 75.000 | 5.000 |
| 5.000 | 105.000 | 10.000 | 0.642 | 0.400 | 0.589 | 0.093 | 0.903 | 4.000 | 39.000 | 56.000 | 6.000 |