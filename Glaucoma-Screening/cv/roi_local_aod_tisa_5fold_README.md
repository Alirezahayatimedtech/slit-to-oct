# Local ROI Biomarker + Metadata 5-Fold CV

Each fold combines separately trained nasal and temporal Van Herick ROI biomarker models. The shallow meta-risk model and threshold are selected inside the fold validation set only, then reported on that fold's held-out test set.

## Test Metric Summary

| metric | folds | mean | std | min | max |
| --- | --- | --- | --- | --- | --- |
| auroc | 5 | 0.674 | 0.138 | 0.503 | 0.839 |
| auprc | 5 | 0.220 | 0.147 | 0.098 | 0.439 |
| sensitivity | 5 | 0.536 | 0.124 | 0.429 | 0.750 |
| specificity | 5 | 0.659 | 0.139 | 0.558 | 0.897 |
| ppv | 5 | 0.157 | 0.073 | 0.105 | 0.273 |
| npv | 5 | 0.933 | 0.020 | 0.915 | 0.962 |
| accuracy | 5 | 0.649 | 0.126 | 0.553 | 0.859 |
| balanced_min | 5 | 0.518 | 0.086 | 0.429 | 0.662 |

## Fold Test Rows

| fold | feature_set | model | threshold_name | auroc | sensitivity | specificity | ppv | npv | balanced_min | tp | fp | tn | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | clinical_metadata | logreg_balanced | youden_from_val | 0.597 | 0.500 | 0.558 | 0.105 | 0.915 | 0.500 | 4 | 34 | 43 | 4 |
| 2 | clinical_metadata | logreg_balanced | balanced_from_val | 0.644 | 0.500 | 0.588 | 0.108 | 0.922 | 0.500 | 4 | 33 | 47 | 4 |
| 3 | angle_all | rf_depth2 | youden_from_val | 0.786 | 0.429 | 0.897 | 0.273 | 0.946 | 0.429 | 3 | 8 | 70 | 4 |
| 4 | temporal_angle_plus_clinical | rf_depth2 | youden_from_val | 0.839 | 0.750 | 0.662 | 0.188 | 0.962 | 0.662 | 6 | 26 | 51 | 2 |
| 5 | direct_prob | raw | balanced_from_val | 0.503 | 0.500 | 0.590 | 0.111 | 0.920 | 0.500 | 4 | 32 | 46 | 4 |