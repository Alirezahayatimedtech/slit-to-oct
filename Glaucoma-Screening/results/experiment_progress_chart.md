# AUROC Progress Chart

This chart tracks only the milestone experiments that changed the modeling direction or produced a meaningful AUROC jump.

![AUROC progress](experiment_progress_chart.svg)

Update workflow:

```bash
python Glaucoma-Screening/scripts/update_experiment_progress_chart.py
```

Source data: `Glaucoma-Screening/results/experiment_progress_milestones.csv`.

Important: fixed-split, 80/20, and 5-fold CV points are shown together for history. They should not be interpreted as directly equivalent evidence.

| # | Method | Validation | AUROC | Sens | Spec | Why it matters |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 1 | Whole-image direct binary classifier | Fixed split diagnostic | 0.317 | 0.250 | 0.630 | Weak direct classifier; pushed us away from plain binary heads. |
| 2 | Image multitask model plus shallow anatomical risk | Fixed split validation-selected | 0.702 | 0.875 | 0.630 | First useful jump from predicting anatomy-related outputs instead of only class probability. |
| 3 | Weighted Shaffer-grade regression with balanced threshold | 80/20 validation | 0.755 | 0.700 | 0.755 | Continuous grade prediction improved the simple image-only signal. |
| 4 | ResNet-50 predicts 10 AS-OCT anatomy targets; logistic risk | 80/20 validation | 0.764 | 0.714 | 0.721 | Best quick 80/20 strict-label anatomy-stack signal. |
| 5 | Frozen ConvNeXt-Tiny predicts all 10 anatomy targets; mean eye aggregation | 5-fold CV | 0.655 | 0.657 | 0.642 | First rigorous 5-fold ConvNeXt baseline; lower than 80/20 because validation is stricter. |
| 6 | Frozen ConvNeXt-Tiny with angle-focused 6 anatomy targets | 5-fold CV | 0.662 | 0.663 | 0.654 | Small gain from focusing targets on ACD/lens vault/AOD/TISA. |
| 7 | Unfrozen ConvNeXt-Tiny all-10 anatomy model | 5-fold CV | 0.684 | 0.679 | 0.668 | Backbone adaptation improved AUROC but showed overfitting. |
| 8 | Regularized unfrozen ConvNeXt-Tiny with angle-6 targets | 5-fold CV | 0.737 | 0.707 | 0.722 | Best current 5-fold result; first mean sensitivity and specificity both above 0.70. |
