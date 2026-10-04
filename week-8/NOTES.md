# NOTES.md — Week 8: Drift and Observability Monitoring

**Student ID used with `generate_for_student.py`:**
142301024

**Seed generated:**
706208327


## Drift level vs. expectation

The drift report showed a PSI (Population Stability Index) of **0.1346**, which classifies as **moderate drift** (`0.10 <= PSI < 0.25`).

This aligns directly with expectations. The two synthetic camera streams were synthesized with deliberately different visual profiles:
- `camera_A_daylight`: background RGB `(170, 175, 180)`, low noise level `8`, and vehicle bounding box scale `1.0`.
- `camera_B_lowlight`: low-light background RGB `(60, 62, 70)`, high noise level `22`, and smaller bounding box scale `0.7`.

In the mock detector, confidence score is determined by blob density and bounding box dimensions (`score = min(0.98, 0.4 + len(pts) / (bw * bh + 1))`). The darker background, higher pixel noise, and compressed box sizes in `camera_B_lowlight` alter the resulting flood-fill component geometry and score distributions relative to `camera_A_daylight` (reference mean: 0.9698, std: 0.0524 vs live mean: 0.9767, std: 0.0290), producing a measurable shift that cleanly clears the moderate threshold (`0.10`) while remaining below significant divergence (`0.25`).


## What confidence-score-only monitoring misses

Confidence-score monitoring tracks prediction/output distribution drift ($P(\hat{Y})$) and can serve as an indirect proxy for covariate shift ($P(X)$) when image degradation impacts detector confidence. However:

1. **Misses Concept Drift ($P(Y \mid X)$)**:
   Confidence monitoring cannot detect when the relationship between input features and actual ground truth changes. A model can be **confidently wrong**—for example, if environmental noise or background artifacts trigger high-confidence false positive detections (e.g. predicting a vehicle on a bright patch or lamppost with 0.95 confidence), or if vehicles are misclassified into wrong categories with high confidence. The confidence score distribution remains elevated and looks healthy, masking a severe drop in precision.

2. **Misses False Negatives (Silent Misses / Recall Drop)**:
   If low-light conditions cause vehicles to fall below the detection threshold or fail flood fill entirely, the detector produces *no* detection. Because no detection exists, no confidence score is emitted or recorded in the distribution. Consequently, the score distribution of the few vehicles that *were* detected can appear completely normal while actual model recall and mAP collapse.

3. **What to monitor IN ADDITION to confidence score with delayed ground-truth labels:**
   - **Ground-Truth Performance Metrics**: Mean Average Precision (**mAP@[.50:.95]**, **mAP@50**), Precision, Recall, and F1-score across IoU thresholds to directly evaluate detection accuracy, false positive rate (FPR), and false negative rate (FNR).
   - **Ground-Truth Label Distribution Drift ($P(Y)$)**: Comparing the frequency and distribution of actual vehicle categories over time (e.g. sudden surge in two-wheelers or heavy commercial vehicles).
   - **Error / Confusion Segmentation**: Per-class error rates and confusion matrices to uncover whether specific vehicle classes or illumination conditions suffer disproportionate degradation.
   - **Input Covariate Statistics ($P(X)$)**: Image-level metrics (mean luminance, contrast, sharpness/Laplacian variance, noise) to diagnose root causes before delayed labels arrive.
