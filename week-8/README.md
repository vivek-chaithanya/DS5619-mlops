# Week 8 — Drift and Observability Monitoring

**Course:** DS5619 Machine Learning Systems Operations  
**Student ID:** `142301024`  
**Seed:** `706208327` (salt: `week08`)

---

## What Was Implemented

In this lab, I built an unsupervised drift monitoring pipeline in `src/drift_monitor.py` to detect distribution shifts between reference daylight CCTV camera data and live low-light camera data using the Population Stability Index (PSI).

The core functions implemented are:

1. **`extract_confidence_scores(camera_dir)`**:
   - Loops over all `.jpg` images in sorted alphabetical order from the camera directory.
   - Runs `mock_detector.detect()` on each image.
   - Extracts and returns a flat list of float confidence scores for all detected vehicles.

2. **`summarize_scores(scores)`**:
   - Calculates basic descriptive statistics: `count`, `mean`, sample standard deviation (`std` with Bessel's correction $N-1$), `min`, and `max`.
   - Returns 0.0 values gracefully if the score list is empty or has fewer than 2 elements.

3. **`compute_psi(reference_scores, live_scores, n_bins=10)`**:
   - Bins confidence scores into 10 equal-width bins across $[0.0, 1.0]$ (e.g. $[0.0, 0.1), [0.1, 0.2), \dots$).
   - Calculates the proportion of samples falling into each bin for both reference and live distributions.
   - Applies a small regularization floor (`1e-4`) to every bin proportion so we never take $\ln(0)$ or divide by zero.
   - Sums the symmetric divergence formula:
     $$\text{PSI} = \sum_{b=1}^{B} (P_{\text{live}, b} - Q_{\text{ref}, b}) \times \ln\left(\frac{P_{\text{live}, b}}{Q_{\text{ref}, b}}\right)$$

4. **`classify_drift(psi)`**:
   - Maps the computed PSI value into standard alert thresholds:
     - `psi < 0.10`: `"none"`
     - `0.10 <= psi < 0.25`: `"moderate"`
     - `psi >= 0.25`: `"significant"`

---

## Results

Running the pipeline (`python src/run_pipeline.py`) compared daylight reference images against live low-light images and generated `drift_report.json`:

| Metric | Reference Camera (`camera_A_daylight`) | Live Camera (`camera_B_lowlight`) |
|:---|:---:|:---:|
| **Detection Count** | 73 | 75 |
| **Mean Confidence** | 0.9698 | 0.9767 |
| **Standard Deviation** | 0.0524 | 0.0290 |
| **Score Range** | [0.6240, 0.9800] | [0.7290, 0.9800] |

- **Calculated PSI:** `0.1346`
- **Drift Classification:** `moderate` (falls in the range $0.10 \le \text{PSI} < 0.25$)

### Key Takeaway:
Low-light camera conditions shifted the confidence distribution enough to cross the 0.10 threshold into **moderate drift**. While not yet a critical failure (significant drift $\ge 0.25$), this alerts the engineering team that the model's environment has shifted and should be monitored for potential retraining or calibration.
