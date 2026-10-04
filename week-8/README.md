# Week 8 — Drift and Observability Monitoring (Track B)

**Course:** DS5619 Machine Learning Systems Operations  
**Student ID:** 142301024  
**Deterministic Seed:** 706208327  
**Dataset / Domain:** BMD-45 Urban Traffic CCTV Vehicle Detection  

---

## Executive Summary

In production machine learning systems, models degrade silently when real-world distributions drift away from training distributions. In Sharma et al. (BMD-45, CVPR 2026 Findings), detectors trained on external benchmarks experienced a severe domain shift on real-world CCTV footage—dropping from ~83.8% mAP to ~33.6% mAP.

Because ground-truth annotations are unavailable at inference time, production pipelines rely on **unsupervised ML observability** techniques. This lab implements a complete drift monitoring subsystem using the **Population Stability Index (PSI)** applied to the vehicle detector's confidence scores ($P(\hat{Y})$).

Using student roll number `142301024`, synthetic CCTV images simulating two distinct cameras (`camera_A_daylight` as reference, `camera_B_lowlight` as live) were generated. The implemented drift monitor calculates score distributions, evaluates PSI, and flags distribution shifts against standardized industry thresholds.

---

## Architecture and Data Flow

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               Production Drift Pipeline                                │
└────────────────────────────────────────────────────────────────────────────────────────┘
          │                                                    │
          ▼                                                    ▼
┌───────────────────────────┐                        ┌───────────────────────────┐
│ Reference Camera Images   │                        │   Live Camera Traffic     │
│ (camera_A_daylight: *.jpg)│                        │ (camera_B_lowlight: *.jpg)│
└─────────────┬─────────────┘                        └─────────────┬─────────────┘
              │                                                    │
              ▼                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        extract_confidence_scores(camera_dir)                           │
│  - Glob sorted *.jpg images (deterministic traversal)                                  │
│  - Load with PIL (Image.open().convert("RGB"))                                         │
│  - Run mock_detector.detect() (flood fill on reddish blobs)                            │
│  - Collect float confidence scores: score = min(0.98, 0.4 + |pts| / (bw * bh + 1))    │
└─────────────┬────────────────────────────────────────────────────┬─────────────────────┘
              │                                                    │
              │ reference_scores                                   │ live_scores
              ▼                                                    ▼
┌───────────────────────────┐                        ┌───────────────────────────┐
│  summarize_scores(ref)    │                        │  summarize_scores(live)   │
│  count, mean, std, min,max│                        │  count, mean, std, min,max│
└─────────────┬─────────────┘                        └─────────────┬─────────────┘
              │                                                    │
              └──────────────────────────┬─────────────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       compute_psi(ref_scores, live_scores, n_bins=10)                  │
│  - Equal-width binning on [0.0, 1.0]                                                   │
│  - Proportion calculation: ref_pct, live_pct                                           │
│  - Regularization: clamp min proportion to 1e-4                                        │
│  - Formula: PSI = Σ (live_pct - ref_pct) * ln(live_pct / ref_pct)                      │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼ psi
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                classify_drift(psi)                                     │
│     psi < 0.10: "none"  |  0.10 <= psi < 0.25: "moderate"  |  psi >= 0.25: "significant"│
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              drift_report.json                                         │
│  JSON report with reference/live summaries, computed PSI, and alert level              │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical and Theoretical Foundations

### 1. Population Stability Index (PSI)

The Population Stability Index measures the distance between a reference probability distribution $Q$ and a monitored (live) probability distribution $P$ over discretized bins:

$$\text{PSI} = \sum_{b=1}^{B} \left( P_b - Q_b \right) \times \ln\left( \frac{P_b}{Q_b} \right)$$

where:
- $B$ is the number of bins (default: $B = 10$).
- $P_b = \frac{\text{Count}_{\text{live}, b}}{N_{\text{live}}}$ is the observed proportion in bin $b$.
- $Q_b = \frac{\text{Count}_{\text{ref}, b}}{N_{\text{ref}}}$ is the reference proportion in bin $b$.

#### Relation to Kullback-Leibler (KL) Divergence
PSI is mathematically equivalent to the **symmetric Kullback-Leibler divergence** (also known as Jeffrey divergence):

$$\text{PSI} = D_{\text{KL}}(P \parallel Q) + D_{\text{KL}}(Q \parallel P)$$

$$\text{PSI} = \sum_{b=1}^{B} P_b \ln\left(\frac{P_b}{Q_b}\right) + \sum_{b=1}^{B} Q_b \ln\left(\frac{Q_b}{P_b}\right) = \sum_{b=1}^{B} (P_b - Q_b) \ln\left(\frac{P_b}{Q_b}\right)$$

#### Properties of PSI:
1. **Non-negativity**: $\text{PSI} \ge 0$.
   - When $P_b > Q_b \implies (P_b - Q_b) > 0$ and $\ln(P_b / Q_b) > 0 \implies \text{term} > 0$.
   - When $P_b < Q_b \implies (P_b - Q_b) < 0$ and $\ln(P_b / Q_b) < 0 \implies \text{term} > 0$.
2. **Zero at Equality**: If $P_b = Q_b$ for all $b$, $\text{PSI} = 0.0$.
3. **Symmetric**: $\text{PSI}(P, Q) = \text{PSI}(Q, P)$.

### 2. Zero-Frequency Clamping / Regularization

When a bin contains zero occurrences in either distribution, $\frac{P_b}{Q_b}$ causes division by zero or $\ln(0)$ (undefined).
To guarantee numerical stability, every proportion is bounded from below by a small epsilon:

$$P'_b = \max(P_b, 10^{-4}), \quad Q'_b = \max(Q_b, 10^{-4})$$

This maintains numerical stability while still heavily penalizing the emergence of values in regions previously unseen.

### 3. Drift Classification Thresholds

Industry standard empirical thresholds categorize drift severity:

| PSI Value Range | Drift Classification | Action Required in Production |
|---|---|---|
| $\text{PSI} < 0.10$ | `"none"` | No significant distribution change; baseline operations. |
| $0.10 \le \text{PSI} < 0.25$ | `"moderate"` | Moderate drift detected; increase monitoring frequency, log samples, prepare retraining pipeline. |
| $\text{PSI} \ge 0.25$ | `"significant"` | Severe distribution divergence; alert on-call engineer, trigger automated retraining or fallback to baseline model. |

---

## Detailed Code Walkthrough (`src/drift_monitor.py`)

All four required functions were implemented according to specifications:

### Part 1: Monitored Feature Extraction

```python
def extract_confidence_scores(camera_dir):
    scores = []
    pattern = os.path.join(camera_dir, "*.jpg")
    for img_path in sorted(glob.glob(pattern)):
        with Image.open(img_path) as raw_img:
            img = raw_img.convert("RGB")
            detections = det.detect(img)
            for d in detections:
                scores.append(float(d.score))
    return scores
```

**Key Implementation Details:**
- **Determinism**: `sorted(glob.glob(...))` guarantees consistent file traversal across operating systems and file system implementations.
- **Resource Management**: Uses a context manager (`with Image.open(...)`) to prevent OS file descriptor leaks when iterating through image archives.
- **Conversion**: `.convert("RGB")` standardizes channels before passing to the detector.
- **Type Safety**: Extracts `d.score` explicitly cast to `float`.

### Part 2: PSI Computation

```python
def compute_psi(reference_scores, live_scores, n_bins=N_BINS):
    def _bin_index(score):
        idx = int(score * n_bins)
        return min(max(idx, 0), n_bins - 1)

    ref_counts = [0] * n_bins
    for s in reference_scores:
        ref_counts[_bin_index(s)] += 1

    live_counts = [0] * n_bins
    for s in live_scores:
        live_counts[_bin_index(s)] += 1

    ref_total = len(reference_scores) or 1
    live_total = len(live_scores) or 1

    psi = 0.0
    for i in range(n_bins):
        ref_pct = max(ref_counts[i] / ref_total, 1e-4)
        live_pct = max(live_counts[i] / live_total, 1e-4)
        psi += (live_pct - ref_pct) * math.log(live_pct / ref_pct)

    return float(psi)
```

**Key Implementation Details:**
- **Bin Index Calculation**: Maps continuous interval $[0.0, 1.0]$ into $n\_bins$ buckets. A score of $1.0$ yields $\lfloor 1.0 \times 10 \rfloor = 10$, which is clamped via `min(..., n_bins - 1)` to bin index 9.
- **Epsilon Clamping**: Proportions are floored at `1e-4` to prevent math errors.
- **Natural Log**: `math.log()` calculates $\ln$, conforming to standard PSI specifications.

### Part 3: Drift Classification

```python
def classify_drift(psi):
    if psi < PSI_MODERATE_THRESHOLD:
        return "none"
    elif psi < PSI_SIGNIFICANT_THRESHOLD:
        return "moderate"
    else:
        return "significant"
```

**Key Implementation Details:**
- Evaluates strict inequalities and inclusive lower bounds to respect boundary conditions:
  - $0.09999 \implies$ `"none"`
  - $0.10000 \implies$ `"moderate"`
  - $0.25000 \implies$ `"significant"`

### Part 4: Distribution Summary Statistics

```python
def summarize_scores(scores):
    if not scores:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    count = len(scores)
    mean_val = round(float(statistics.mean(scores)), 4)
    std_val = round(float(statistics.stdev(scores)), 4) if count >= 2 else 0.0
    min_val = round(float(min(scores)), 4)
    max_val = round(float(max(scores)), 4)

    return {
        "count": count,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
    }
```

**Key Implementation Details:**
- **Single-Value & Zero-Value Resilience**: `statistics.stdev()` raises `StatisticsError` when $N < 2$. The conditional `std_val = ... if count >= 2 else 0.0` prevents runtime exceptions.
- **Precision**: All float fields are rounded to 4 decimal places.

---

## Mock Detector Mechanics (`src/mock_detector.py`)

The mock detector provides a lightweight, GPU-free surrogate for real detectors (YOLOv12 / RT-DETRv2 / D-FINE) allowing fast CI testing.

1. **Pixel Thresholding**:
   ```python
   def _is_reddish(pixel):
       r, g, b = pixel
       return r > 120 and r - g > 40 and r - b > 40
   ```
2. **Connected Component Flood Fill**:
   Iterative Depth-First Search (DFS) grouping 4-connected reddish pixels into bounding boxes:
   $$[bx, by, bw, bh]$$
3. **Confidence Scoring**:
   $$\text{score} = \min\left(0.98, 0.4 + \frac{|pts|}{bw \times bh + 1}\right)$$
   Confidence is proportional to fill density (solid vehicle vs noisy boundary) and capped at 0.98. Detections with $\text{score} < 0.30$ are filtered out.

---

## Student Fixture Generation (`generate_for_student.py`)

Running fixture generation for Student ID `142301024`:
```bash
python generate_for_student.py --student-id 142301024
```

### Deterministic Seed Derivation:
```python
base_seed = seed_from_student_id("142301024", salt="week08") # -> 706208327
```

### Synthetic Camera Profiles:
- **Reference Camera (`camera_A_daylight`)**:
  - Background RGB: `(170, 175, 180)` (bright overcast/daylight)
  - Gaussian Noise Level: `8`
  - Bounding Box Scale Factor: `1.0`
- **Live Camera (`camera_B_lowlight`)**:
  - Background RGB: `(60, 62, 70)` (night / low-light)
  - Gaussian Noise Level: `22` (sensor gain noise)
  - Bounding Box Scale Factor: `0.7` (distance / perspective variance)

### Reseeding Guard:
`generate_for_student.py` checks candidate fixtures against the mock detector. If pure randomness results in $PSI < 0.10$, it increments the sub-seed until a moderate drift signal is achieved.

**Run Result:**
- Seed: `706208327`
- 40 images (20 per camera profile)
- 149 bounding box annotations in `_annotations.coco.json`
- Sanity-checked PSI: `0.1346`

---

## Pipeline Execution and Drift Report

Executing the complete driver pipeline:
```bash
python src/run_pipeline.py
```

### Generated `drift_report.json`:
```json
{
  "reference_camera": "camera_A_daylight",
  "live_camera": "camera_B_lowlight",
  "reference_summary": {
    "count": 73,
    "mean": 0.9698,
    "std": 0.0524,
    "min": 0.624,
    "max": 0.98
  },
  "live_summary": {
    "count": 75,
    "mean": 0.9767,
    "std": 0.029,
    "min": 0.729,
    "max": 0.98
  },
  "psi": 0.1346,
  "drift_level": "moderate",
  "thresholds": {
    "moderate": 0.1,
    "significant": 0.25
  }
}
```

### Distribution Comparison:

| Metric | Reference (`camera_A_daylight`) | Live (`camera_B_lowlight`) | Delta ($\Delta$) |
|---|---|---|---|
| **Detections Count** | 73 | 75 | +2 |
| **Mean Confidence** | 0.9698 | 0.9767 | +0.0069 |
| **Standard Deviation** | 0.0524 | 0.0290 | -0.0234 |
| **Min Confidence** | 0.6240 | 0.7290 | +0.1050 |
| **Max Confidence** | 0.9800 | 0.9800 | 0.0000 |
| **Population Stability Index (PSI)** | — | **0.1346** | Moderate Drift |

**Interpretation:**
In `camera_B_lowlight`, the higher noise and darker background cause small, faint bounding boxes to either fail the reddish threshold or produce very compact components with high fill density. This compressed the distribution variance ($\sigma = 0.0524 \to 0.0290$) and eliminated low-confidence detections ($min = 0.6240 \to 0.7290$). The resulting shift across probability bins produces a PSI of `0.1346`, correctly triggering a `"moderate"` drift alert.

---

## Observability Deep Dive: Strengths and Blind Spots

### 1. Why Monitor Confidence Scores?
In production, true labels $Y$ are typically absent at prediction time. Model outputs $\hat{Y}$ and confidence scores $s \in [0, 1]$ are always available synchronously. Significant changes in confidence distributions serve as early warnings that the data generating process has shifted.

### 2. What Confidence-Score-Only Monitoring Misses

| Drift Type | Mathematical Definition | Can Confidence Monitoring Catch It? | Why / Why Not? |
|---|---|---|---|
| **Covariate Shift** (Data Drift) | $P(X)$ changes, $P(Y \mid X)$ unchanged | **Partially** | Catches shifts when visual degradation directly reduces confidence scores. Misses shifts if degraded inputs happen to produce similar score distributions. |
| **Concept Drift** (Relationship Drift) | $P(Y \mid X)$ changes, $P(X)$ may/may not change | **NO (Completely Misses)** | The model can be **confidently wrong**. If background noise or novel objects trigger detections with 0.95 confidence, the score distribution appears healthy while actual precision drops to zero. |
| **Prior Probability Shift** (Label Drift) | $P(Y)$ changes, $P(X \mid Y)$ unchanged | **NO** | Changes in true vehicle distribution (e.g. surge in bicycles vs trucks) cannot be detected from raw confidence magnitude. |
| **False Negatives** (Silent Misses) | Ground truth vehicle present, $\hat{Y} = \emptyset$ | **NO (Completely Misses)** | Missed vehicles generate zero detection records. Because no score is emitted, the score distribution of remaining detections appears normal while recall drops. |

### 3. Comprehensive Observability Roadmap with Delayed Ground Truth

When ground-truth labels arrive (e.g., via human auditing, crowdsourced verification, or downstream traffic logs):

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        Comprehensive ML Observability Framework                        │
├────────────────────────────┬─────────────────────────────┬─────────────────────────────┤
│ 1. Input Features P(X)     │ 2. Predictions P(Ŷ)         │ 3. Performance & P(Y|X)     │
│    (Unsupervised)          │    (Unsupervised)           │    (Supervised, Delayed)    │
├────────────────────────────┼─────────────────────────────┼─────────────────────────────┤
│ - Mean scene luminance     │ - Detection confidence PSI  │ - Mean Average Precision:   │
│ - Pixel noise variance     │ - Bounding box area PSI     │   mAP@[.50:.95], mAP@50     │
│ - Laplacian image blur     │ - Aspect ratio distribution │ - Precision, Recall, F1     │
│ - Weather/glare heuristics │ - Detections count per frame│ - False Negative Rate (FNR) │
│ - Missing frame rate       │ - Category distribution P(Ŷ)│ - Per-class confusion matrix│
└────────────────────────────┴─────────────────────────────┴─────────────────────────────┘
```

---

## Test Suite & Verification

The suite (`tests/test_smoke.py`) validates core edge cases, mathematical correctness, determinism, and end-to-end execution.

```
============================= test session starts ==============================
platform darwin -- Python 3.14.0, pytest-8.3.3, pluggy-1.6.0
collected 9 items

tests/test_smoke.py::test_extract_confidence_scores_returns_floats_in_range PASSED
tests/test_smoke.py::test_extract_confidence_scores_is_deterministic PASSED
tests/test_smoke.py::test_compute_psi_is_zero_for_identical_distributions PASSED
tests/test_smoke.py::test_compute_psi_is_positive_for_shifted_distributions PASSED
tests/test_smoke.py::test_compute_psi_small_for_similar_distributions PASSED
tests/test_smoke.py::test_classify_drift_bands PASSED
tests/test_smoke.py::test_summarize_scores_shape PASSED
tests/test_smoke.py::test_summarize_scores_single_value_has_zero_std PASSED
tests/test_smoke.py::test_full_pipeline_runs_and_flags_camera_drift PASSED

============================== 9 passed in 1.18s ===============================
```

### Verification Matrix:
- `test_extract_confidence_scores_returns_floats_in_range`: Confirms all returned scores are floats bounded in $[0.0, 1.0]$.
- `test_extract_confidence_scores_is_deterministic`: Verifies identical score ordering across repeated invocations.
- `test_compute_psi_is_zero_for_identical_distributions`: Confirms $\text{PSI} < 10^{-6}$ for identical input sets.
- `test_compute_psi_is_positive_for_shifted_distributions`: Confirms $\text{PSI} > 1.0$ for non-overlapping shifted distributions ($[0.1, 0.15]$ vs $[0.85, 0.90]$).
- `test_compute_psi_small_for_similar_distributions`: Confirms $\text{PSI} < 0.10$ for minor sampling perturbations.
- `test_classify_drift_bands`: Verifies boundary values $0.05 \to$ `"none"`, $0.09999 \to$ `"none"`, $0.10 \to$ `"moderate"`, $0.20 \to$ `"moderate"`, $0.25 \to$ `"significant"`, $0.50 \to$ `"significant"`.
- `test_summarize_scores_shape`: Verifies count, mean, min, max, std keys.
- `test_summarize_scores_single_value_has_zero_std`: Ensures $N=1$ sets $\text{std}=0.0$ without raising `StatisticsError`.
- `test_full_pipeline_runs_and_flags_camera_drift`: Executes `run_pipeline.py` in an isolated temporary sandbox and verifies valid report emission.

---

## Deliverables Summary

1. `src/drift_monitor.py`: Complete implementation with all 4 functions verified and flake8-clean.
2. `data/fixtures/`: Synthetic images generated deterministically with student ID `142301024`.
3. `drift_report.json`: Valid JSON report generated by `src/run_pipeline.py` showing moderate drift ($PSI = 0.1346$).
4. `NOTES.md`: Documented student ID (`142301024`), seed (`706208327`), drift level vs expectation, and lecture concept drift analysis.
5. `README.md` & `README.pdf`: Complete technical design document and compiled PDF reference.
