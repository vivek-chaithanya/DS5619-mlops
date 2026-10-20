# NOTES.md — Week 10: Fairness and Robustness Audit

**Student ID used with `generate_for_student.py`:**
`142301024`

**Deterministic Base Seed:**
`2290012310` (derived with salt `week10`, selected attempt 9 with seed `2290012319`)


## Which classes got flagged?

Running the clean fairness audit (`recall_threshold = 0.90`) surfaced three underperforming categories sorted worst-first:

1. **Two-wheeler:** 11/13 detected = **84.6%** recall (`< 90%` threshold)
2. **Other:** 15/17 detected = **88.2%** recall (`< 90%` threshold)
3. **Bicycle:** 16/18 detected = **88.9%** recall (`< 90%` threshold)

All large-vehicle classes passed comfortably above the bar: **Bus** (92.3%), **Truck** (92.6%), **Tempo-traveller** (91.7%), **Mini-bus** (100.0%), **Hatchback** (100.0%), **Sedan** (100.0%), **SUV** (100.0%), and **MUV** (100.0%).

### Does this match what we expect from fixture generation?
**Yes, perfectly.**
In `generate_for_student.py`, vehicle categories are parameterized with size multipliers (`CLASS_SIZE_MULTIPLIER`):
- `Bicycle` has a scale factor of `0.35`
- `Two-wheeler` has a scale factor of `0.45`
- Contrast these with `Bus` (`2.2`), `Truck` (`2.0`), `Mini-bus` (`1.8`), `SUV` (`1.25`), and `Hatchback` (`1.0`).

Because `draw_scene()` scales bounding box dimensions by $bw = \text{base} \times \text{size\_mult} \times \text{box\_scale}$, bicycles and two-wheelers are rendered with systematically smaller drawn boxes (e.g., $bw \times bh \approx 8 \times 6$ to $13 \times 12$). In `mock_detector.py`, object localization depends on flood-filling contiguous reddish pixels with a strict threshold: `len(pts) >= 20`. A small bounding box (especially under low-light noise or camera B's 0.7 box scale) often possesses fewer than 20 interior reddish pixels, causing the detector to skip the entire connected component without emitting a detection. Furthermore, detection confidence scores are penalized for smaller pixel counts ($\text{score} = \min(0.98, 0.4 + \text{len}(pts) / (bw \times bh + 1))$), occasionally falling below the 0.3 score threshold. Thus, the underperformance of **Two-wheeler** and **Bicycle** reflects a **structural disparity** inherent to spatial resolution limits and detector design—mirroring the real-world findings of Sharma et al. (BMD-45 paper) where small objects are structurally harder for detectors to resolve.

In contrast, **Other** has a default scale multiplier of `1.0` (identical to Hatchback). However, in a 60-image sample, each class contains only 5 to 30 ground-truth annotations. For `Other`, total boxes $N = 17$. Missing just two boxes out of 17 yields $15/17 = 88.2\%$, landing slightly below 90%. If just one additional box had been detected, recall would be $16/17 = 94.1\%$. Therefore, the flagging of `Other` is **sampling noise** arising from small per-class sample size rather than a structural disparity.


## If the grouping variable were a protected attribute

If the grouping variable were a protected demographic attribute (such as race, gender, age, disability, or national origin under Title VII, the Equal Credit Opportunity Act, or the EU AI Act) rather than vehicle class:

1. **Stakes & Harm Shift from Operational Inaccuracy to Civil Rights Violations & Allocative/Representational Harm:**
   - In vehicle detection, a disparity means missed traffic telemetry or elevated collision risks for micro-mobility users, which can be mitigated with auxiliary sensors (e.g., radar/LiDAR).
   - For protected demographic groups, disparities create direct **allocative harm** (denying mortgages, healthcare interventions, employment hiring, or pre-trial bail) and **representational harm** (systematically stereotyping, misgendering, or misidentifying individuals, leading to unlawful arrests via biased facial recognition). These harms directly infringe upon legal civil rights and bodily liberty, with irreversible human impact.

2. **Legal and Regulatory Obligations:**
   - A measurable disparity across protected groups triggers the legal doctrine of **Disparate Impact** (unintentional discrimination under the EEOC 80% / four-fifths rule).
   - Under regulatory frameworks like the EU AI Act (High-Risk AI Systems) and NIST AI Risk Management Framework, continuous auditing, disparate impact assessments, and bias mitigation are strict legal requirements. Deploying a model with unmitigated demographic disparities creates severe liability, regulatory injunctions, and substantial civil penalties.

3. **Engineering Obligations & Standards:**
   - As an engineer, you cannot simply treat a disparity as an informative statistical finding. You have an affirmative duty to:
     - **Halt deployment** until the disparate impact is root-caused and mitigated.
     - Apply formal fairness metrics: **Equal Opportunity** ($P(\hat{Y}=1 | Y=1, A=a) = P(\hat{Y}=1 | Y=1, A=b)$), **Equalized Odds**, and **Demographic Parity**, rather than heuristic thresholds.
     - Implement pre-processing (rebalancing, adversarial debiasing), in-processing (fairness-constrained empirical risk minimization), or post-processing (group-specific threshold adjustment).
     - Maintain immutable Model Cards, Datasheets for Datasets, and continuous fairness drift monitoring pipelines that trigger automatic alerts and rollback mechanisms upon parity violation.


## Which attack could corrupt this audit?

### Lifecycle Stage & Attack:
- **Lifecycle Stage:** Data Curation / Evaluation Fixture Storage (Pre-Audit Ingestion).
- **Attack:** **Data Poisoning (Clean-Label Targeted Evaluation Poisoning).**
  - An adversary (e.g., an insider seeking to pass compliance checks, or an external attacker compromising the S3/GCS fixture bucket) subtly manipulates the evaluation fixtures or ground-truth annotations (`data/fixtures/` and `_annotations.coco.json`).
  - By adding slight reddish tinting to missed small-vehicle boxes (raising pixel redness above $r > 120, r-g > 40$), or by subtly adjusting the ground-truth bounding box coordinates $[x, y, w, h]$ to artificially overlap random background noise detections with $\text{IoU} \ge 0.3$, the attacker silently inflates the measured recall of `Two-wheeler` and `Bicycle` above 90%.
  - This **clean-label audit corruption** leaves the model unchanged but deceives the automated fairness audit into certifying an unsafe, discriminatory model for production release.

### Control from the Lecture:
- **Cryptographic Provenance, Content-Addressable Storage & Signed Metadata:**
  - Fast-hash verification (SHA-256 digests) and Git LFS / DVC cryptographic locks on all evaluation datasets, fixture images, and annotations.
  - Every batch of audit data must be cryptographically signed by authorized curators and verified against an immutable digest in a tamper-evident audit ledger before running the audit script.
  - Pair this with automated **input distribution sanity checks** (e.g., running automated anomaly detection and histogram verification on evaluation images to detect artificial pixel-level color tampering).


## Would SHAP or LIME explain a missed detection?

### Concrete Missed Detection Example:
- **Image:** `camera_A_daylight/000.jpg`
- **Annotation ID:** `3`
- **Category:** `Bicycle`
- **Ground-Truth Box:** $[x=261, y=126, w=10, h=7]$ (Area = 70 pixels)
- **Detector Output:** No detection emitted (0 detections in that region; closest detection has $\text{IoU} = 0.0$).

### Can SHAP or LIME answer WHY the detector missed this box?
**No. Standard SHAP (KernelSHAP / DeepSHAP) and LIME cannot answer this question for a bounding-box object detector.**

### What breaks the foundational assumptions of SHAP and LIME:
1. **The "Absence of Output" Problem (No Scalar Target to Attribute):**
   - Both SHAP and LIME are designed for functions $f: \mathbb{R}^d \to \mathbb{R}^C$ where a model produces a fixed-dimensional prediction vector for an existing instance (e.g., probability of class $C$). Attribution is computed as the gradient or perturbed marginal contribution towards *that prediction*.
   - When a detector misses an object (a False Negative), there is **no output bounding box, no category ID, and no confidence score** generated at that location. The model's output at that spatial coordinate is $\emptyset$ (the empty set).
   - Because there is no scalar output value $y$, SHAP and LIME have no mathematical objective $f(x)$ to compute Shapley values or train a local linear surrogate model $\arg\min_g \mathcal{L}(f, g) + \Omega(g)$ against.

2. **Structured, Variable-Length Output Space vs. Fixed Feature Vectors:**
   - Object detectors map an image $\mathbb{R}^{H \times W \times 3}$ to a dynamic, variable-length set of tuples $\mathcal{D} = \{(bbox_k, class_k, score_k)\}_{k=1}^M$. The dimensionality $M$ varies freely between images.
   - SHAP/LIME assume fixed output semantics. If you perturb superpixels (as LIME does), connected components fragment, candidate boxes shift, merge, or disappear, and Non-Maximum Suppression (NMS) creates discontinuous step-function changes in which detections survive.

3. **What is needed instead:**
   - To explain a missed detection, engineers must inspect **pre-threshold internal states** and utilize **counterfactual generation**:
     - In our mock detector, inspecting the intermediate connected components reveals the ground truth: the crop contains a $10 \times 7$ box with a 1-pixel black border, leaving only $8 \times 5 = 40$ potential vehicle pixels. Due to background noise, fewer than 20 contiguous pixels satisfied $r > 120 \land r - g > 40$, failing the flood-fill size check `len(pts) < 20`.
     - In real deep learning detectors (YOLO / RT-DETR), explaining a false negative requires probing the dense feature pyramid network (FPN) heatmaps, inspecting the classification and regression logits at candidate anchor/query points before NMS, or computing counterfactuals (e.g., determining the minimum perturbation needed to cause an anchor to fire above threshold).


## What did your attack show?

### Attack Progression Across Epsilon Budgets:
From `fairness_report.json`, recall collapses systematically as perturbation budget $\epsilon \in \{0, 10, 20, 30, 40\}$ increases:

| Category | $\epsilon=0$ (Clean) | $\epsilon=10$ | $\epsilon=20$ | $\epsilon=30$ | $\epsilon=40$ | Absolute Drop ($\Delta_{\epsilon=20}$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Two-wheeler** | 0.8462 | 0.7692 | 0.7692 | 0.4615 | 0.1538 | 0.0769 |
| **Other** | 0.8824 | 0.7647 | 0.5882 | 0.5294 | 0.4118 | 0.2942 |
| **Bicycle** | 0.8889 | 0.7778 | 0.5000 | 0.3333 | 0.1667 | **0.3889** |
| **Bus** | 0.9231 | 0.8462 | 0.6538 | 0.5385 | 0.4615 | 0.2693 |
| **Truck** | 0.9259 | 0.8889 | 0.7778 | 0.5926 | 0.4815 | 0.1481 |
| **Van** | 0.9412 | 0.7647 | 0.7059 | 0.4706 | 0.2941 | 0.2353 |
| **Three-wheeler** | 0.9412 | 0.8824 | 0.7647 | 0.7059 | 0.5882 | 0.1765 |
| **LCV** | 0.9412 | 0.8824 | 0.7059 | 0.5882 | 0.4118 | 0.2353 |
| **Tempo-traveller**| 0.9167 | 0.8333 | 0.8333 | 0.8333 | 0.6667 | 0.0833 |
| **Hatchback** | 1.0000 | 1.0000 | 0.9565 | 0.7826 | 0.6087 | 0.0435 |
| **Sedan** | 1.0000 | 0.8571 | 0.8571 | 0.7857 | 0.7143 | 0.1429 |
| **SUV** | 1.0000 | 1.0000 | 0.9091 | 0.6818 | 0.4091 | 0.0909 |
| **MUV** | 1.0000 | 1.0000 | 0.8000 | 0.6000 | 0.5333 | 0.2000 |
| **Mini-bus** | 1.0000 | 0.7895 | 0.7368 | 0.6842 | 0.5789 | 0.2632 |

- At $\epsilon = 10$, initial degradation occurs across mid-sized classes as marginal pixels fail the threshold.
- At $\epsilon = 20$, severe drop manifests across multiple classes.
- At $\epsilon = 30$ and $\epsilon = 40$, performance collapses completely: **Two-wheeler** plummets to **15.4%** and **Bicycle** plummets to **16.7%**.

### Most Fragile Class vs. Clean Flagged Class:
- At $\epsilon = 20$, the `most_fragile` ranking (largest drop $\Delta = Recall_0 - Recall_{20}$) identifies:
  1. **Bicycle:** Drop = **0.3889** (-38.9%)
  2. **Other:** Drop = **0.2942** (-29.4%)
  3. **Bus:** Drop = **0.2693** (-26.9%)
- **Bicycle** is both flagged in the clean audit (88.9%) AND ranks as the #1 most fragile class under attack.
- However, **Bus** (which passed the clean audit easily at 92.3%) experiences a dramatic collapse to 65.4% at $\epsilon=20$, placing it in the top 3 most fragile classes.

### Comparing Drop Ranking (`most_fragile`) with `lowest_recall_under_attack`:
- `lowest_recall_under_attack` at $\epsilon=20$:
  1. **Bicycle:** **50.0%**
  2. **Other:** **58.8%**
  3. **Bus:** **65.4%**
  (Closely followed by **Van** at 70.6% and **LCV** at 70.6%, with **Two-wheeler** at 76.9%).
- **Do they agree?**
  At $\epsilon=20$, the top 3 categories match (`Bicycle`, `Other`, `Bus`). However, as $\epsilon$ increases to 30 and 40, `Two-wheeler` drops to 15.4% (the lowest absolute recall of all 14 classes), yet because its baseline clean recall was already low (84.6%), its relative drop appears smaller than that of a large class that started at 100%.

### Which is the better measure of who is hurt?
**Lowest absolute recall is by far the superior measure of real-world harm.**
- A metric based on relative drop favors classes that already had high baseline performance. But in safety-critical cyber-physical systems (autonomous driving, intelligent CCTV surveillance), safety is dictated by **absolute survival probability**.
- If a cyclist or motorcyclist has only a 15–16% chance of being detected, they face life-threatening collision risks. The fact that their baseline recall was already suppressed (85%) does not make their 15% under attack any less lethal. Measuring absolute recall prevents masking structural vulnerabilities in historically underperforming groups.

### Who bears the cost of an attack or unmonitored failure?
**Vulnerable road users and marginalized groups bear the primary burden.**
Small-vehicle classes (bicycles, two-wheelers, auto-rickshaws) and pedestrians have the smallest physical and visual footprint. When environmental conditions degrade (rain, low-light, fog) or an adversary manipulates the scene, the safety margin of small classes evaporates first and fastest.

### Adversarial Defense: Input Validation & Adversarial Training:
1. **Input Validation (Pre-Inference Sanitization):**
   - Implement an automated distribution check on incoming CCTV frames: calculate the global and local chromaticity histogram.
   - If the red channel is uniformly suppressed relative to green and blue ($\bar{R} < \bar{G} - \theta$), flag the frame as anomalous or apply automated white-balance / histogram equalization (e.g., Gray-World algorithm) before passing to the detector.
2. **Adversarial Training:**
   - For a learning-based model, augment training batches with adversarial examples generated via FGSM/PGD ($\min_\theta \mathbb{E}_{(x,y)} [\max_{\|\delta\|_\infty \le \epsilon} \mathcal{L}(f_\theta(x+\delta), y)]$).
   - For this rule-based detector, calibrate multi-spectral thresholds or dynamic chromatic adaptation rather than a hardcoded static rule.

### Why does a rule this simple break so much more easily than a CNN?
- **Zero-Dimensional Rigid Decision Boundary:**
  `mock_detector.py` relies on an explicit linear hyperplane in 3D color space: $r > 120 \land r - g > 40 \land r - b > 40$.
  Perturbing each channel by $\epsilon$ in the adversarial direction lowers $(r - g)$ by exactly $2\epsilon$. At $\epsilon=20$, the difference collapses by 40, directly forcing pixels across the threshold.
- **Lack of Semantic Redundancy:**
  A deep CNN extracts hierarchical representations across millions of parameters: edges, corners, geometric contours, texture, and contextual priors (e.g., a bicycle on an asphalt road). Even if pixel colors shift, convolutional filters sensitive to edge gradients, wheel circularity, and spatial context still activate.
- **The CNN Tradeoff:**
  While a CNN does not break under naive uniform color shifts, CNNs are still vulnerable to high-dimensional, gradient-aligned adversarial noise (Goodfellow et al., 2014) because small perturbations across thousands of input dimensions accumulate linearly into large activation shifts at the output layer.

