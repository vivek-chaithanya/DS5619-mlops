# Lab 10 — Fairness and Robustness Audit

**Track B (BMD-45 vehicle detection) · Week 10 · DS5619 Machine Learning Systems Operations**

*Comprehensive documentation with complete code implementations, logic explanations, mathematical formulations, and audit findings.*

---

## Submission Metadata

| Attribute | Details |
|:---|:---|
| **Student ID** | `142301024` |
| **Deterministic Base Seed** | `2290012310` (salt: `week10`, selected attempt 9 with seed `2290012319`) |
| **Dataset Fixtures** | 60 images across 2 camera profiles (30 daylight, 30 lowlight), 257 annotations |
| **Verification Status** | **13 / 13 Pytest Smoke Tests PASS** + Extra Held-out Edge Cases PASS |
| **Flagged Categories** | **Two-wheeler** (84.6%), **Other** (88.2%), **Bicycle** (88.9%) (< 90% threshold) |
| **Most Fragile Category** | **Bicycle** (-38.89% recall drop at $\epsilon = 20$) |

---

## Executive Summary & Architectural Overview

In production computer vision pipelines, models often display severe performance disparities across different data subpopulations. In traffic monitoring and autonomous navigation, detectors may reliably detect large vehicles (buses, trucks) while failing to resolve small, vulnerable road users (bicycles, two-wheelers). Furthermore, models that appear performant in clean benchmarks can exhibit catastrophic fragility under minor distribution shifts or adversarial perturbations.

This lab implements an end-to-end **Fairness and Robustness Audit Pipeline** for vehicle detection on the BMD-45 benchmark (*Sharma et al., CVPR 2026 Findings*):
1. **Fairness Audit (`src/fairness_audit.py`)**: Computes bounding-box Intersection-over-Union (IoU), performs recall-based ground-truth matching, evaluates per-class recall across all 14 BMD-45 categories, and surfaces underperforming categories strictly below the 90% recall threshold.
2. **Adversarial Robustness Check (`src/adversarial.py`)**: Simulates a white-box Fast Gradient Sign Method (FGSM) style attack against the detector's decision boundary by perturbing pixel channels within an $L_\infty$ budget ($\epsilon \in \{0, 10, 20, 30, 40\}$).
3. **Fragility & Harm Assessment**: Evaluates class degradation curves, contrasts relative recall drop with absolute survival probability, and explores lifecycle governance risks.

```
                      +---------------------------------------+
                      |       COCO Fixtures Dataset           |
                      | 60 images, 257 boxes, 14 categories   |
                      +---------------------------------------+
                                          |
                        +-----------------+-----------------+
                        |                                   |
                        v                                   v
             [ Clean Audit Pipeline ]             [ Adversarial Pipeline ]
             Transform: None                     Transform: perturb_image(img, eps)
                        |                                   |
                        v                                   v
             mock_detector.detect()              mock_detector.detect()
                        |                                   |
                        v                                   v
             match_ground_truth(IoU >= 0.3)      match_ground_truth(IoU >= 0.3)
                        |                                   |
                        v                                   v
             Per-Category Clean Recall           Multi-Budget Recall Matrix
             Flagged: Recall < 90%               Ranking: Most Fragile (Drop)
                        |                                   |
                        +-----------------+-----------------+
                                          |
                                          v
                              +-----------------------+
                              | fairness_report.json  |
                              +-----------------------+
```

---

## Implemented Code & Technical Logic Explanations

### Part 1: Intersection over Union (`compute_iou`)

#### Purpose & Mathematical Formulation
Intersection-over-Union (IoU), or the Jaccard index, quantifies spatial overlap between two axis-aligned bounding boxes $A = [x_a, y_a, w_a, h_a]$ and $B = [x_b, y_b, w_b, h_b]$.

Converting each box to corner coordinates $(x_1, y_1, x_2, y_2)$:
$$\begin{aligned}
x_{a1} &= x_a, & y_{a1} &= y_a, & x_{a2} &= x_a + w_a, & y_{a2} &= y_a + h_a \\
x_{b1} &= x_b, & y_{b1} &= y_b, & x_{b2} &= x_b + w_b, & y_{b2} &= y_b + h_b
\end{aligned}$$

The intersection rectangle dimensions are clamped to $\ge 0$:
$$\begin{aligned}
\text{inter\_w} &= \max(0, \min(x_{a2}, x_{b2}) - \max(x_{a1}, x_{b1})) \\
\text{inter\_h} &= \max(0, \min(y_{a2}, y_{b2}) - \max(y_{a1}, y_{b1})) \\
\text{Area}_{\text{inter}} &= \text{inter\_w} \times \text{inter\_h}
\end{aligned}$$

The union area is:
$$\text{Area}_{\text{union}} = (w_a \times h_a) + (w_b \times h_b) - \text{Area}_{\text{inter}}$$

$$\text{IoU}(A, B) = \begin{cases} 
\frac{\text{Area}_{\text{inter}}}{\text{Area}_{\text{union}}}, & \text{if } \text{Area}_{\text{union}} > 0 \\
0.0, & \text{otherwise}
\end{cases}$$

#### Python Implementation (`src/fairness_audit.py`)
```python
def compute_iou(box_a, box_b):
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection_area = inter_w * inter_h

    union_area = box_a[2] * box_a[3] + box_b[2] * box_b[3] - intersection_area
    return float(intersection_area / union_area) if union_area > 0 else 0.0
```

#### Detailed Logic & Edge Case Handling:
- **Disjoint / Non-overlapping Boxes**: `min(ax2, bx2) - max(ax1, bx1)` is negative; `max(0.0, ...)` clamps width/height to 0.0, yielding IoU = 0.0.
- **Identical Boxes**: Intersection equals individual area; union equals area; IoU returns exactly 1.0.
- **Zero-Area Boxes**: Guard `union_area > 0` returns 0.0, preventing `ZeroDivisionError`.
- **Touching Edges/Corners**: Overlap width or height is 0.0, correctly returning 0.0.

---

### Part 2: Ground Truth Matching (`match_ground_truth`)

#### Purpose
Determines whether each ground-truth object was localized by the detector. This is an **equal-opportunity / recall check**, not one-to-one bipartite matching (like Hungarian matching in mAP). A single broad detection can match multiple ground-truth boxes if it sufficiently overlaps both.

#### Python Implementation (`src/fairness_audit.py`)
```python
def match_ground_truth(gt_boxes, detections, iou_threshold=0.3):
    return [
        any(compute_iou(gt_box, d.bbox) >= iou_threshold for d in detections)
        for gt_box in gt_boxes
    ]
```

#### Detailed Logic & Edge Case Handling:
- **Empty Detections**: When `detections = []`, `any(...)` evaluates to `False` for each box, returning `[False] * len(gt_boxes)`.
- **Empty Ground Truth**: When `gt_boxes = []`, the comprehension yields an empty list `[]`.
- **Threshold Boundary**: Uses `>= iou_threshold` (0.3), properly treating an exact match at 0.3 as detected.

---

### Part 3: Per-Category Recall Accumulation (`audit_by_category`)

#### Purpose
Iterates over the entire COCO dataset, loads images, applies optional transforms (for adversarial evaluation), runs detection, matches ground-truth annotations, and accumulates total vs detected counts per category.

#### Python Implementation (`src/fairness_audit.py`)
```python
def audit_by_category(coco_path, fixtures_dir, iou_threshold=0.3, transform=None):
    with open(coco_path, "r") as f:
        coco = json.load(f)

    cat_id_to_name = {cat["id"]: cat["name"] for cat in coco["categories"]}
    image_anns = defaultdict(list)
    for ann in coco["annotations"]:
        image_anns[ann["image_id"]].append(ann)

    category_totals = defaultdict(int)
    category_detected = defaultdict(int)

    for image_info in coco["images"]:
        img_id = image_info["id"]
        img_path = os.path.join(fixtures_dir, image_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        if transform is not None:
            img = transform(img)

        detections = det.detect(img)
        anns = image_anns.get(img_id, [])
        if not anns:
            continue

        gt_boxes = [ann["bbox"] for ann in anns]
        matched_flags = match_ground_truth(gt_boxes, detections, iou_threshold=iou_threshold)

        for ann, matched in zip(anns, matched_flags):
            cat_name = cat_id_to_name[ann["category_id"]]
            category_totals[cat_name] += 1
            if matched:
                category_detected[cat_name] += 1

    stats = {}
    for cat in coco["categories"]:
        cat_name = cat["name"]
        total = category_totals[cat_name]
        if total > 0:
            detected = category_detected[cat_name]
            recall = round(detected / total, 4)
            stats[cat_name] = {
                "total": total,
                "detected": detected,
                "recall": recall,
            }
    return stats
```

#### Detailed Logic & Edge Case Handling:
- **Pre-indexing Optimization**: An $O(1)$ lookup dictionary `image_anns` maps `image_id -> [annotations]`, avoiding repeated linear scans.
- **Transform Hook**: If `transform` is provided, it is applied directly to the RGB PIL image before detection. Ground-truth bounding boxes are never transformed.
- **Zero-Count Category Skipping**: `if total > 0:` skips categories with no occurrences in the dataset, guarding against division-by-zero.
- **Rounding**: Per-class recall is strictly rounded to 4 decimal places.

---

### Part 4: Underperforming Category Detection (`find_underperforming_categories`)

#### Purpose
Identifies all categories with recall strictly below the audit threshold (default 90%), ordered worst-first (lowest recall first).

#### Python Implementation (`src/fairness_audit.py`)
```python
def find_underperforming_categories(category_stats, recall_threshold=0.90):
    underperforming = [
        (name, data["recall"])
        for name, data in category_stats.items()
        if data["recall"] < recall_threshold
    ]
    underperforming.sort(key=lambda x: (x[1], x[0]))
    return underperforming
```

#### Detailed Logic & Edge Case Handling:
- **Strict Inequality**: Requires `recall < recall_threshold`. A category with recall exactly 0.90 is not flagged.
- **Sorting & Tie-Breaking**: Primary sort key `x[1]` ensures worst performers appear first. Secondary sort key `x[0]` sorts alphabetically on ties for deterministic results.

---

### Part 5: FGSM White-Box Perturbation (`perturb_image`)

#### Purpose & Mathematical Formulation
The detector detects vehicle pixels via a deterministic color rule:
$$\text{is\_reddish}(r, g, b) = (r > 120) \land (r - g > 40) \land (r - b > 40)$$

In a white-box setting (Goodfellow et al., 2014), the adversary perturbs inputs in the direction of the sign of the gradient $\text{sign}(\nabla_x \mathcal{L})$. To cause vehicle pixels to fail this rule, the optimal direction is obvious:
1. Decrease the red channel $r$
2. Increase the green channel $g$
3. Increase the blue channel $b$

Under an $L_\infty$ perturbation budget $\epsilon \ge 0$, every pixel is updated:
$$\begin{aligned}
r' &= \max(0, r - \epsilon) \\
g' &= \min(255, g + \epsilon) \\
b' &= \min(255, b + \epsilon)
\end{aligned}$$

Crucially, the color difference metrics collapse at **twice the rate of $\epsilon$**:
$$(r' - g') = (r - \epsilon) - (g + \epsilon) = (r - g) - 2\epsilon$$
$$(r' - b') = (r - \epsilon) - (b + \epsilon) = (r - b) - 2\epsilon$$

At $\epsilon = 20$, the channel difference drops by 40, completely erasing the margin required by $r - g > 40$!

#### Python Implementation (`src/adversarial.py`)
```python
def perturb_image(img, eps):
    if eps == 0:
        return img.copy()

    r, g, b = img.split()
    r_pert = r.point(lambda v: max(0, v - eps))
    g_pert = g.point(lambda v: min(255, v + eps))
    b_pert = b.point(lambda v: min(255, v + eps))
    return Image.merge("RGB", (r_pert, g_pert, b_pert))
```

#### Detailed Logic & Edge Case Handling:
- **Non-Mutation**: The input `img` is never modified. When $\epsilon = 0$, `img.copy()` is returned.
- **Fast Vectorized Lookup**: Uses PIL's C-level `Image.point()` LUT (Lookup Table) instead of slow nested Python pixel loops.
- **Value Clamping**: Explicitly clamps values to $[0, 255]$ preventing 8-bit unsigned integer underflow/overflow.

---

### Part 6: Audit Under Adversarial Attack (`audit_under_attack`)

#### Purpose
Executes the category audit across multiple perturbation budgets $\epsilon \in (0, 10, 20, 30, 40)$, providing clean baseline stats at $\epsilon = 0$.

#### Python Implementation (`src/adversarial.py`)
```python
def audit_under_attack(coco_path, fixtures_dir, eps_values=(0, 10, 20, 30, 40),
                       iou_threshold=0.3):
    attack_stats = {}
    for eps in eps_values:
        transform = (lambda im, e=eps: perturb_image(im, e)) if eps != 0 else None
        attack_stats[eps] = fa.audit_by_category(
            coco_path, fixtures_dir, iou_threshold=iou_threshold, transform=transform
        )
    return attack_stats
```

#### Detailed Logic & Edge Case Handling:
- **Clean Audit Identity**: For $\epsilon = 0$, `transform = None` guarantees `stats[0] == clean_audit`.
- **Late-Binding Closure Guard**: Default parameter `e=eps` in `lambda im, e=eps: ...` binds `eps` at loop definition time, avoiding Python late-binding bugs.

---

### Part 7: Most Fragile Categories Identification (`find_most_fragile`)

#### Purpose
Computes the recall drop $\Delta = \text{recall}_{\epsilon=0} - \text{recall}_{\epsilon}$ and surfaces the top $k$ classes with the largest drops.

#### Python Implementation (`src/adversarial.py`)
```python
def find_most_fragile(attack_stats, eps, top_k=3):
    clean_stats = attack_stats[0]
    pert_stats = attack_stats[eps]
    drops = []
    for cat_name, clean in clean_stats.items():
        if cat_name in pert_stats:
            drop = round(clean["recall"] - pert_stats[cat_name]["recall"], 4)
            drops.append((cat_name, drop))
    drops.sort(key=lambda x: (-x[1], x[0]))
    return drops[:top_k]
```

#### Detailed Logic & Edge Case Handling:
- **Descending Drop, Ascending Name**: Sort key `(-x[1], x[0])` orders highest drops first, breaking ties alphabetically.
- **Precision**: Drop values are rounded to 4 decimal places.

---

## Experimental Findings & Audit Reports

### 1. Clean Fairness Audit Results (Student ID: 142301024)

| Category Name | Ground Truth Count | Detected Count | Recall | Audit Status |
|:---|:---:|:---:|:---:|:---|
| **Two-wheeler** | 13 | 11 | **84.6%** | **FLAGGED (< 90%)** |
| **Other** | 17 | 15 | **88.2%** | **FLAGGED (< 90%)** |
| **Bicycle** | 18 | 16 | **88.9%** | **FLAGGED (< 90%)** |
| **Tempo-traveller** | 12 | 11 | 91.7% | PASS |
| **Bus** | 26 | 24 | 92.3% | PASS |
| **Truck** | 27 | 25 | 92.6% | PASS |
| **Three-wheeler** | 17 | 16 | 94.1% | PASS |
| **LCV** | 17 | 16 | 94.1% | PASS |
| **Van** | 17 | 16 | 94.1% | PASS |
| **Hatchback** | 23 | 23 | 100.0% | PASS |
| **Sedan** | 14 | 14 | 100.0% | PASS |
| **SUV** | 22 | 22 | 100.0% | PASS |
| **MUV** | 15 | 15 | 100.0% | PASS |
| **Mini-bus** | 19 | 19 | 100.0% | PASS |

### 2. Multi-Budget Adversarial Collapse Matrix ($\epsilon \in \{0, 10, 20, 30, 40\}$)

| Category | Clean ($\epsilon=0$) | $\epsilon=10$ | $\epsilon=20$ | $\epsilon=30$ | $\epsilon=40$ | Recall Drop ($\Delta_{\epsilon=20}$) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Two-wheeler** | 0.8462 | 0.7692 | 0.7692 | 0.4615 | **0.1538** | 0.0769 |
| **Other** | 0.8824 | 0.7647 | 0.5882 | 0.5294 | 0.4118 | 0.2942 |
| **Bicycle** | 0.8889 | 0.7778 | **0.5000** | 0.3333 | **0.1667** | **0.3889** |
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

- **Top 3 Most Fragile at $\epsilon = 20$ (by Drop)**:
  1. `Bicycle`: -38.89% drop
  2. `Other`: -29.42% drop
  3. `Bus`: -26.93% drop
- **Top 3 Lowest Recall at $\epsilon = 20$ (Absolute)**:
  1. `Bicycle`: 50.0%
  2. `Other`: 58.8%
  3. `Bus`: 65.4%

---

## Technical & Conceptual Discussions

### 1. Structural Disparity vs. Sampling Noise
- **Structural Disparity (Two-wheeler & Bicycle)**: `generate_for_student.py` enforces size multipliers of 0.45 and 0.35 on two-wheelers and bicycles. The resulting bounding boxes are tiny ($8 \times 6$ to $13 \times 12$). With a 1-pixel border, inner vehicle pixels barely meet the detector's 20-pixel connected component flood threshold. This matches real-world detector physics (*Sharma et al.*), where small physical scale leads to low spatial resolution and high miss rates.
- **Sampling Noise (Other)**: `Other` has a normal size multiplier of 1.0. With only 17 annotations in the entire dataset, missing just 2 boxes drops recall to $15/17 = 88.2\%$. A single box difference accounts for a 5.9% swing. This is small-sample statistical variance.

### 2. Grouping by Protected Demographic Attributes
If the audit grouped by protected demographic attributes (race, gender, age, disability) under Title VII or the EU AI Act:
- **Harm Shift**: Shifts from operational vehicle miscounts to civil rights violations, allocative harms (denial of credit, jobs, medical triage), and representational harms (biased facial recognition arrests).
- **Legal Mandate**: Triggers strict scrutiny under Disparate Impact doctrine (EEOC 80% four-fifths rule). Engineers are legally obligated to halt deployment, perform root-cause analysis, and apply algorithmic mitigation.

### 3. Lifecycle Security Attacks & Mitigations
- **Threat**: Clean-Label Data Poisoning during fixture curation. An adversary alters bounding-box coordinates or introduces imperceptible tinting to evaluation images, falsely inflating small-vehicle recall above 90% to bypass regulatory audits.
- **Control**: Cryptographic data provenance using SHA-256 content-addressable storage (Git LFS / DVC locks), signed curator audit trails, and automated input distribution anomaly detection.

### 4. Why SHAP and LIME Fail on Object Detectors
- **Absence of Scalar Output**: Both SHAP and LIME require a scalar prediction $f(x) \in \mathbb{R}$ to compute feature attributions $\phi_i$. When a detector misses an object (False Negative), the output at that coordinate is the empty set $\emptyset$. There is no prediction score to explain!
- **Variable-Length Structured Outputs**: Detectors output variable-length sets of bounding boxes. Perturbing superpixels (LIME) alters spatial contiguity, causing boxes to split, merge, or vanish unpredictably through Non-Maximum Suppression (NMS).
- **Remedy**: Counterfactual probe generation and pre-NMS dense feature map inspection.

### 5. Adversarial Fragility: Simple Rule vs. Deep CNN
- **Rigid Rule Fragility**: The mock detector's rule is a 0-dimensional hyperplane in RGB space. Shifting channels by $\epsilon = 20$ directly collapses the margin $(r - g)$ by $2\epsilon = 40$, destroying 100% of detection signals.
- **Deep CNN Resilience & Limits**: CNNs leverage hierarchical feature detectors (edges, textures, shapes, spatial context) and are robust to uniform channel shifts. However, CNNs remain vulnerable to high-dimensional gradient-aligned noise (FGSM/PGD) due to linear accumulation across thousands of input dimensions.

---

## Verification & Self-Check Matrix

| Test Case | Description | Result |
|:---|:---|:---:|
| `test_compute_iou_identical_boxes_is_one` | Validates IoU = 1.0 on identical bounding boxes | **PASS** |
| `test_compute_iou_non_overlapping_is_zero` | Validates IoU = 0.0 on disjoint bounding boxes | **PASS** |
| `test_compute_iou_half_overlap` | Validates IoU = 1/3 on 50% overlapping boxes | **PASS** |
| `test_match_ground_truth_marks_covered_and_uncovered` | Validates boolean coverage tracking for gt boxes | **PASS** |
| `test_match_ground_truth_empty_detections_all_false` | Validates empty detections return all False | **PASS** |
| `test_audit_by_category_shape_and_skips_zero_total` | Validates non-zero totals and recall math | **PASS** |
| `test_find_underperforming_categories_sorted_worst_first` | Validates worst-first ordering below threshold | **PASS** |
| `test_find_underperforming_categories_respects_threshold` | Validates strict inequality (< threshold) | **PASS** |
| `test_full_pipeline_runs_and_flags_small_vehicle_classes` | Validates end-to-end run and small-vehicle flagging | **PASS** |
| `test_perturb_image_shifts_channels_and_clamps` | Validates RGB shifting and [0, 255] clamping | **PASS** |
| `test_perturb_image_does_not_mutate_and_eps_zero_is_identity` | Validates immutability and identity at eps=0 | **PASS** |
| `test_audit_under_attack_eps0_matches_clean_audit...` | Validates clean match and collapse under attack | **PASS** |
| `test_find_most_fragile_ranks_by_recall_drop` | Validates recall drop ranking and tie-breaking | **PASS** |
