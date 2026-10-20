"""
Self-check for the Week 10 lab. Not the grader — see README.md.

Run with: pytest tests/ -q
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import fairness_audit as fa  # noqa: E402
import adversarial as adv  # noqa: E402
from PIL import Image  # noqa: E402
import mock_detector as det  # noqa: E402

COCO_PATH = os.path.join(REPO_ROOT, "data", "fixtures", "_annotations.coco.json")
FIXTURES_DIR = os.path.join(REPO_ROOT, "data", "fixtures")


def test_compute_iou_identical_boxes_is_one():
    box = [10, 10, 20, 20]
    assert abs(fa.compute_iou(box, box) - 1.0) < 1e-6


def test_compute_iou_non_overlapping_is_zero():
    assert fa.compute_iou([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0


def test_compute_iou_half_overlap():
    # box_a: x in [0,10], box_b: x in [5,15] -> overlap width 5, area 5*10=50
    # union = 100 + 100 - 50 = 150 -> iou = 50/150 = 1/3
    iou = fa.compute_iou([0, 0, 10, 10], [5, 0, 10, 10])
    assert abs(iou - (1 / 3)) < 1e-6


def test_match_ground_truth_marks_covered_and_uncovered():
    gt_boxes = [[0, 0, 10, 10], [100, 100, 10, 10]]
    detections = [det.Detection(bbox=[1, 1, 10, 10], category_id=0, score=0.9)]
    matched = fa.match_ground_truth(gt_boxes, detections, iou_threshold=0.3)
    assert matched == [True, False]


def test_match_ground_truth_empty_detections_all_false():
    gt_boxes = [[0, 0, 10, 10], [20, 20, 10, 10]]
    assert fa.match_ground_truth(gt_boxes, [], iou_threshold=0.3) == [False, False]


def test_audit_by_category_shape_and_skips_zero_total():
    stats = fa.audit_by_category(COCO_PATH, FIXTURES_DIR)
    assert len(stats) > 0
    for name, s in stats.items():
        assert s["total"] > 0
        assert 0 <= s["detected"] <= s["total"]
        assert abs(s["recall"] - s["detected"] / s["total"]) < 1e-3  # allows for round(..., 4)


def test_find_underperforming_categories_sorted_worst_first():
    stats = {
        "A": {"total": 10, "detected": 9, "recall": 0.9},
        "B": {"total": 10, "detected": 5, "recall": 0.5},
        "C": {"total": 10, "detected": 10, "recall": 1.0},
        "D": {"total": 10, "detected": 7, "recall": 0.7},
    }
    result = fa.find_underperforming_categories(stats, recall_threshold=0.95)
    names = [name for name, _ in result]
    assert names == ["B", "D", "A"]  # worst (lowest recall) first
    assert "C" not in names


def test_find_underperforming_categories_respects_threshold():
    stats = {
        "A": {"total": 10, "detected": 10, "recall": 1.0},
        "B": {"total": 10, "detected": 9, "recall": 0.9},
    }
    assert fa.find_underperforming_categories(stats, recall_threshold=0.85) == []
    result = fa.find_underperforming_categories(stats, recall_threshold=0.95)
    assert [name for name, _ in result] == ["B"]


def test_full_pipeline_runs_and_flags_small_vehicle_classes():
    with tempfile.TemporaryDirectory() as scratch:
        for sub in ("src", "data"):
            shutil.copytree(os.path.join(REPO_ROOT, sub), os.path.join(scratch, sub))

        result = subprocess.run(
            [sys.executable, os.path.join(scratch, "src", "run_pipeline.py")],
            cwd=scratch, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

        report_path = os.path.join(scratch, "fairness_report.json")
        assert os.path.exists(report_path)
        with open(report_path) as f:
            report = json.load(f)

        assert len(report["category_stats"]) == 14
        # This dataset is deliberately built so small-vehicle classes are
        # harder to detect — the audit should actually find something.
        assert len(report["underperforming_categories"]) > 0

        adv_part = report["adversarial"]
        assert adv_part["eps_values"][0] == 0
        assert len(adv_part["most_fragile"]) == 3


# --- Part 5-7: adversarial robustness -------------------------------------

def test_perturb_image_shifts_channels_and_clamps():
    img = Image.new("RGB", (2, 1))
    img.putpixel((0, 0), (200, 50, 60))
    img.putpixel((1, 0), (5, 250, 255))
    out = adv.perturb_image(img, 10)
    assert out.getpixel((0, 0)) == (190, 60, 70)
    assert out.getpixel((1, 0)) == (0, 255, 255)  # clamped at both ends


def test_perturb_image_does_not_mutate_and_eps_zero_is_identity():
    img = Image.new("RGB", (3, 3), (200, 50, 60))
    out = adv.perturb_image(img, 0)
    assert list(out.getdata()) == list(img.getdata())
    adv.perturb_image(img, 25)
    assert img.getpixel((0, 0)) == (200, 50, 60)


def test_audit_under_attack_eps0_matches_clean_audit_and_recall_collapses():
    clean = fa.audit_by_category(COCO_PATH, FIXTURES_DIR)
    stats = adv.audit_under_attack(COCO_PATH, FIXTURES_DIR, (0, 20, 60))
    assert set(stats) == {0, 20, 60}
    assert stats[0] == clean
    mean = lambda s: sum(v["recall"] for v in s.values()) / len(s)
    assert mean(stats[20]) < mean(stats[0])
    assert mean(stats[60]) < 0.5 * mean(stats[0])


def test_find_most_fragile_ranks_by_recall_drop():
    stats = {
        0: {"A": {"recall": 1.0}, "B": {"recall": 0.9}, "C": {"recall": 0.8}, "D": {"recall": 1.0}},
        20: {"A": {"recall": 0.9}, "B": {"recall": 0.4}, "C": {"recall": 0.7}, "D": {"recall": 0.9}},
    }
    # drops: A 0.1, B 0.5, C 0.1, D 0.1 -> B first, then ties alphabetical
    assert adv.find_most_fragile(stats, 20, top_k=3) == [("B", 0.5), ("A", 0.1), ("C", 0.1)]
