"""
Self-check for the Week 8 lab. Not the grader — see README.md.

Run with: pytest tests/ -q
"""
import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import drift_monitor as dm  # noqa: E402

CAMERA_A = os.path.join(REPO_ROOT, "data", "fixtures", "camera_A_daylight")
CAMERA_B = os.path.join(REPO_ROOT, "data", "fixtures", "camera_B_lowlight")


def test_extract_confidence_scores_returns_floats_in_range():
    scores = dm.extract_confidence_scores(CAMERA_A)
    assert len(scores) > 0
    assert all(isinstance(s, float) for s in scores)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_extract_confidence_scores_is_deterministic():
    assert dm.extract_confidence_scores(CAMERA_A) == dm.extract_confidence_scores(CAMERA_A)


def test_compute_psi_is_zero_for_identical_distributions():
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] * 10
    psi = dm.compute_psi(scores, scores)
    assert psi < 1e-6


def test_compute_psi_is_positive_for_shifted_distributions():
    reference = [0.1] * 50 + [0.15] * 50
    live = [0.85] * 50 + [0.9] * 50
    psi = dm.compute_psi(reference, live)
    assert psi > 1.0  # a total distribution shift should register as large PSI


def test_compute_psi_small_for_similar_distributions():
    reference = ([0.1] * 30 + [0.5] * 40 + [0.9] * 30)
    live = ([0.1] * 32 + [0.5] * 38 + [0.9] * 30)  # nearly identical, tiny wobble
    psi = dm.compute_psi(reference, live)
    assert psi < dm.PSI_MODERATE_THRESHOLD


def test_classify_drift_bands():
    assert dm.classify_drift(0.05) == "none"
    assert dm.classify_drift(0.09999) == "none"
    assert dm.classify_drift(0.10) == "moderate"
    assert dm.classify_drift(0.20) == "moderate"
    assert dm.classify_drift(0.25) == "significant"
    assert dm.classify_drift(0.50) == "significant"


def test_summarize_scores_shape():
    summary = dm.summarize_scores([0.2, 0.4, 0.6, 0.8])
    assert summary["count"] == 4
    assert abs(summary["mean"] - 0.5) < 1e-6
    assert summary["min"] == 0.2
    assert summary["max"] == 0.8
    assert summary["std"] > 0


def test_summarize_scores_single_value_has_zero_std():
    summary = dm.summarize_scores([0.5])
    assert summary["count"] == 1
    assert summary["std"] == 0.0


def test_full_pipeline_runs_and_flags_camera_drift():
    with tempfile.TemporaryDirectory() as scratch:
        for sub in ("src", "data"):
            shutil.copytree(os.path.join(REPO_ROOT, sub), os.path.join(scratch, sub))

        result = subprocess.run(
            [sys.executable, os.path.join(scratch, "src", "run_pipeline.py")],
            cwd=scratch, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

        report_path = os.path.join(scratch, "drift_report.json")
        assert os.path.exists(report_path)
        import json
        with open(report_path) as f:
            report = json.load(f)

        assert report["drift_level"] in ("none", "moderate", "significant")
        assert report["psi"] >= 0
        assert report["reference_summary"]["count"] > 0
        assert report["live_summary"]["count"] > 0
