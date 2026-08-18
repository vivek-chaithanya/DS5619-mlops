"""
Self-check smoke test. Run with: pytest -q (from the repo root).

This does NOT grade your submission — see ASSIGNMENT.md's grading checklist
for what actually counts.
"""
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import expectations as exp  # noqa: E402


def test_expect_column_not_null():
    rows = [{"x": "1"}, {"x": None}, {"x": ""}, {"x": "4"}]
    violations = exp.expect_column_not_null(rows, "x")
    assert {v.row_index for v in violations} == {1, 2}


def test_expect_column_positive():
    rows = [{"x": "5"}, {"x": "-3"}, {"x": "0"}, {"x": "abc"}]
    violations = exp.expect_column_positive(rows, "x")
    assert {v.row_index for v in violations} == {1, 2, 3}


def test_expect_column_in_set():
    rows = [{"x": "a"}, {"x": "b"}, {"x": "z"}]
    violations = exp.expect_column_in_set(rows, "x", {"a", "b"})
    assert {v.row_index for v in violations} == {2}


def test_expect_column_unique():
    rows = [{"x": "a"}, {"x": "b"}, {"x": "a"}, {"x": "a"}]
    violations = exp.expect_column_unique(rows, "x")
    # first "a" (index 0) is not a violation; the repeats at 2 and 3 are
    assert {v.row_index for v in violations} == {2, 3}


def test_full_etl_pipeline_quarantines_known_bad_rows():
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "src", "etl.py"), "--config", os.path.join(REPO_ROOT, "config.yaml")],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, f"etl.py exited nonzero:\n{result.stderr}"

    clean_path = os.path.join(REPO_ROOT, "data", "clean_transactions.csv")
    quarantine_path = os.path.join(REPO_ROOT, "data", "quarantined_transactions.csv")
    report_path = os.path.join(REPO_ROOT, "data", "validation_report.json")

    assert os.path.exists(clean_path)
    assert os.path.exists(quarantine_path)
    assert os.path.exists(report_path)

    report = json.load(open(report_path))
    total_violations = sum(entry["n_violations"] for entry in report["expectations"])
    assert total_violations >= 7, (
        f"expected at least 7 violations across the known-dirty rows, found {total_violations}"
    )

    import csv as _csv
    with open(quarantine_path, newline="") as f:
        n_quarantined = sum(1 for _ in _csv.DictReader(f))
    with open(clean_path, newline="") as f:
        n_clean = sum(1 for _ in _csv.DictReader(f))

    assert n_quarantined >= 6, f"expected at least 6 distinct rows quarantined, found {n_quarantined}"
    assert n_clean == 600 - n_quarantined
