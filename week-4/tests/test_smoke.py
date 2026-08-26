"""
Self-check for the Week 4 lab. Not the grader — see ASSIGNMENT.md.

Run with: pytest tests/ -q
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import mini_feature_store as mfs  # noqa: E402


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_snapshot_raw_version_creates_manifest(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(str(csv_path), [{"a": "1", "b": "2"}], ["a", "b"])
    registry = str(tmp_path / "registry")

    vid = mfs.snapshot_raw_version(str(csv_path), registry)
    manifest_path = os.path.join(registry, "raw_versions", vid, "manifest.json")
    assert os.path.exists(manifest_path)
    with open(manifest_path) as f:
        manifest = json.load(f)
    assert manifest["row_count"] == 1
    assert set(manifest["columns"]) == {"a", "b"}


def test_snapshot_raw_version_is_idempotent(tmp_path):
    csv_path = tmp_path / "data.csv"
    _write_csv(str(csv_path), [{"a": "1", "b": "2"}], ["a", "b"])
    registry = str(tmp_path / "registry")

    vid1 = mfs.snapshot_raw_version(str(csv_path), registry)
    vid2 = mfs.snapshot_raw_version(str(csv_path), registry)
    assert vid1 == vid2

    existing = os.listdir(os.path.join(registry, "raw_versions"))
    assert len(existing) == 1


def test_snapshot_raw_version_detects_change(tmp_path):
    csv_path = tmp_path / "data.csv"
    registry = str(tmp_path / "registry")

    _write_csv(str(csv_path), [{"a": "1", "b": "2"}], ["a", "b"])
    vid1 = mfs.snapshot_raw_version(str(csv_path), registry)

    _write_csv(str(csv_path), [{"a": "1", "b": "3"}], ["a", "b"])
    vid2 = mfs.snapshot_raw_version(str(csv_path), registry)

    assert vid1 != vid2


def test_build_features_v1_schema():
    rows = [
        {"card_id": "card_A", "amount": "100.0", "country": "US", "card_present": "True", "timestamp": "2026-01-01T00:00:00Z"},
        {"card_id": "card_A", "amount": "50.0", "country": "US", "card_present": "False", "timestamp": "2026-01-02T00:00:00Z"},
        {"card_id": "card_B", "amount": "10.0", "country": "IN", "card_present": "True", "timestamp": "2026-01-03T00:00:00Z"},
    ]
    features = mfs.build_features(rows)
    by_card = {f["card_id"]: f for f in features}

    assert set(by_card.keys()) == {"card_A", "card_B"}
    assert by_card["card_A"]["txn_count"] == 2
    assert by_card["card_A"]["avg_amount"] == 75.0
    assert by_card["card_A"]["max_amount"] == 100.0
    assert abs(by_card["card_A"]["pct_card_present"] - 0.5) < 1e-6
    assert by_card["card_A"]["event_time"] == "2026-01-02T00:00:00Z"


def test_build_features_v2_schema():
    rows = [
        {"card_id": "card_A", "amount_minor_units": "10000", "country_code": "US", "card_present": "True", "timestamp": "2026-02-01T00:00:00Z", "device_fingerprint": "dev_1"},
        {"card_id": "card_A", "amount_minor_units": "5000", "country_code": "US", "card_present": "True", "timestamp": "2026-02-02T00:00:00Z", "device_fingerprint": "dev_1"},
    ]
    features = mfs.build_features(rows)
    assert len(features) == 1
    f = features[0]
    assert f["card_id"] == "card_A"
    assert f["txn_count"] == 2
    assert f["avg_amount"] == 75.0  # (100.0 + 50.0) / 2, converted from minor units


def test_register_feature_group_does_not_overwrite(tmp_path):
    registry = str(tmp_path / "registry")
    rows = [{"card_id": "card_A", "txn_count": 1, "avg_amount": 1.0, "max_amount": 1.0,
             "pct_card_present": 1.0, "event_time": "2026-01-01T00:00:00Z"}]

    fg1 = mfs.register_feature_group("fg", rows, "v1", registry, transform_version="v1")
    fg2 = mfs.register_feature_group("fg", rows, "v2", registry, transform_version="v1")

    assert fg1 != fg2
    assert os.path.exists(os.path.join(registry, "feature_groups", "fg", fg1, "manifest.json"))
    assert os.path.exists(os.path.join(registry, "feature_groups", "fg", fg2, "manifest.json"))

    with open(os.path.join(registry, "feature_groups", "fg", fg2, "manifest.json")) as f:
        m2 = json.load(f)
    assert m2["source_raw_version_id"] == "v2"


def test_get_lineage_traces_to_raw_source(tmp_path):
    registry = str(tmp_path / "registry")
    csv_path = tmp_path / "data.csv"
    _write_csv(str(csv_path), [{"a": "1"}], ["a"])

    raw_vid = mfs.snapshot_raw_version(str(csv_path), registry)
    rows = [{"card_id": "card_A", "txn_count": 1, "avg_amount": 1.0, "max_amount": 1.0,
             "pct_card_present": 1.0, "event_time": "2026-01-01T00:00:00Z"}]
    fg_vid = mfs.register_feature_group("fg", rows, raw_vid, registry, transform_version="v1")

    lineage = mfs.get_lineage("fg", fg_vid, registry)
    assert lineage["feature_group"]["feature_group_version_id"] == fg_vid
    assert lineage["raw_source"]["version_id"] == raw_vid
    assert lineage["raw_source"]["source_path"] == str(csv_path)


def test_full_pipeline_runs_and_writes_lineage_report():
    # Run against a scratch copy of the repo so we don't depend on / pollute
    # the student's own .feature_store or lineage_report.json state.
    with tempfile.TemporaryDirectory() as scratch:
        for sub in ("src", "data"):
            shutil.copytree(os.path.join(REPO_ROOT, sub), os.path.join(scratch, sub))

        result = subprocess.run(
            [sys.executable, os.path.join(scratch, "src", "run_pipeline.py")],
            cwd=scratch, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

        report_path = os.path.join(scratch, "lineage_report.json")
        assert os.path.exists(report_path)
        with open(report_path) as f:
            report = json.load(f)

        assert report["v1"]["raw_source"]["version_id"] != report["v2"]["raw_source"]["version_id"]
        assert report["v1"]["feature_group"]["feature_group_version_id"] != \
            report["v2"]["feature_group"]["feature_group_version_id"]

        fg_dir = os.path.join(scratch, ".feature_store", "feature_groups", "card_activity")
        assert len(os.listdir(fg_dir)) == 2  # v1 and v2, neither overwritten
