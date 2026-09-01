"""
Self-check for the Week 5 lab. Not the grader — see ASSIGNMENT.md.

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
import mini_model_registry as reg  # noqa: E402

FILLED_CARD = {
    "intended_use": "Flags candidate fraudulent card transactions for manual review.",
    "training_data": "Trained on the card_activity feature group (Week 4), v1 raw source.",
    "limitations": "A pure amount-threshold rule misses low-amount fraud and structured/split transactions.",
    "ethical_considerations": "False positives inconvenience legitimate customers; false negatives cost the business directly.",
}


def _write_model_file(path, threshold):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"type": "amount_threshold_classifier", "threshold": threshold}, f)


def test_register_model_creates_manifest(tmp_path):
    model_path = tmp_path / "model.json"
    _write_model_file(str(model_path), 300.0)
    registry = str(tmp_path / "registry")

    vid = reg.register_model("m", str(model_path), {"f1": 0.5}, registry)
    manifest_path = os.path.join(registry, "models", "m", vid, "manifest.json")
    assert os.path.exists(manifest_path)
    with open(manifest_path) as f:
        m = json.load(f)
    assert m["stage"] == "None"
    assert m["metrics"] == {"f1": 0.5}

    assert os.path.exists(os.path.join(registry, "models", "m", vid, "model.json"))


def test_generate_model_card_rejects_incomplete(tmp_path):
    model_path = tmp_path / "model.json"
    _write_model_file(str(model_path), 300.0)
    registry = str(tmp_path / "registry")
    vid = reg.register_model("m", str(model_path), {"f1": 0.9}, registry)

    incomplete = dict(FILLED_CARD)
    incomplete["limitations"] = "TODO: fill this in"
    try:
        reg.generate_model_card("m", vid, incomplete, registry)
        assert False, "should have raised ValueError for a TODO field"
    except ValueError:
        pass

    missing = dict(FILLED_CARD)
    del missing["ethical_considerations"]
    try:
        reg.generate_model_card("m", vid, missing, registry)
        assert False, "should have raised ValueError for a missing field"
    except ValueError:
        pass


def test_generate_model_card_writes_file(tmp_path):
    model_path = tmp_path / "model.json"
    _write_model_file(str(model_path), 300.0)
    registry = str(tmp_path / "registry")
    vid = reg.register_model("m", str(model_path), {"f1": 0.9}, registry)

    card_path = reg.generate_model_card("m", vid, FILLED_CARD, registry)
    assert os.path.exists(card_path)
    with open(card_path) as f:
        card = json.load(f)
    assert card["intended_use"] == FILLED_CARD["intended_use"]
    assert card["metrics"] == {"f1": 0.9}


def test_promote_blocked_without_card(tmp_path):
    model_path = tmp_path / "model.json"
    _write_model_file(str(model_path), 300.0)
    registry = str(tmp_path / "registry")
    vid = reg.register_model("m", str(model_path), {"f1": 0.9}, registry)

    try:
        reg.promote_model("m", vid, "Production", registry)
        assert False, "should have raised GovernanceError with no card"
    except reg.GovernanceError:
        pass


def test_promote_blocked_below_f1_threshold(tmp_path):
    model_path = tmp_path / "model.json"
    _write_model_file(str(model_path), 300.0)
    registry = str(tmp_path / "registry")
    vid = reg.register_model("m", str(model_path), {"f1": 0.3}, registry)
    reg.generate_model_card("m", vid, FILLED_CARD, registry)

    try:
        reg.promote_model("m", vid, "Production", registry)
        assert False, "should have raised GovernanceError below f1 threshold"
    except reg.GovernanceError:
        pass


def test_promote_succeeds_and_archives_previous_production(tmp_path):
    registry = str(tmp_path / "registry")

    model_1 = tmp_path / "model1.json"
    _write_model_file(str(model_1), 300.0)
    v1 = reg.register_model("m", str(model_1), {"f1": 0.9}, registry)
    reg.generate_model_card("m", v1, FILLED_CARD, registry)
    m1 = reg.promote_model("m", v1, "Production", registry)
    assert m1["stage"] == "Production"
    assert len(m1["history"]) == 1
    assert m1["history"][0]["to_stage"] == "Production"

    model_2 = tmp_path / "model2.json"
    _write_model_file(str(model_2), 400.0)
    v2 = reg.register_model("m", str(model_2), {"f1": 0.95}, registry)
    reg.generate_model_card("m", v2, FILLED_CARD, registry)
    m2 = reg.promote_model("m", v2, "Production", registry)
    assert m2["stage"] == "Production"

    with open(os.path.join(registry, "models", "m", v1, "manifest.json")) as f:
        m1_after = json.load(f)
    assert m1_after["stage"] == "Archived"

    prod = reg.get_production_model("m", registry)
    assert prod["version_id"] == v2


def test_get_production_model_returns_none_when_empty(tmp_path):
    registry = str(tmp_path / "registry")
    model_path = tmp_path / "model.json"
    _write_model_file(str(model_path), 300.0)
    reg.register_model("m", str(model_path), {"f1": 0.9}, registry)

    assert reg.get_production_model("m", registry) is None


def test_full_pipeline_runs_end_to_end():
    with tempfile.TemporaryDirectory() as scratch:
        for sub in ("src", "data"):
            shutil.copytree(os.path.join(REPO_ROOT, sub), os.path.join(scratch, sub))
        with open(os.path.join(scratch, "model_card_fields.json"), "w") as f:
            json.dump(FILLED_CARD, f)

        result = subprocess.run(
            [sys.executable, os.path.join(scratch, "src", "run_pipeline.py")],
            cwd=scratch, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert "was blocked" in result.stdout  # both expected GovernanceErrors happened

        summary_path = os.path.join(scratch, "registry_summary.json")
        assert os.path.exists(summary_path)
        with open(summary_path) as f:
            summary = json.load(f)
        assert summary["production_version"] is not None
        assert summary["production_metrics"]["f1"] >= reg.PRODUCTION_F1_THRESHOLD
