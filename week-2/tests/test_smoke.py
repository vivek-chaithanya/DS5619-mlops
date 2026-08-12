"""
Self-check smoke test. Run with: pytest -q

This does NOT grade your submission — it only checks your pipeline runs
end-to-end and produces sane output. Passing this is necessary, not
sufficient, for full credit (see ASSIGNMENT.md's grading checklist).
"""
import json
import os
import subprocess
import sys
import tempfile

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_pipeline(config_path):
    result = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "src", "pipeline.py"), "--config", config_path],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return result


def test_csv_config_runs_and_produces_report():
    with tempfile.TemporaryDirectory() as tmp:
        output_path = os.path.join(tmp, "report_csv.json")
        config = {
            "input_path": os.path.join(REPO_ROOT, "data", "v1", "transactions.csv"),
            "input_format": "csv",
            "high_value_threshold": 4000,
            "output_path": output_path,
        }
        config_path = os.path.join(tmp, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = _run_pipeline(config_path)
        assert result.returncode == 0, f"pipeline.py exited nonzero:\n{result.stderr}"
        assert os.path.exists(output_path), "pipeline did not write the configured output_path"

        report = json.load(open(output_path))
        for key in ("n_transactions", "total_amount", "fraud_rate", "n_high_value", "high_value_threshold"):
            assert key in report, f"report missing expected key: {key}"
        assert report["n_transactions"] > 0
        assert report["high_value_threshold"] == 4000


def test_json_config_runs_without_editing_pipeline_py():
    """Same data, different format and threshold, purely via config."""
    with tempfile.TemporaryDirectory() as tmp:
        output_path = os.path.join(tmp, "report_json.json")
        config = {
            "input_path": os.path.join(REPO_ROOT, "data", "v1", "transactions.json"),
            "input_format": "json",
            "high_value_threshold": 9000,
            "output_path": output_path,
        }
        config_path = os.path.join(tmp, "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        result = _run_pipeline(config_path)
        assert result.returncode == 0, f"pipeline.py exited nonzero:\n{result.stderr}"

        report = json.load(open(output_path))
        assert report["high_value_threshold"] == 9000


def test_missing_key_raises_clear_error():
    with tempfile.TemporaryDirectory() as tmp:
        config_path = os.path.join(tmp, "bad_config.yaml")
        with open(config_path, "w") as f:
            yaml.dump({"input_path": "data/v1/transactions.csv"}, f)  # missing everything else

        result = _run_pipeline(config_path)
        assert result.returncode != 0, "pipeline.py should fail on an incomplete config"
        assert "high_value_threshold" in (result.stderr + result.stdout) or \
               "input_format" in (result.stderr + result.stdout) or \
               "output_path" in (result.stderr + result.stdout), \
            "error message should name which config key is missing"
