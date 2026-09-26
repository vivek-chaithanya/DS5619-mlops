"""
Self-check for THIS week's actual deliverables: the GitHub Actions workflow
and the integration test script. Not the grader, and not a substitute for
actually pushing to GitHub and watching the workflow run (see
CI_VERIFICATION.md) — this only checks structure and syntax offline, since
this environment may not have Docker or a GitHub Actions runner available.

Run with: pytest tests/ -q
"""
import os
import subprocess

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKFLOW_PATH = os.path.join(REPO_ROOT, ".github", "workflows", "ci.yml")
SCRIPT_PATH = os.path.join(REPO_ROOT, "scripts", "integration_test.sh")


def _raw(path):
    with open(path) as f:
        return f.read()


def test_workflow_file_exists_and_has_no_placeholders():
    assert os.path.exists(WORKFLOW_PATH)
    raw = _raw(WORKFLOW_PATH)
    assert "???" not in raw, "ci.yml still has unfilled ??? placeholders"


def test_workflow_defines_three_jobs_with_correct_dependency():
    with open(WORKFLOW_PATH) as f:
        workflow = yaml.safe_load(f)

    jobs = workflow.get("jobs", {})
    for name in ("lint", "unit-test", "integration-test"):
        assert name in jobs, f"missing job '{name}' in ci.yml"

    needs = jobs["integration-test"].get("needs")
    assert needs is not None, "integration-test job must declare `needs:`"
    needs_set = {needs} if isinstance(needs, str) else set(needs)
    assert needs_set == {"lint", "unit-test"}, (
        f"integration-test should need exactly [lint, unit-test], got {needs_set}"
    )


def test_workflow_triggers_on_push_and_pull_request_to_main():
    with open(WORKFLOW_PATH) as f:
        workflow = yaml.safe_load(f)
    # YAML parses the bare key `on:` as boolean True in some parsers/versions;
    # handle both spellings defensively.
    triggers = workflow.get("on", workflow.get(True))
    assert "push" in triggers and "pull_request" in triggers
    assert "main" in triggers["push"].get("branches", [])


def test_lint_job_runs_flake8():
    with open(WORKFLOW_PATH) as f:
        workflow = yaml.safe_load(f)
    steps = workflow["jobs"]["lint"]["steps"]
    run_commands = " ".join(s.get("run", "") for s in steps)
    assert "flake8" in run_commands


def test_unit_test_job_runs_pytest():
    with open(WORKFLOW_PATH) as f:
        workflow = yaml.safe_load(f)
    steps = workflow["jobs"]["unit-test"]["steps"]
    run_commands = " ".join(s.get("run", "") for s in steps)
    assert "pytest" in run_commands


def test_integration_test_job_runs_the_script():
    with open(WORKFLOW_PATH) as f:
        workflow = yaml.safe_load(f)
    steps = workflow["jobs"]["integration-test"]["steps"]
    run_commands = " ".join(s.get("run", "") for s in steps)
    assert "integration_test.sh" in run_commands


def test_integration_script_exists_and_has_no_placeholders():
    assert os.path.exists(SCRIPT_PATH)
    raw = _raw(SCRIPT_PATH)
    assert "???" not in raw, "integration_test.sh still has unfilled ??? placeholders"


def test_integration_script_is_valid_bash_syntax():
    result = subprocess.run(["bash", "-n", SCRIPT_PATH], capture_output=True, text=True)
    assert result.returncode == 0, f"bash -n failed:\n{result.stderr}"


def test_integration_script_references_docker_and_curl():
    raw = _raw(SCRIPT_PATH)
    assert "docker build" in raw
    assert "docker run" in raw
    assert "curl" in raw
