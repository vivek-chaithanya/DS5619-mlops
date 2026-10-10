"""
Self-check for the Week 9 lab. Not the grader — see README.md.

Run with: pytest tests/ -q
"""
import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import mini_orchestrator as orch  # noqa: E402


def test_topological_order_respects_linear_dependency():
    tasks = {
        "a": orch.Task(name="a", fn=lambda ctx: None, depends_on=[]),
        "b": orch.Task(name="b", fn=lambda ctx: None, depends_on=["a"]),
        "c": orch.Task(name="c", fn=lambda ctx: None, depends_on=["b"]),
    }
    order = orch.topological_order(tasks)
    assert order.index("a") < order.index("b") < order.index("c")


def test_topological_order_respects_branching():
    tasks = {
        "extract": orch.Task(name="extract", fn=lambda ctx: None, depends_on=[]),
        "score": orch.Task(name="score", fn=lambda ctx: None, depends_on=["extract"]),
        "load": orch.Task(name="load", fn=lambda ctx: None, depends_on=["score"]),
        "notify": orch.Task(name="notify", fn=lambda ctx: None, depends_on=["score"]),
    }
    order = orch.topological_order(tasks)
    assert order.index("extract") < order.index("score")
    assert order.index("score") < order.index("load")
    assert order.index("score") < order.index("notify")
    assert len(order) == 4


def test_topological_order_detects_cycle():
    tasks = {
        "a": orch.Task(name="a", fn=lambda ctx: None, depends_on=["b"]),
        "b": orch.Task(name="b", fn=lambda ctx: None, depends_on=["a"]),
    }
    try:
        orch.topological_order(tasks)
        assert False, "should have raised ValueError on a cycle"
    except ValueError:
        pass


def test_topological_order_detects_missing_dependency():
    tasks = {
        "a": orch.Task(name="a", fn=lambda ctx: None, depends_on=["does_not_exist"]),
    }
    try:
        orch.topological_order(tasks)
        assert False, "should have raised ValueError on a missing dependency"
    except ValueError:
        pass


def test_run_task_with_retry_succeeds_first_try():
    calls = []
    task = orch.Task(name="t", fn=lambda ctx: calls.append(1), max_retries=2, retry_delay_seconds=0)
    attempts = orch.run_task_with_retry(task, {})
    assert attempts == 1
    assert len(calls) == 1


def test_run_task_with_retry_succeeds_after_failures():
    state = {"n": 0}

    def flaky(ctx):
        state["n"] += 1
        if state["n"] < 3:
            raise RuntimeError("transient")

    task = orch.Task(name="t", fn=flaky, max_retries=3, retry_delay_seconds=0)
    attempts = orch.run_task_with_retry(task, {})
    assert attempts == 3


def test_run_task_with_retry_raises_after_exhausting_retries():
    def always_fails(ctx):
        raise RuntimeError("permanent failure")

    task = orch.Task(name="t", fn=always_fails, max_retries=2, retry_delay_seconds=0)
    try:
        orch.run_task_with_retry(task, {})
        assert False, "should have raised after exhausting retries"
    except RuntimeError as e:
        assert "permanent failure" in str(e)


def test_run_dag_happy_path_shares_context():
    def step1(ctx):
        ctx["x"] = 1

    def step2(ctx):
        ctx["y"] = ctx["x"] + 1

    tasks = {
        "step1": orch.Task(name="step1", fn=step1, depends_on=[]),
        "step2": orch.Task(name="step2", fn=step2, depends_on=["step1"]),
    }
    context = {}
    result = orch.run_dag(tasks, context)
    assert context["y"] == 2
    assert result["task_results"]["step1"]["status"] == "success"
    assert result["task_results"]["step2"]["status"] == "success"


def test_run_dag_stops_after_a_failed_task():
    ran = []

    def ok(ctx):
        ran.append("ok")

    def fails(ctx):
        ran.append("fails")
        raise RuntimeError("boom")

    def never_runs(ctx):
        ran.append("never_runs")

    tasks = {
        "ok": orch.Task(name="ok", fn=ok, depends_on=[]),
        "fails": orch.Task(name="fails", fn=fails, depends_on=["ok"], max_retries=1, retry_delay_seconds=0),
        "downstream": orch.Task(name="downstream", fn=never_runs, depends_on=["fails"]),
    }
    result = orch.run_dag(tasks, {})
    assert result["task_results"]["ok"]["status"] == "success"
    assert result["task_results"]["fails"]["status"] == "failed"
    assert "downstream" not in result["task_results"]
    assert "never_runs" not in ran


def test_full_pipeline_runs_and_retries_notify():
    with tempfile.TemporaryDirectory() as scratch:
        for sub in ("src", "data"):
            shutil.copytree(os.path.join(REPO_ROOT, sub), os.path.join(scratch, sub))

        result = subprocess.run(
            [sys.executable, os.path.join(scratch, "src", "run_pipeline.py")],
            cwd=scratch, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

        import json
        with open(os.path.join(scratch, "batch_report.json")) as f:
            report = json.load(f)
        assert report["images_scored"] > 0

        with open(os.path.join(scratch, "dag_run_summary.json")) as f:
            summary = json.load(f)
        assert summary["task_results"]["notify"]["status"] == "success"
        assert summary["task_results"]["notify"]["attempts"] == 3
