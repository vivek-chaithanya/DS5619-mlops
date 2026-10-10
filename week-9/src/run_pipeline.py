r"""
Driver script — defines the actual batch-scoring DAG and runs it with
mini_orchestrator.py. Complete, don't edit. Run with:

    python src/run_pipeline.py

DAG shape (this is the "anatomy of a DAG" from this week's lecture, applied):

    extract --> score --> load
                      \-> notify

- extract: list every *.jpg under data/fixtures/.
- score:   run the detector over each image, accumulate detection counts.
- load:    write batch_report.json (depends only on score).
- notify:  a deliberately FLAKY task (fails twice, then succeeds) standing
           in for something like a webhook/Slack notification call that's
           allowed to be unreliable — it also depends only on score, and
           its retries are what should make the whole DAG succeed anyway.

Both `load` and `notify` depend on `score` but not on each other — your
topological_order just needs to place `score` before both of them.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mini_orchestrator as orch
import mock_detector as det
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES_DIR = os.path.join(REPO_ROOT, "data", "fixtures")

_notify_attempt_counter = {"n": 0}


def extract(context):
    paths = sorted(glob.glob(os.path.join(FIXTURES_DIR, "**", "*.jpg"), recursive=True))
    context["image_paths"] = paths


def score(context):
    per_image = {}
    total_detections = 0
    for path in context["image_paths"]:
        image = Image.open(path).convert("RGB")
        detections = det.detect(image)
        rel_path = os.path.relpath(path, FIXTURES_DIR)
        per_image[rel_path] = len(detections)
        total_detections += len(detections)
    context["per_image_counts"] = per_image
    context["total_detections"] = total_detections


def load(context):
    report = {
        "images_scored": len(context["image_paths"]),
        "total_detections": context["total_detections"],
        "per_image_counts": context["per_image_counts"],
    }
    with open(os.path.join(REPO_ROOT, "batch_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    context["report_written"] = True


def notify(context):
    # Simulates a flaky downstream call (e.g. a webhook) that fails on its
    # first two attempts and succeeds on the third — this is here so the
    # DAG run actually exercises retry logic, not just the happy path.
    _notify_attempt_counter["n"] += 1
    if _notify_attempt_counter["n"] < 3:
        raise RuntimeError(f"simulated transient failure (attempt {_notify_attempt_counter['n']})")
    context["notified"] = True


def main():
    tasks = {
        "extract": orch.Task(name="extract", fn=extract, depends_on=[]),
        "score": orch.Task(name="score", fn=score, depends_on=["extract"]),
        "load": orch.Task(name="load", fn=load, depends_on=["score"]),
        "notify": orch.Task(
            name="notify", fn=notify, depends_on=["score"], max_retries=3, retry_delay_seconds=0.1
        ),
    }

    context = {}
    result = orch.run_dag(tasks, context)

    print(f"Execution order: {result['order']}")
    for name, info in result["task_results"].items():
        print(f"  {name}: {info}")

    run_summary_path = os.path.join(REPO_ROOT, "dag_run_summary.json")
    with open(run_summary_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {run_summary_path}")

    any_failed = any(r["status"] == "failed" for r in result["task_results"].values())
    if any_failed:
        raise SystemExit("DAG run had a failed task — see dag_run_summary.json")


if __name__ == "__main__":
    main()
