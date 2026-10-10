"""
A tiny local DAG orchestrator — enough to demonstrate the two ideas from
this week's lecture without needing an actual Airflow install (a scheduler,
webserver, and metadata database is a lot of infra for a 110-minute
session):

  1. A DAG is a dependency graph of tasks, executed in an order that
     respects those dependencies (topological order) — not just top to
     bottom in the file.
  2. Individual task retry: a task that fails transiently should be retried
     some number of times before the whole DAG is considered failed, the
     same as Airflow's per-task `retries` parameter.

Fill in the three functions marked # TODO. The `Task` dataclass above them
is given.
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass
class Task:
    name: str
    # called as fn(context) -> mutates context in place
    fn: Callable[[dict], None]
    depends_on: List[str] = field(default_factory=list)
    max_retries: int = 0
    retry_delay_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Part 1 — Topological order (a DAG is a dependency graph, not a list)
# ---------------------------------------------------------------------------

def topological_order(tasks: Dict[str, Task]) -> List[str]:
    """Given a dict of {task_name: Task}, return a list of task names such
    that every task appears AFTER all the tasks in its `depends_on`.

    If there's more than one valid order (e.g. two independent tasks that
    both depend on the same upstream task), any valid order is fine.

    If the tasks contain a dependency cycle (or a task depends on a name
    that doesn't exist in `tasks`), raise ValueError.

    Hint: Kahn's algorithm — compute in-degree (number of dependencies) for
    every task, repeatedly pick a task with in-degree 0, "remove" it
    (decrementing the in-degree of everything that depended on it), and
    append it to the result. If you run out of in-degree-0 tasks before
    placing every task, there's a cycle.
    """
    # Build adjacency list and compute in-degrees
    in_degree = {name: 0 for name in tasks}
    adj = {name: [] for name in tasks}

    # Check for missing dependencies and build graph
    for name, task in tasks.items():
        for dep in task.depends_on:
            if dep not in tasks:
                raise ValueError(
                    f"Task '{name}' depends on missing task '{dep}'"
                )
            adj[dep].append(name)
            in_degree[name] += 1

    # Kahn's algorithm: start with nodes that have in-degree 0
    queue = [name for name, deg in in_degree.items() if deg == 0]
    result = []

    while queue:
        # Pop a task with in-degree 0 (any valid order is fine)
        current = queue.pop(0)
        result.append(current)

        # Decrement in-degree of all tasks that depend on current
        for neighbor in adj[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If we couldn't place all tasks, there's a cycle
    if len(result) != len(tasks):
        raise ValueError("Dependency cycle detected in tasks")

    return result


# ---------------------------------------------------------------------------
# Part 2 — Retry a single task
# ---------------------------------------------------------------------------

def run_task_with_retry(task: Task, context: dict) -> int:
    """Run `task.fn(context)`. If it raises, retry up to `task.max_retries`
    additional times (so max_retries=2 means up to 3 total attempts),
    sleeping `task.retry_delay_seconds` between attempts (use time.sleep).

    If the task eventually succeeds, return the number of attempts it took
    (1 if it succeeded on the first try).

    If it never succeeds, let the LAST exception propagate (don't swallow
    it) after all retries are exhausted.
    """
    last_exception = None
    max_attempts = task.max_retries + 1

    for attempt in range(1, max_attempts + 1):
        try:
            task.fn(context)
            return attempt
        except Exception as e:
            last_exception = e
            if attempt < max_attempts:
                time.sleep(task.retry_delay_seconds)

    # All retries exhausted, propagate the last exception
    raise last_exception 


# ---------------------------------------------------------------------------
# Part 3 — Run the whole DAG
# ---------------------------------------------------------------------------

def run_dag(tasks: Dict[str, Task], context: dict) -> dict:
    """Run every task in `tasks` in an order that respects dependencies
    (use topological_order), passing the SAME `context` dict to every task
    (so a downstream task can read what an upstream task wrote into it —
    e.g. context["extract"] = [...] written by the "extract" task, read by
    the "score" task).

    For each task, run it via run_task_with_retry and record:
      task_results[name] = {"status": "success", "attempts": N}
    on success, or, if it ultimately fails:
      task_results[name] = {
          "status": "failed", "attempts": N, "error": str(exc)
      }
    and then STOP — don't run any tasks that come after a failed task in
    the order (this mirrors Airflow: downstream tasks don't run if an
    upstream one fails).

    Return {"order": <the order you computed>, "task_results": task_results}.
    """
    order = topological_order(tasks)
    task_results = {}

    for name in order:
        task = tasks[name]
        try:
            attempts = run_task_with_retry(task, context)
            task_results[name] = {"status": "success", "attempts": attempts}
        except Exception as e:
            # We need to know how many attempts were made before failure
            # run_task_with_retry propagates the exception after all retries
            # The number of attempts = max_retries + 1
            attempts = task.max_retries + 1
            task_results[name] = {
                "status": "failed",
                "attempts": attempts,
                "error": str(e),
            }
            # Stop running further tasks (Airflow behavior)
            break 

    return {"order": order, "task_results": task_results}
