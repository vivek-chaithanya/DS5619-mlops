# NOTES.md — Week 9: Orchestrated Batch Scoring

**Student ID used with `generate_for_student.py`:**
`142301024`

**Deterministic Seed:**
`2083192409` (generated with salt `week09`)


## notify retry count

**3 attempts** (first 2 attempts failed with simulated transient errors, 3rd attempt succeeded).

The `notify` task simulated an unreliable downstream consumer (such as a third-party webhook, Slack alert, or HTTP notification service). With `max_retries=3` and `retry_delay_seconds=0.1`, `run_task_with_retry` executed attempt 1 (failed), slept 0.1s, executed attempt 2 (failed), slept 0.1s, and executed attempt 3 (succeeded), completing the task successfully on attempt 3.


## Branch independence

The `load` task succeeds independently of `notify` because they are **sibling tasks** in the DAG — both depend directly on `score`, but neither depends on the other (`load.depends_on = ['score']`, `notify.depends_on = ['score']`).

In a properly designed DAG orchestrator:
1. **Parallel / Sibling Branch Isolation:** Failures or retries in one branch must not block or abort unrelated parallel branches that share the same upstream parent.
2. **Topological Order Execution:** The orchestrator executes tasks in an order that respects upstream dependencies (`extract` -> `score` -> `load` -> `notify` or `extract` -> `score` -> `notify` -> `load`).
3. **Targeted Failure Containment:** Only downstream descendants of a failed task are prevented from executing. If `notify` had ultimately failed after exhausting all 3 retries, `load` would already have successfully completed and written `batch_report.json`.

This demonstrates that DAG orchestration preserves true branch isolation: a transient failure in an auxiliary notification webhook does not prevent persisting the core batch scoring dataset.


## Retry cost under FinOps

If `notify` were a metered API call (e.g., an external webhook, SMS gateway, or commercial notification API billed per request attempt), a naive, uniform retry policy (fixed immediate retries for any failure) would introduce severe operational and financial risks:

1. **Runaway Cost during Outages:** When downstream systems undergo persistent outages, blindly retrying all failed calls exhausts budgets without delivering messages. A $0.01 internal database write retry has vastly different cost implications than a $0.50/SMS or metered LLM API retry.
2. **Thundering Herd & Cascading Failures:** Fixed-delay retries cause simultaneous retries across pipeline workers, amplifying load on already degraded target systems.
3. **Risk of Duplicate Side Effects:** If an API endpoint timed out but actually processed the payload, uncoordinated retries without idempotency guarantees result in duplicate charges and duplicate messages.

### Recommended Architectural Improvements under FinOps:
- **Exponential Backoff with Full Jitter:** Instead of a fixed delay, increase delay exponentially ($t_{retry} = base \times 2^{attempt} + jitter$) to alleviate traffic spikes.
- **Strict Error Classification:** Differentiate between transient errors (HTTP 502/503/504, network timeouts) that warrant retries, and permanent errors (HTTP 400 Bad Request, 401/403 Unauthorized, 422 Unprocessable Entity) that should fail fast immediately without wasting paid API calls.
- **Global Retry Budgets:** Enforce a system-wide or pipeline-level retry budget (e.g., maximum 5% retry overhead across all tasks) so a single downstream failure cannot consume unlimited calls.
- **Idempotency Keys:** Include a unique `Idempotency-Key` or transaction hash in every webhook request to ensure the receiving service does not charge or execute twice if the response was lost.
- **Cost-Tiered Retry Policies:** Allocate retry allowances proportionally to task cost and business criticality (e.g., critical data persistence gets 3 retries, non-critical notifications get 1 retry or divert directly to a dead-letter queue).
