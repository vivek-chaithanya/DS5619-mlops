# NOTES.md — Week 7: CI/CD Integration Testing

**Student ID used with `generate_for_student.py`:**
142301024

**Seed generated:** 3361648908


## Why gate integration-test on needs: [lint, unit-test]?

The integration-test job runs Docker build and container execution, which is significantly more expensive in terms of time (minutes vs seconds) and compute resources than lint and unit-test. By gating it behind `needs: [lint, unit-test]`, we avoid wasting CI minutes building and running containers for code that has syntax errors, style violations, or failing unit tests. This is a classic CI optimization: fail fast on cheap checks before spending resources on expensive integration tests.
