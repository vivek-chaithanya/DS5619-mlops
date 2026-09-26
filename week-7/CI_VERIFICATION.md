## Workflow run

https://github.com/vivek-chaithanya/DS5619-mlops/actions/runs/36263816507

## Job summary

For each job, note pass/fail and how long it took:

- `lint`: Pass (8s)
- `unit-test`: Pass (9s)
- `integration-test`: Pass (19s)

## What broke on the way there (optional but useful)

Initially, the `lint` and `unit-test` jobs failed with `Process completed with exit code 1` because the project files (`src/`, `tests/`, and `requirements.txt`) were structured inside a `week-7` subdirectory rather than the repository root. This caused flake8 and pytest to look in the wrong paths. We resolved this by updating the CI workflow file paths to correctly target the `week-7` subdirectory context.