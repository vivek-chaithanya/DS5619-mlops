# NOTES.md — Week 2: Config-Driven Data Pipelines

**Student ID used with `generate_for_student.py`:**
142301024


## What was hardcoded, and what would switching it have required?

The original `pipeline_hardcoded.py` had the following hardcoded values:

1. **`INPUT_PATH = "data/v1/transactions.csv"`** — The input file path was hardcoded. To change it, you would need to edit the source code and redeploy.

2. **`HIGH_VALUE_THRESHOLD = 5000`** — The business logic threshold for "high value" transactions was a literal constant. Changing it required modifying the code and redeploying.

3. **`OUTPUT_PATH = "data/v1/report_hardcoded.json"`** — The output file path was hardcoded. Any change required a code edit.

4. **Format handling** — The script only supported CSV via a dedicated `load_csv()` function. Adding JSON support would have required writing a new `load_json()` function and modifying the `main()` function to choose between them (e.g., via an if/else on file extension), meaning code changes for every new format.

After the refactor, all of these are driven by the YAML config file (`config/pipeline.yaml` or `config/pipeline_json.yaml`). Switching formats, thresholds, or paths now requires only editing the config file — no code changes needed.