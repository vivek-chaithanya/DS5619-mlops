"""
Driver script — wires together mini_model_registry.py into the flow this lab
is about. Complete, don't edit. Run with:

    python src/run_pipeline.py

What it does:
  1. Registers candidate_a and candidate_b as two versions of model
     "fraud-detector" (their metrics come from data/candidate_*/metrics.json
     — candidate_a is deliberately weak, candidate_b is deliberately strong).
  2. Tries to promote candidate_a straight to Production with NO model card
     yet — this MUST fail with GovernanceError (no card). That failure is
     expected and printed, not a bug.
  3. Fills in a model card for candidate_a from model_card_fields.json and
     tries again — this MUST fail with GovernanceError a second time
     (card exists now, but candidate_a's f1=0.58 is below the 0.70 bar).
  4. Fills in the same card content for candidate_b and promotes it to
     Production — this MUST succeed (f1=0.79 clears the bar).
  5. Writes registry_summary.json showing what's currently in Production.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mini_model_registry as reg

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_DIR = os.path.join(REPO_ROOT, ".model_registry")
MODEL_NAME = "fraud-detector"


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    card_fields = _load_json(os.path.join(REPO_ROOT, "model_card_fields.json"))
    for key, value in card_fields.items():
        if "TODO" in value:
            raise SystemExit(
                f"model_card_fields.json still has a TODO placeholder in '{key}'. "
                f"Fill it in with real content before running the pipeline."
            )

    metrics_a = _load_json(os.path.join(REPO_ROOT, "data", "candidate_a", "metrics.json"))
    metrics_b = _load_json(os.path.join(REPO_ROOT, "data", "candidate_b", "metrics.json"))

    v_a = reg.register_model(MODEL_NAME, os.path.join(REPO_ROOT, "data", "candidate_a", "model.json"), metrics_a, REGISTRY_DIR)
    v_b = reg.register_model(MODEL_NAME, os.path.join(REPO_ROOT, "data", "candidate_b", "model.json"), metrics_b, REGISTRY_DIR)
    print(f"Registered candidate_a as {v_a}, candidate_b as {v_b}")

    try:
        reg.promote_model(MODEL_NAME, v_a, "Production", REGISTRY_DIR)
        raise SystemExit("ERROR: promoting a card-less model to Production should have failed but didn't.")
    except reg.GovernanceError as e:
        print(f"[expected] promoting {v_a} with no card was blocked: {e}")

    reg.generate_model_card(MODEL_NAME, v_a, card_fields, REGISTRY_DIR)
    try:
        reg.promote_model(MODEL_NAME, v_a, "Production", REGISTRY_DIR)
        raise SystemExit("ERROR: promoting a low-f1 model to Production should have failed but didn't.")
    except reg.GovernanceError as e:
        print(f"[expected] promoting {v_a} with f1 below threshold was blocked: {e}")

    reg.generate_model_card(MODEL_NAME, v_b, card_fields, REGISTRY_DIR)
    manifest_b = reg.promote_model(MODEL_NAME, v_b, "Production", REGISTRY_DIR)
    print(f"Promoted {v_b} to Production. History: {manifest_b['history']}")

    prod = reg.get_production_model(MODEL_NAME, REGISTRY_DIR)
    summary = {
        "model_name": MODEL_NAME,
        "production_version": prod["version_id"] if prod else None,
        "production_metrics": prod["metrics"] if prod else None,
    }
    out_path = os.path.join(REPO_ROOT, "registry_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_path}: production version is {summary['production_version']}")


if __name__ == "__main__":
    main()
