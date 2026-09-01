"""
A tiny local model registry — enough to demonstrate the governance ideas from
this week's lecture without needing an MLflow/W&B server:

  1. The registry as an artifact store (register_model) — one immutable,
     named version per trained model, not forty nearly-identical runs.
  2. The model card as the governance control (generate_model_card) — must
     actually be filled in, not just present.
  3. Promotion between stages (promote_model) as a GATE, not a rename — you
     cannot reach Production without a complete card and metrics that clear
     the bar. Promoting a new model to Production auto-archives whichever
     version was there before, so "what's in production" always has exactly
     one answer.

Fill in the four functions marked # TODO. Helpers/constants above them are
done.
"""
import json
import os
from datetime import datetime, timezone

PRODUCTION_F1_THRESHOLD = 0.70
REQUIRED_CARD_FIELDS = ["intended_use", "training_data", "limitations", "ethical_considerations"]


class GovernanceError(Exception):
    """Raised when a promotion is attempted that violates a governance rule."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _next_version_id(existing_dir):
    """Given a directory of existing v1/, v2/, ... subfolders, return the
    next version id string. Given — you don't need to touch this."""
    if not os.path.isdir(existing_dir):
        return "v1"
    nums = []
    for name in os.listdir(existing_dir):
        if name.startswith("v") and name[1:].isdigit():
            nums.append(int(name[1:]))
    return f"v{max(nums, default=0) + 1}"


def _model_dir(registry_dir, name, version_id):
    return os.path.join(registry_dir, "models", name, version_id)


# ---------------------------------------------------------------------------
# Part 1 — Register a model version (the artifact store)
# ---------------------------------------------------------------------------

def register_model(name, model_path, metrics, registry_dir):
    """Register a new version of model `name` in the registry.

    Steps:
      1. Allocate version_id via _next_version_id(os.path.join(registry_dir,
         "models", name)).
      2. Create _model_dir(registry_dir, name, version_id).
      3. Copy the model file's contents into that directory as "model.json"
         (read model_path as JSON, write it back out — this is your
         "artifact").
      4. Write manifest.json in that directory with at least these keys:
           version_id, name, metrics (the dict you were given, as-is),
           stage (str, initial value "None" — matches the convention that a
           freshly-registered model isn't in any deployment stage yet),
           created_at (use _now()).
      5. Return version_id (str).
    """
    version_id = _next_version_id(os.path.join(registry_dir, "models", name))
    model_dir = _model_dir(registry_dir, name, version_id)
    os.makedirs(model_dir, exist_ok=True)

    with open(model_path) as f:
        model_contents = json.load(f)
    with open(os.path.join(model_dir, "model.json"), "w") as f:
        json.dump(model_contents, f, indent=2)

    manifest = {
        "version_id": version_id,
        "name": name,
        "metrics": metrics,
        "stage": "None",
        "created_at": _now(),
    }
    with open(os.path.join(model_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return version_id


# ---------------------------------------------------------------------------
# Part 2 — Generate a model card (must be genuinely filled in, not just present)
# ---------------------------------------------------------------------------

def generate_model_card(name, version_id, card_fields, registry_dir):
    """Validate and write a model card for an already-registered model
    version.

    `card_fields` is a dict that should contain every key in
    REQUIRED_CARD_FIELDS, each mapped to a non-empty string that does NOT
    contain the literal substring "TODO" (a card with a TODO in it isn't
    actually filled in — reject it).

    Steps:
      1. For each key in REQUIRED_CARD_FIELDS: if it's missing from
         card_fields, or its value is empty/whitespace-only, or its value
         contains "TODO", raise ValueError naming the offending field.
      2. Read the version's existing manifest.json (from _model_dir(...)) to
         pull in its "metrics" for the card.
      3. Write model_card.json into _model_dir(registry_dir, name,
         version_id) containing: name, version_id, the fields from
         card_fields, metrics (from step 2), created_at (use _now()).
      4. Return the path you wrote to.
    """
    for field in REQUIRED_CARD_FIELDS:
        value = card_fields.get(field)
        if value is None or not str(value).strip():
            raise ValueError(f"model card field '{field}' is missing or empty")
        if "TODO" in str(value):
            raise ValueError(f"model card field '{field}' still contains TODO")

    model_dir = _model_dir(registry_dir, name, version_id)
    with open(os.path.join(model_dir, "manifest.json")) as f:
        manifest = json.load(f)

    card = {
        "name": name,
        "version_id": version_id,
        **card_fields,
        "metrics": manifest["metrics"],
        "created_at": _now(),
    }
    card_path = os.path.join(model_dir, "model_card.json")
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)

    return card_path


# ---------------------------------------------------------------------------
# Part 3 — Promote a model version (the governance gate)
# ---------------------------------------------------------------------------

def promote_model(name, version_id, target_stage, registry_dir):
    """Move model `name` version `version_id` to `target_stage`
    ("Staging" or "Production").

    Governance rule enforced HERE, not just documented: promoting to
    "Production" requires BOTH of:
      (a) a model_card.json exists for this version (use
          os.path.exists(os.path.join(_model_dir(...), "model_card.json"))).
      (b) this version's metrics["f1"] >= PRODUCTION_F1_THRESHOLD (read
          metrics from its manifest.json).
    If either fails, raise GovernanceError with a message saying which
    condition failed. Promotion to "Staging" has no such gate.

    On a successful promotion to "Production": if any OTHER version of the
    same model `name` currently has stage == "Production", set that other
    version's stage to "Archived" first (so there is at most one Production
    version at a time — write its updated manifest.json back to disk).

    Then:
      1. Update this version's manifest.json: set "stage" to target_stage.
      2. Append an entry {"from_stage": <old stage>, "to_stage":
         target_stage, "at": _now()} to a list under manifest["history"]
         (create the list if it doesn't exist yet — never overwrite earlier
         entries, this is the audit trail).
      3. Write the updated manifest.json back to disk.
      4. Return the updated manifest (dict).
    """
    model_dir = _model_dir(registry_dir, name, version_id)
    manifest_path = os.path.join(model_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    if target_stage == "Production":
        card_path = os.path.join(model_dir, "model_card.json")
        if not os.path.exists(card_path):
            raise GovernanceError(
                f"cannot promote {name} {version_id} to Production: no model_card.json"
            )
        if manifest["metrics"]["f1"] < PRODUCTION_F1_THRESHOLD:
            raise GovernanceError(
                f"cannot promote {name} {version_id} to Production: "
                f"f1={manifest['metrics']['f1']} < {PRODUCTION_F1_THRESHOLD}"
            )

        models_dir = os.path.join(registry_dir, "models", name)
        if os.path.isdir(models_dir):
            for other_version_id in os.listdir(models_dir):
                if other_version_id == version_id:
                    continue
                other_manifest_path = os.path.join(models_dir, other_version_id, "manifest.json")
                if not os.path.exists(other_manifest_path):
                    continue
                with open(other_manifest_path) as f:
                    other_manifest = json.load(f)
                if other_manifest.get("stage") == "Production":
                    other_manifest["stage"] = "Archived"
                    with open(other_manifest_path, "w") as f:
                        json.dump(other_manifest, f, indent=2)

    old_stage = manifest.get("stage", "None")
    manifest["stage"] = target_stage
    manifest.setdefault("history", []).append(
        {"from_stage": old_stage, "to_stage": target_stage, "at": _now()}
    )
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest


# ---------------------------------------------------------------------------
# Part 4 — Look up what's currently in production
# ---------------------------------------------------------------------------

def get_production_model(name, registry_dir):
    """Return the manifest (dict) of whichever version of model `name` is
    currently in stage "Production", by scanning every version's
    manifest.json under registry_dir/models/{name}/.

    Return None if no version is currently in Production.
    """
    models_dir = os.path.join(registry_dir, "models", name)
    if not os.path.isdir(models_dir):
        return None
    for version_id in os.listdir(models_dir):
        manifest_path = os.path.join(models_dir, version_id, "manifest.json")
        if not os.path.exists(manifest_path):
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("stage") == "Production":
            return manifest
    return None
