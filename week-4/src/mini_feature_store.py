"""
A tiny, local, dependency-free feature store — enough to demonstrate the
three ideas from this week's lecture without needing a Hopsworks/Feast
account:

  1. Raw data versioning (content-hash based, like DVC).
  2. Feature groups built from raw data, with recorded lineage back to the
     exact raw version and transform that produced them.
  3. A breaking schema change (v1 -> v2 transactions) producing a NEW
     feature group version rather than silently overwriting history.

Everything is stored under a "registry" directory as plain JSON, so you can
open any file and read exactly what was recorded — that transparency is the
point of the exercise.

Fill in the four functions marked # TODO. Helpers above them are done.
"""
import csv
import hashlib
import json
import os
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def _read_csv_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def content_hash(file_path):
    """Sha256 of the file's bytes. Given — this is what makes versioning
    idempotent: the same bytes always produce the same hash."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


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


# ---------------------------------------------------------------------------
# Part 1 — Raw data versioning
# ---------------------------------------------------------------------------

def snapshot_raw_version(input_path, registry_dir):
    """Register `input_path` as a new raw data version under
    `registry_dir/raw_versions/`.

    Must be IDEMPOTENT: if a file with this exact content hash has already
    been snapshotted, return the EXISTING version_id instead of creating a
    duplicate — this is what makes it safe to re-run.

    Steps:
      1. Compute content_hash(input_path).
      2. Look through registry_dir/raw_versions/*/manifest.json for one
         whose "content_hash" matches. If found, return its "version_id".
      3. Otherwise, allocate a new version id with _next_version_id(
         os.path.join(registry_dir, "raw_versions")).
      4. Create registry_dir/raw_versions/{version_id}/ and inside it write
         manifest.json with at least these keys:
           version_id, source_path, content_hash, columns (list, from the
           CSV header), row_count, created_at (use _now()).
      5. Return the version_id (str).
    """
    # Step 1: Compute content hash
    file_hash = content_hash(input_path)

    # Step 2: Check for existing version with same hash
    raw_versions_dir = os.path.join(registry_dir, "raw_versions")
    if os.path.isdir(raw_versions_dir):
        for name in os.listdir(raw_versions_dir):
            manifest_path = os.path.join(raw_versions_dir, name, "manifest.json")
            if os.path.isfile(manifest_path):
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                if manifest.get("content_hash") == file_hash:
                    return manifest["version_id"]

    # Step 3: Allocate new version id
    version_id = _next_version_id(raw_versions_dir)

    # Step 4: Create directory and write manifest
    version_dir = os.path.join(raw_versions_dir, version_id)
    os.makedirs(version_dir, exist_ok=True)

    # Read CSV to get columns and row_count
    rows = _read_csv_rows(input_path)
    columns = list(rows[0].keys()) if rows else []
    row_count = len(rows)

    manifest = {
        "version_id": version_id,
        "source_path": input_path,
        "content_hash": file_hash,
        "columns": columns,
        "row_count": row_count,
        "created_at": _now()
    }

    manifest_path = os.path.join(version_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Step 5: Return version_id
    return version_id


# ---------------------------------------------------------------------------
# Part 2 — Feature engineering (must handle the v1 -> v2 schema change)
# ---------------------------------------------------------------------------

def build_features(rows):
    """Given a list of transaction row-dicts (either v1 OR v2 schema —
    detect which by checking for the "country_code" key vs "country"),
    compute one feature row per distinct card_id with these keys:

      card_id        (str)
      txn_count      (int)   - number of transactions for this card
      avg_amount     (float, rounded to 2 dp) - mean transaction amount
      max_amount     (float, rounded to 2 dp) - max transaction amount
      pct_card_present (float, rounded to 3 dp) - fraction with card_present true
      event_time     (str)   - the MAX timestamp seen for this card (as-is string
                                comparison works fine since timestamps are ISO8601)

    Schema handling:
      - v1 rows have "amount" (already a float-ish string) and "country".
      - v2 rows have "amount_minor_units" (integer string, cents) instead of
        "amount", and "country_code" instead of "country". Convert
        amount_minor_units back to the same unit as v1's amount by dividing
        by 100 before aggregating, so features are comparable across
        versions.
      - "card_present" is the string "True"/"False" in both — treat it as
        true if it equals "True".

    Return: list of feature row dicts, one per card_id, in any order.
    """
    # Group transactions by card_id
    card_transactions = {}
    for row in rows:
        card_id = row["card_id"]
        if card_id not in card_transactions:
            card_transactions[card_id] = []
        card_transactions[card_id].append(row)

    # Detect schema: v2 has "amount_minor_units" and "country_code"
    # v1 has "amount" and "country"
    is_v2 = False
    if rows:
        is_v2 = "amount_minor_units" in rows[0]

    feature_rows = []
    for card_id, txns in card_transactions.items():
        txn_count = len(txns)

        # Extract amounts based on schema
        amounts = []
        card_present_count = 0
        max_timestamp = None

        for txn in txns:
            if is_v2:
                # v2: amount_minor_units is in cents, convert to same unit as v1
                amount = float(txn["amount_minor_units"]) / 100.0
            else:
                # v1: amount is already in the correct unit
                amount = float(txn["amount"])
            amounts.append(amount)

            # card_present is "True"/"False" string
            if txn.get("card_present", "False") == "True":
                card_present_count += 1

            # Track max timestamp
            timestamp = txn["timestamp"]
            if max_timestamp is None or timestamp > max_timestamp:
                max_timestamp = timestamp

        avg_amount = round(sum(amounts) / txn_count, 2)
        max_amount = round(max(amounts), 2)
        pct_card_present = round(card_present_count / txn_count, 3)

        feature_rows.append({
            "card_id": card_id,
            "txn_count": txn_count,
            "avg_amount": avg_amount,
            "max_amount": max_amount,
            "pct_card_present": pct_card_present,
            "event_time": max_timestamp
        })

    return feature_rows


# ---------------------------------------------------------------------------
# Part 3 — Feature group registration (this IS the lineage record)
# ---------------------------------------------------------------------------

def register_feature_group(name, feature_rows, source_version_id, registry_dir, transform_version="v1"):
    """Register a new version of feature group `name`.

    Must NEVER overwrite a previous version — each call creates a new
    incrementing version under registry_dir/feature_groups/{name}/{fg_version_id}/,
    exactly like snapshot_raw_version does for raw data. This is what "a
    breaking schema change creates a new version rather than silently
    mutating history" means in practice.

    Steps:
      1. Allocate fg_version_id via _next_version_id(os.path.join(
         registry_dir, "feature_groups", name)).
      2. Create that directory.
      3. Write features.json inside it containing `feature_rows` (the list
         you were given, as-is).
      4. Write manifest.json inside it with at least these keys:
           feature_group_version_id, name, source_raw_version_id
           (= the source_version_id argument), transform_version, schema
           (sorted list of the keys present in feature_rows[0]), row_count,
           created_at (use _now()).
      5. Return fg_version_id (str).
    """
    # Step 1: Allocate new feature group version id
    fg_dir = os.path.join(registry_dir, "feature_groups", name)
    fg_version_id = _next_version_id(fg_dir)

    # Step 2: Create directory
    version_dir = os.path.join(fg_dir, fg_version_id)
    os.makedirs(version_dir, exist_ok=True)

    # Step 3: Write features.json
    features_path = os.path.join(version_dir, "features.json")
    with open(features_path, "w") as f:
        json.dump(feature_rows, f, indent=2)

    # Step 4: Write manifest.json
    schema = sorted(feature_rows[0].keys()) if feature_rows else []
    manifest = {
        "feature_group_version_id": fg_version_id,
        "name": name,
        "source_raw_version_id": source_version_id,
        "transform_version": transform_version,
        "schema": schema,
        "row_count": len(feature_rows),
        "created_at": _now()
    }

    manifest_path = os.path.join(version_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Step 5: Return fg_version_id
    return fg_version_id


# ---------------------------------------------------------------------------
# Part 4 — Lineage lookup
# ---------------------------------------------------------------------------

def get_lineage(name, fg_version_id, registry_dir):
    """Trace a feature group version back to the raw source it was built
    from, and return a single dict describing the full chain:

      {
        "feature_group": { ...the feature group's manifest.json contents... },
        "raw_source": { ...the manifest.json of the raw version named by
                         the feature group's "source_raw_version_id"... }
      }

    Read both manifest.json files from disk and assemble this dict. Raise
    FileNotFoundError (the default behavior of open() on a missing file is
    fine — don't catch it) if either manifest is missing.
    """
    # Read feature group manifest
    fg_manifest_path = os.path.join(registry_dir, "feature_groups", name, fg_version_id, "manifest.json")
    with open(fg_manifest_path, "r") as f:
        fg_manifest = json.load(f)

    # Read raw source manifest
    source_version_id = fg_manifest["source_raw_version_id"]
    raw_manifest_path = os.path.join(registry_dir, "raw_versions", source_version_id, "manifest.json")
    with open(raw_manifest_path, "r") as f:
        raw_manifest = json.load(f)

    return {
        "feature_group": fg_manifest,
        "raw_source": raw_manifest
    }
