"""
Driver script — wires together the four functions in mini_feature_store.py
into the full flow this lab is about:

  1. Snapshot data/v1/transactions.csv as a raw version.
  2. Build features from it, register as feature group "card_activity" v1.
  3. Snapshot data/v2/transactions.csv (the schema-changed revision) as a
     NEW raw version.
  4. Build features from it (handling the schema change), register as
     feature group "card_activity" v2 — NOT overwriting v1.
  5. Look up lineage for both feature group versions and write the combined
     result to lineage_report.json at the repo root.

This file is complete — you don't need to edit it. Run it with:

    python src/run_pipeline.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mini_feature_store as mfs

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_DIR = os.path.join(REPO_ROOT, ".feature_store")
FEATURE_GROUP_NAME = "card_activity"


def _rows_from(path):
    return mfs._read_csv_rows(path)


def main():
    v1_path = os.path.join(REPO_ROOT, "data", "v1", "transactions.csv")
    v2_path = os.path.join(REPO_ROOT, "data", "v2", "transactions.csv")

    # --- v1 ---
    v1_raw_version = mfs.snapshot_raw_version(v1_path, REGISTRY_DIR)
    v1_features = mfs.build_features(_rows_from(v1_path))
    v1_fg_version = mfs.register_feature_group(
        FEATURE_GROUP_NAME, v1_features, v1_raw_version, REGISTRY_DIR, transform_version="v1"
    )

    # --- v2 (schema changed upstream) ---
    v2_raw_version = mfs.snapshot_raw_version(v2_path, REGISTRY_DIR)
    v2_features = mfs.build_features(_rows_from(v2_path))
    v2_fg_version = mfs.register_feature_group(
        FEATURE_GROUP_NAME, v2_features, v2_raw_version, REGISTRY_DIR, transform_version="v1"
    )

    # --- re-running v1 again should be idempotent: same raw version id ---
    v1_raw_version_again = mfs.snapshot_raw_version(v1_path, REGISTRY_DIR)
    assert v1_raw_version_again == v1_raw_version, (
        "snapshot_raw_version should be idempotent for identical file content"
    )

    lineage_report = {
        "feature_group": FEATURE_GROUP_NAME,
        "v1": mfs.get_lineage(FEATURE_GROUP_NAME, v1_fg_version, REGISTRY_DIR),
        "v2": mfs.get_lineage(FEATURE_GROUP_NAME, v2_fg_version, REGISTRY_DIR),
    }

    out_path = os.path.join(REPO_ROOT, "lineage_report.json")
    with open(out_path, "w") as f:
        json.dump(lineage_report, f, indent=2)

    print(f"v1 raw version: {v1_raw_version} -> feature group version: {v1_fg_version}")
    print(f"v2 raw version: {v2_raw_version} -> feature group version: {v2_fg_version}")
    print(f"idempotency check ok: re-snapshotting v1 returned {v1_raw_version_again}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
