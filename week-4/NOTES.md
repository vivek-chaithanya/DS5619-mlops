# NOTES.md — Week 4: Versioning, Feature Store & Lineage

**Student ID used with `generate_for_student.py`:**
142301024

**Deterministic Random Seed:**
1413445892 (derived via `seed_from_student_id("142301024", salt="week04")`)


## v1 vs. v2 Manifest Comparison

### Raw Data Manifests (in `.feature_store/raw_versions/`)
- **v1 raw source (`version_id: v1`)**:
  - `row_count`: 500
  - `content_hash`: `3ae42e3b3de77ad2238bebc123cd1af2180872e65ee2b43813f84914bd251022`
  - `columns`: `["transaction_id", "card_id", "timestamp", "amount", "merchant_category", "country", "card_present", "is_fraud"]`
  - Notes: Uses base currency float `amount` and string `country`. No `device_fingerprint`.

- **v2 raw source (`version_id: v2`)**:
  - `row_count`: 125
  - `content_hash`: `2e5411f82667ed59adb61d14cf200b36009a739156d1323b397882da874fc140`
  - `columns`: `["transaction_id", "card_id", "timestamp", "merchant_category", "card_present", "is_fraud", "device_fingerprint", "country_code", "amount_minor_units"]`
  - Notes: Upstream breaking schema change: `amount` replaced by `amount_minor_units` (integer cents), `country` renamed to `country_code`, and added `device_fingerprint`.

### Feature Group Manifests (in `.feature_store/feature_groups/card_activity/`)
- **v1 feature group (`fg_version_id: v1`)**:
  - `source_raw_version_id`: `v1`
  - `transform_version`: `v1`
  - `row_count`: 383 unique card entities (from 500 raw transactions)
  - `schema`: `["avg_amount", "card_id", "event_time", "max_amount", "pct_card_present", "txn_count"]`

- **v2 feature group (`fg_version_id: v2`)**:
  - `source_raw_version_id`: `v2`
  - `transform_version`: `v1`
  - `row_count`: 115 unique card entities (from 125 raw transactions)
  - `schema`: `["avg_amount", "card_id", "event_time", "max_amount", "pct_card_present", "txn_count"]`

### Key Differences & Invariant Observations:
1. **Source Raw Version Linkage**: `source_raw_version_id` points directly to the corresponding immutable raw data version (`v1` vs `v2`).
2. **Row Count Differences**: v1 feature group contains 383 aggregated cards, while v2 feature group contains 115 aggregated cards due to the different volume of transactions in each feed (500 vs 125).
3. **Content Hashes**: The raw content hashes differ (`3ae42e...` vs `2e5411...`), guaranteeing distinct cryptographic identity for each dataset revision.
4. **Feature Schema Invariance**: Despite breaking upstream changes (renamed fields, retyped units, added columns), the output feature group schema is **identical** across both versions. The transformation layer normalizes raw variations so downstream consumers (model training / inference) receive consistent feature definitions.


## Why Treat `amount_minor_units` Differently from `amount`?

In **v1**, transaction values are represented by `amount` as a standard floating-point number in the base currency unit (e.g., $67.45).

In **v2**, upstream systems migrated to `amount_minor_units` as an integer representing cents or minor units (e.g., 6745 cents = $67.45).

`build_features` must explicitly detect the schema and divide `amount_minor_units` by `100.0` before computing aggregations (`avg_amount` and `max_amount`). Without this conversion:
- An account with average spend of $50 would have an `avg_amount` of `50.0` in v1, but `5000.0` in v2.
- Aggregates would differ by two orders of magnitude (100×) solely due to schema changes.
- Downstream machine learning models trained on historical v1 data would experience severe feature drift and catastrophic prediction degradation when evaluated or served on v2 data.

By normalizing `amount_minor_units` back to the base currency unit at feature extraction time, both feature group versions maintain identical semantic definitions, distributions, and units.