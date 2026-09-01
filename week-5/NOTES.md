# NOTES.md — Week 5: Model Registry Governance

**Student ID used with `generate_for_student.py`:**
142301024

## Which candidate reached Production, and why?

**candidate_b (v2)** reached Production.

**Why:**
- candidate_a (v1) had f1=0.495, which is below the PRODUCTION_F1_THRESHOLD of 0.70
- candidate_b (v2) had f1=0.899, which clears the 0.70 bar
- Both candidates received model cards (generated from model_card_fields.json)
- The governance gate in `promote_model` enforces TWO conditions for Production promotion:
  1. A complete model_card.json must exist for the version
  2. The version's metrics["f1"] must be >= PRODUCTION_F1_THRESHOLD (0.70)
- candidate_a failed condition 2 (f1 too low), candidate_b passed both conditions

## Gating stale feature data

To block promotion of a model trained on stale (>30-day-old) feature data, I would need to add the following to `promote_model`'s gate:

1. **Track training data freshness**: Add a `training_data_date` or `feature_data_date` field to the manifest.json when registering the model (in `register_model`). This could be extracted from the feature group metadata (e.g., the latest event timestamp in the training data) or passed explicitly.

2. **Add a staleness check in `promote_model`**: When promoting to Production, after checking the model card and f1 threshold, also verify:
   ```python
   from datetime import datetime, timezone
   training_date = datetime.fromisoformat(manifest["training_data_date"])
   age_days = (datetime.now(timezone.utc) - training_date).days
   if age_days > 30:
       raise GovernanceError(f"cannot promote to Production: training data is {age_days} days old (>30 day limit)")
   ```

3. **Optional — Make the threshold configurable**: Similar to PRODUCTION_F1_THRESHOLD, add a `MAX_TRAINING_DATA_AGE_DAYS = 30` constant to make it adjustable without code changes.

This would require the training pipeline (or the caller of `register_model`) to supply the training data timestamp, which is a reasonable requirement for governance.

## Scaling the gate to 40 candidates

**What would NOT need to change:**
- The `register_model` function already handles arbitrary numbers of versions — it uses `_next_version_id` to allocate a unique version ID (v1, v2, ..., v40) by scanning existing directories. No modification needed.
- The `generate_model_card` function works per-version and would handle 40 cards the same way as 2.
- The `promote_model` logic is already version-agnostic — it checks the specific version's card and metrics, and archives the prior Production version. This works identically whether there are 2 or 40 candidates.
- The `get_production_model` scan is O(n) in versions and would work fine with 40.

**What WOULD need to change (or be added) for a 40-candidate workflow:**
1. **Bulk registration helper**: A convenience function to register multiple candidates in one call (e.g., `register_models(name, model_paths, metrics_list, registry_dir)`) to avoid repetitive calls.
2. **Candidate comparison/ranking**: With 40 candidates, you'd want a way to query "which version has the highest f1?" or "show all versions above the threshold" before deciding which to promote. Could add a `list_versions(name, registry_dir)` or `get_best_version(name, registry_dir, metric="f1")` function.
3. **Automated promotion pipeline**: Instead of manually calling `promote_model` for each candidate, you'd want a script that evaluates all registered versions and promotes the best one that clears the bar (or the top-k to Staging).
4. **Storage considerations**: 40 model artifacts + manifests + cards is still trivial for local disk, but if scaling further, you'd want to consider artifact deduplication (models with identical thresholds/weights) or cloud storage integration.
5. **UI/observability**: A CLI or simple web view to visualize the candidate leaderboard would become necessary for human-in-the-loop decisions.

The core governance logic (card requirement + f1 threshold + single-production-version) is already designed to scale — it's the *workflow around it* that would need ergonomic improvements for 40 candidates.