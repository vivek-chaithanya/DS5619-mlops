#!/usr/bin/env python3
"""
Run this FIRST, before anything else in this lab:

    python generate_for_student.py --student-id <your roll number or institute email>

Generates YOUR OWN copy of data/candidate_a/{model.json,metrics.json} and
data/candidate_b/{model.json,metrics.json} — same STRUCTURE as everyone
else's (candidate_a always fails the f1 >= 0.70 production bar, candidate_b
always clears it — otherwise the lab's governance-gate point wouldn't
demonstrate anything), but different actual threshold/metric numbers,
seeded deterministically from your student ID.

Record your --student-id in NOTES.md when you submit — the grader
regenerates data/ from it and diffs against what you committed, and checks
that your registry_summary.json's production metrics actually match YOUR
candidate_b, not a generic or shared example.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_shared"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))
from student_seed import seed_from_student_id  # noqa: E402


def _metrics_for(rng, f1_low, f1_high):
    precision = round(rng.uniform(f1_low, f1_high) + rng.uniform(-0.03, 0.05), 3)
    precision = max(0.05, min(0.99, precision))
    recall = round(rng.uniform(f1_low, f1_high) + rng.uniform(-0.05, 0.03), 3)
    recall = max(0.05, min(0.99, recall))
    f1 = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) > 0 else 0.0
    auc = round(min(0.99, max(f1, f1 + rng.uniform(0.03, 0.12))), 3)
    return {"precision": precision, "recall": recall, "f1": f1, "auc": auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True, help="Your roll number or institute email")
    args = ap.parse_args()

    seed = seed_from_student_id(args.student_id, salt="week05")
    rng = random.Random(seed)
    repo_root = os.path.dirname(os.path.abspath(__file__))

    # candidate_a: deliberately below the 0.70 production f1 bar
    metrics_a = _metrics_for(rng, 0.40, 0.60)
    while metrics_a["f1"] >= 0.68:  # guard against an unlucky roll landing too close to the bar
        metrics_a = _metrics_for(rng, 0.40, 0.60)
    threshold_a = round(rng.uniform(150.0, 320.0), 1)

    # candidate_b: deliberately above the 0.70 production f1 bar
    metrics_b = _metrics_for(rng, 0.75, 0.92)
    while metrics_b["f1"] < 0.72:
        metrics_b = _metrics_for(rng, 0.75, 0.92)
    threshold_b = round(rng.uniform(400.0, 600.0), 1)

    for name, metrics, threshold in (
        ("candidate_a", metrics_a, threshold_a),
        ("candidate_b", metrics_b, threshold_b),
    ):
        out_dir = os.path.join(repo_root, "data", name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "model.json"), "w") as f:
            json.dump(
                {"type": "amount_threshold_classifier", "threshold": threshold,
                 "trained_on": "card_activity_features_v1"},
                f, indent=2,
            )
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    print(f"student_id: {args.student_id}")
    print(f"seed: {seed}")
    print(f"candidate_a: f1={metrics_a['f1']} (below 0.70 bar)")
    print(f"candidate_b: f1={metrics_b['f1']} (clears 0.70 bar)")
    print("\nRecord this seed in NOTES.md when you submit (see ASSIGNMENT.md).")


if __name__ == "__main__":
    main()
