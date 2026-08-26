#!/usr/bin/env python3
"""
Run this FIRST, before anything else in this lab:

    python generate_for_student.py --student-id <your roll number or institute email>

Generates YOUR OWN copy of data/v1/transactions.csv (500 rows) and
data/v2/transactions.csv (125 rows, schema-changed) — same shape as
everyone else's, different actual values, seeded deterministically from
your student ID. Re-running with the same --student-id always reproduces
the exact same files.

Record your --student-id in NOTES.md when you submit — the grader
regenerates data/ from it and diffs against what you committed, including
your `.feature_store/` manifests (which embed a content hash of these exact
files, so mismatched data shows up immediately as a mismatched hash).
"""
import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_shared"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))
from student_seed import seed_from_student_id  # noqa: E402
from generate_fraud_dataset import gen_transaction  # noqa: E402

N_V1 = 500
N_V2 = 125


def _write_csv(records, out_dir, basename):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{basename}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True, help="Your roll number or institute email")
    args = ap.parse_args()

    seed = seed_from_student_id(args.student_id, salt="week04")
    rng = random.Random(seed)
    base_time = datetime(2026, 1, 1)

    repo_root = os.path.dirname(os.path.abspath(__file__))

    v1_records = [gen_transaction(i, base_time, rng, schema_v2=False) for i in range(N_V1)]
    v1_path = _write_csv(v1_records, os.path.join(repo_root, "data", "v1"), "transactions")

    v2_records = [
        gen_transaction(i, base_time, rng, schema_v2=True) for i in range(N_V1, N_V1 + N_V2)
    ]
    v2_path = _write_csv(v2_records, os.path.join(repo_root, "data", "v2"), "transactions")

    print(f"student_id: {args.student_id}")
    print(f"seed: {seed}")
    print(f"Wrote {len(v1_records)} v1 records -> {v1_path}")
    print(f"Wrote {len(v2_records)} v2 records -> {v2_path}")
    print("\nRecord this seed in NOTES.md when you submit (see ASSIGNMENT.md).")


if __name__ == "__main__":
    main()
