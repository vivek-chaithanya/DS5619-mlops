#!/usr/bin/env python3
"""
Run this FIRST, before anything else in this lab:

    python generate_for_student.py --student-id <your roll number or institute email>

Generates YOUR OWN copy of data/v1/transactions.csv and
data/v1/transactions.json — same shape and format as everyone else's (500
rows, same columns), but different actual transaction values, seeded
deterministically from your student ID. Two students never get the same
data. Re-running with the same --student-id always reproduces the exact
same file, so it's safe to run again if you need to reset your data/.

Do not hand-edit these files, and do not submit data generated from
someone else's --student-id or from a shared/generic seed — the grader
regenerates your data/ from your own submitted --student-id (recorded in
NOTES.md, see ASSIGNMENT.md) and diffs it against what you turned in.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_shared"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))
from student_seed import seed_from_student_id  # noqa: E402
from generate_fraud_dataset import gen_transaction  # noqa: E402

import csv
import json
import random
from datetime import datetime

N_RECORDS = 500


def _write_csv_and_json(records, out_dir, basename):
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{basename}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    json_path = os.path.join(out_dir, f"{basename}.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    return csv_path, json_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True, help="Your roll number or institute email")
    args = ap.parse_args()

    seed = seed_from_student_id(args.student_id, salt="week02")
    rng = random.Random(seed)
    base_time = datetime(2026, 1, 1)

    records = [gen_transaction(i, base_time, rng, schema_v2=False) for i in range(N_RECORDS)]

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "v1")
    csv_path, json_path = _write_csv_and_json(records, out_dir, "transactions")

    print(f"student_id: {args.student_id}")
    print(f"seed: {seed}")
    print(f"Wrote {len(records)} records ->")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print("\nRecord this seed in NOTES.md when you submit (see ASSIGNMENT.md).")


if __name__ == "__main__":
    main()
