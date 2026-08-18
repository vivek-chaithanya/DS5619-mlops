#!/usr/bin/env python3
"""
Run this FIRST, before anything else in this lab:

    python generate_for_student.py --student-id <your roll number or institute email>

Generates YOUR OWN copy of data/raw_transactions.csv: 600 rows, 7 known
violations (2 null amounts, 1 negative amount, 1 invalid category, 1 invalid
country code, 1 null card_id, 1 duplicate transaction_id) — same as
everyone else's IN TYPE AND COUNT, but at DIFFERENT row indices, seeded
deterministically from your student ID. You don't know which specific rows
are broken until your own validation suite finds them; neither does anyone
you might copy an answer from, because their broken rows are somewhere
else.

Re-running with the same --student-id always reproduces the exact same
file. Record your --student-id in NOTES.md when you submit — the grader
regenerates data/ from it and diffs against what you committed.
"""
import argparse
import csv
import os
import random
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_shared"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))
from student_seed import seed_from_student_id  # noqa: E402
from generate_fraud_dataset import gen_transaction  # noqa: E402

N_ROWS = 600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True, help="Your roll number or institute email")
    args = ap.parse_args()

    seed = seed_from_student_id(args.student_id, salt="week03")
    rng = random.Random(seed)
    base_time = datetime(2026, 2, 1)
    rows = [gen_transaction(i, base_time, rng, schema_v2=False) for i in range(N_ROWS)]

    # Pick 7 distinct row indices (away from the very first/last few rows,
    # just so nothing edge-cases oddly) for the 7 violation types, at
    # positions seeded from the student's own ID.
    candidate_indices = list(range(5, N_ROWS - 5))
    rng.shuffle(candidate_indices)
    idx_null_amount_1, idx_null_amount_2, idx_negative_amount, idx_bad_category, \
        idx_bad_country, idx_null_card, idx_dup_target = candidate_indices[:7]
    idx_dup_source = candidate_indices[7]  # a distinct 8th row whose id gets cloned

    rows[idx_null_amount_1]["amount"] = None
    rows[idx_null_amount_2]["amount"] = None
    rows[idx_negative_amount]["amount"] = -450.00
    rows[idx_bad_category]["merchant_category"] = "crypto_kiosk"
    rows[idx_bad_country]["country"] = "ZZ"
    rows[idx_null_card]["card_id"] = None
    rows[idx_dup_target]["transaction_id"] = rows[idx_dup_source]["transaction_id"]

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "raw_transactions.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"student_id: {args.student_id}")
    print(f"seed: {seed}")
    print(f"Wrote {N_ROWS} rows (7 known violations, positions vary per student) -> {out_path}")
    print("\nRecord this seed in NOTES.md when you submit (see ASSIGNMENT.md).")


if __name__ == "__main__":
    main()
