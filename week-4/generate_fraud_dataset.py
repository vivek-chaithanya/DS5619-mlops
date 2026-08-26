#!/usr/bin/env python3
"""
Shared Track A dataset generator — DS5619 MLOps labs (Weeks 2-5).

Generates a synthetic credit-card-fraud transaction dataset used as the
running example across the Week 2 (data formats/config), Week 3 (ETL/validation),
Week 4 (versioning/feature store/lineage), and Week 5 (model registry) labs.

Produces two revisions of the dataset (v1 and v2) to simulate a realistic
upstream schema change (Week 4 needs this), and writes each revision out in
CSV, JSON, and Parquet (Week 2 needs multiple formats).

Usage:
    python generate_fraud_dataset.py --out ./data --n 4000 --seed 7619
"""
import argparse
import json
import os
import random
from datetime import datetime, timedelta

import pandas as pd

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "fuel", "travel", "restaurant",
    "online_retail", "utilities", "pharmacy", "entertainment", "atm_withdrawal",
]
COUNTRIES = ["IN", "US", "GB", "AE", "SG", "DE"]


def gen_transaction(i, base_time, rng, schema_v2=False):
    is_fraud = rng.random() < 0.03
    amount = round(rng.lognormvariate(4.2, 1.1), 2)
    if is_fraud:
        amount = round(amount * rng.uniform(2.5, 6.0), 2)
    ts = base_time + timedelta(seconds=rng.randint(0, 60 * 60 * 24 * 30))

    row = {
        "transaction_id": f"txn_{i:07d}",
        "card_id": f"card_{rng.randint(1, 900):05d}",
        "timestamp": ts.isoformat() + "Z",
        "amount": amount,
        "merchant_category": rng.choice(MERCHANT_CATEGORIES),
        "country": rng.choice(COUNTRIES),
        "card_present": rng.random() < 0.55,
        "is_fraud": is_fraud,
    }

    if schema_v2:
        # Simulate a realistic upstream schema change for Week 4:
        # a new field is added, and an existing field is renamed/retyped.
        row["device_fingerprint"] = f"dev_{rng.randint(1, 5000):06d}"  # new field
        row["country_code"] = row.pop("country")  # renamed field
        row["amount_minor_units"] = int(round(row["amount"] * 100))  # retyped (int, minor units)
        del row["amount"]

    return row


def write_all_formats(records, out_dir, basename):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(records)

    csv_path = os.path.join(out_dir, f"{basename}.csv")
    df.to_csv(csv_path, index=False)

    json_path = os.path.join(out_dir, f"{basename}.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    parquet_path = os.path.join(out_dir, f"{basename}.parquet")
    df.to_parquet(parquet_path, index=False)

    return csv_path, json_path, parquet_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=7619)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base_time = datetime(2026, 1, 1)

    # Revision 1 (v1 schema) — used by Weeks 2 and 3
    v1_records = [gen_transaction(i, base_time, rng, schema_v2=False) for i in range(args.n)]
    write_all_formats(v1_records, os.path.join(args.out, "v1"), "transactions")

    # Revision 2 (v2 schema, changed) — used by Week 4 to demonstrate versioning
    # under a real schema change. Deterministic continuation of the same stream.
    v2_records = [gen_transaction(i, base_time, rng, schema_v2=True) for i in range(args.n, args.n + int(args.n * 0.25))]
    write_all_formats(v2_records, os.path.join(args.out, "v2"), "transactions")

    print(f"Wrote v1: {len(v1_records)} records -> {args.out}/v1/transactions.{{csv,json,parquet}}")
    print(f"Wrote v2: {len(v2_records)} records -> {args.out}/v2/transactions.{{csv,json,parquet}} (schema changed)")
    print(f"Fraud rate v1: {sum(r['is_fraud'] for r in v1_records) / len(v1_records):.3%}")


if __name__ == "__main__":
    main()
