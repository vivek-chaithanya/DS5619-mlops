#!/usr/bin/env python3
"""
Run this FIRST, before anything else in this lab:

    python generate_for_student.py --student-id <your roll number or institute email>

Generates YOUR OWN copy of data/weights.json (256 weights, 40 informative)
and data/eval_data.json (300 labeled examples generated from those exact
weights) — same structure as everyone else's, different actual numbers,
seeded deterministically from your student ID.

The pruning/quantization signal this lab is built around comes from a
structural guarantee, not luck: informative weights are always boosted to
at least magnitude 0.5, noise weights are always kept under magnitude 0.03.
That ~17x magnitude gap means magnitude-based pruning at 70% sparsity
(zeroing 179 of 256 weights) can NEVER reach into the 40 informative
weights no matter what the random draw looks like — there are always 216
noise weights below every informative one. So the "pruned_dense_fp32 keeps
near-baseline accuracy at 70% sparsity, then falls off a cliff at 95%"
story holds for every student ID; this script doesn't need Week 8/10-style
retry-and-check logic.

Record your --student-id in NOTES.md when you submit — the grader
regenerates data/ from it and diffs against what you committed.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_shared"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))
from student_seed import seed_from_student_id  # noqa: E402

N_WEIGHTS = 256
N_IMPORTANT = 40
N_EVAL_EXAMPLES = 300


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True, help="Your roll number or institute email")
    args = ap.parse_args()

    seed = seed_from_student_id(args.student_id, salt="week11")
    rng = random.Random(seed)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)

    important_idx = rng.sample(range(N_WEIGHTS), N_IMPORTANT)
    weights = [0.0] * N_WEIGHTS
    for i in range(N_WEIGHTS):
        if i in important_idx:
            weights[i] = rng.uniform(-2.0, 2.0)
            if abs(weights[i]) < 0.5:
                weights[i] += 0.5 if weights[i] >= 0 else -0.5
        else:
            weights[i] = rng.uniform(-0.03, 0.03)  # near-zero noise weights

    with open(os.path.join(out_dir, "weights.json"), "w") as f:
        json.dump({"weights": weights}, f, indent=2)

    eval_examples = []
    for _ in range(N_EVAL_EXAMPLES):
        features = [rng.uniform(-1.0, 1.0) for _ in range(N_WEIGHTS)]
        score = sum(w * x for w, x in zip(weights, features))
        label = 1 if score > 0 else 0
        eval_examples.append({"features": features, "label": label})

    with open(os.path.join(out_dir, "eval_data.json"), "w") as f:
        json.dump({"examples": eval_examples}, f)

    print(f"student_id: {args.student_id}")
    print(f"seed: {seed}")
    print(f"Wrote {N_WEIGHTS} weights ({N_IMPORTANT} informative) -> {out_dir}/weights.json")
    print(f"Wrote {N_EVAL_EXAMPLES} eval examples -> {out_dir}/eval_data.json")
    print("\nRecord this seed in NOTES.md when you submit (see README.md).")


if __name__ == "__main__":
    main()
