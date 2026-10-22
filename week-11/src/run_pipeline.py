"""
Driver script — wires compression.py into the flow this lab is about.
Complete, don't edit. Run with:

    python src/run_pipeline.py

Benchmarks four variants of the same weight set:
  1. baseline_fp32       - unmodified weights, 4 bytes/param.
  2. pruned_dense_fp32   - 70% pruned by magnitude, but STILL stored as a
                           dense FP32 array (4 bytes/param) — this is the
                           lecture's "common pitfall": pruning alone doesn't
                           shrink anything without a sparse-aware format.
  3. pruned_sparse_coo   - the SAME pruned weights, but sized (with your
                           sparse_coo_size_bytes) as they'd actually take
                           up in a sparse (COO-style) format:
                           only non-zero values are stored, each paying for
                           its value (4 bytes) AND its index (4 bytes).
  4. quantized_int8      - unpruned weights, quantized to INT8 (1 byte/param).

Writes compression_report.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compression as comp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRUNE_SPARSITY = 0.70


def main():
    with open(os.path.join(REPO_ROOT, "data", "weights.json")) as f:
        weights = json.load(f)["weights"]
    with open(os.path.join(REPO_ROOT, "data", "eval_data.json")) as f:
        eval_examples = json.load(f)["examples"]

    results = []

    # 1. baseline
    results.append(comp.benchmark_model("baseline_fp32", weights, eval_examples, dtype_bytes=4))

    # 2 & 3. pruned, measured two ways
    pruned = comp.prune_weights(weights, PRUNE_SPARSITY)
    n_nonzero = sum(1 for w in pruned if w != 0.0)

    pruned_dense = comp.benchmark_model("pruned_dense_fp32", pruned, eval_examples, dtype_bytes=4)
    results.append(pruned_dense)

    pruned_sparse = dict(pruned_dense)
    pruned_sparse["label"] = "pruned_sparse_coo"
    # COO storage: each non-zero value pays for itself (4 bytes) plus an
    # index (4 bytes) instead of every one of the original 256 slots.
    pruned_sparse["size_bytes"] = comp.sparse_coo_size_bytes(pruned)
    results.append(pruned_sparse)

    # 4. quantized
    quantized, scale = comp.quantize_int8(weights)
    dequantized = comp.dequantize_int8(quantized, scale)
    quantized_result = comp.benchmark_model(
        "quantized_int8", dequantized, eval_examples, dtype_bytes=1
    )
    results.append(quantized_result)

    baseline_size = results[0]["size_bytes"]
    for r in results:
        r["compression_ratio_vs_baseline"] = round(baseline_size / r["size_bytes"], 2)

    report = {
        "prune_sparsity": PRUNE_SPARSITY,
        "n_weights": len(weights),
        "n_nonzero_after_pruning": n_nonzero,
        "results": results,
    }

    out_path = os.path.join(REPO_ROOT, "compression_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2) 

    print(f"{'label':22s} {'size_bytes':>10s} {'ratio':>7s} {'accuracy':>9s} {'latency_ms':>11s}")
    for r in results:
        print(
            f"{r['label']:22s} {r['size_bytes']:>10d} "
            f"{r['compression_ratio_vs_baseline']:>6.2f}x {r['accuracy']:>9.2%} "
            f"{r['latency_ms_per_pass']:>11.4f}"
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main() 
