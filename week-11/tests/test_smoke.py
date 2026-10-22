"""
Self-check for the Week 11 lab. Not the grader — see README.md.

Run with: pytest tests/ -q
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import compression as comp  # noqa: E402


def test_model_size_bytes():
    assert comp.model_size_bytes([1.0, 2.0, 3.0], dtype_bytes=4) == 12
    assert comp.model_size_bytes([1.0, 2.0, 3.0], dtype_bytes=1) == 3
    assert comp.model_size_bytes([], dtype_bytes=4) == 0


def test_prune_weights_zeroes_smallest_magnitude():
    weights = [0.01, -5.0, 0.02, 3.0, -0.005, 1.0]
    pruned = comp.prune_weights(weights, sparsity=0.5)  # zero out 3 smallest
    zeroed = [i for i, w in enumerate(pruned) if w == 0.0]
    assert set(zeroed) == {0, 2, 4}  # the three smallest-magnitude indices
    # untouched entries keep their original value
    assert pruned[1] == -5.0 and pruned[3] == 3.0 and pruned[5] == 1.0


def test_prune_weights_does_not_mutate_input():
    weights = [0.01, -5.0, 0.02, 3.0]
    original = list(weights)
    comp.prune_weights(weights, sparsity=0.5)
    assert weights == original


def test_prune_weights_full_sparsity_zeroes_everything():
    weights = [1.0, 2.0, 3.0, 4.0]
    pruned = comp.prune_weights(weights, sparsity=1.0)
    assert all(w == 0.0 for w in pruned)


def test_quantize_int8_round_trip_is_close():
    weights = [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0]
    quantized, scale = comp.quantize_int8(weights)
    assert len(quantized) == len(weights)
    assert all(isinstance(q, int) for q in quantized)
    assert all(-127 <= q <= 127 for q in quantized)

    dequantized = comp.dequantize_int8(quantized, scale)
    for original, approx in zip(weights, dequantized):
        assert abs(original - approx) < 0.05  # small quantization error is expected


def test_quantize_int8_scale_and_extremes():
    weights = [10.0, -10.0, 0.0, 5.0]
    quantized, scale = comp.quantize_int8(weights)
    assert abs(scale - 10.0 / 127) < 1e-12          # scale = max|w| / 127
    assert quantized[0] == 127 and quantized[1] == -127  # largest magnitude hits the boundary
    assert quantized[2] == 0
    assert quantized[3] == round(5.0 / scale)        # in-between values use plain rounding


def test_quantize_int8_round_trip_error_is_at_most_half_a_step():
    weights = [0.013 * i - 1.3 for i in range(201)]
    quantized, scale = comp.quantize_int8(weights)
    dequantized = comp.dequantize_int8(quantized, scale)
    for original, approx in zip(weights, dequantized):
        assert abs(original - approx) <= scale / 2 + 1e-9  # rounds to NEAREST, not floor


def test_quantize_int8_all_zero_weights_does_not_crash():
    quantized, scale = comp.quantize_int8([0.0, 0.0, 0.0])
    assert quantized == [0, 0, 0]
    assert scale == 1.0


def test_sparse_coo_size_bytes():
    assert comp.sparse_coo_size_bytes([0.0, 1.5, 0.0, -2.0]) == 2 * (4 + 4)
    assert comp.sparse_coo_size_bytes([0.0, 0.0]) == 0
    assert comp.sparse_coo_size_bytes([]) == 0
    assert comp.sparse_coo_size_bytes([1.0, 2.0, 3.0], value_bytes=2, index_bytes=1) == 9


def test_benchmark_model_shape():
    weights = [1.0, -1.0, 0.5]
    eval_examples = [
        {"features": [1.0, 0.0, 0.0], "label": 1},
        {"features": [0.0, 1.0, 0.0], "label": 0},
        {"features": [0.0, 0.0, 1.0], "label": 1},
    ]
    result = comp.benchmark_model("test", weights, eval_examples, dtype_bytes=4, n_latency_reps=3)
    assert result["label"] == "test"
    assert result["size_bytes"] == 12
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["latency_ms_per_pass"] >= 0.0


def test_benchmark_model_latency_is_measured_and_averaged():
    # Replace predict() with a ~2 ms sleep so latency has a known value.
    import time
    original_predict = comp.predict
    comp.predict = lambda weights, features: (time.sleep(0.002), 1)[1]
    try:
        examples = [{"features": [1.0], "label": 1}] * 3  # 3 calls/pass ~ 6 ms
        result = comp.benchmark_model("t", [1.0], examples, dtype_bytes=4, n_latency_reps=2)
    finally:
        comp.predict = original_predict
    # averaged over reps: ~6 ms per pass. Summed over 2 reps would be ~12 ms;
    # a hard-coded constant would not track the sleep at all.
    assert 5.0 <= result["latency_ms_per_pass"] < 10.0


def test_benchmark_model_accuracy_is_correct():
    weights = [1.0]
    eval_examples = [
        {"features": [1.0], "label": 1},
        {"features": [-1.0], "label": 0},
        {"features": [1.0], "label": 0},  # this one will be wrong
    ]
    result = comp.benchmark_model("t", weights, eval_examples, dtype_bytes=4, n_latency_reps=1)
    assert abs(result["accuracy"] - (2 / 3)) < 1e-3  # allows for round(..., 4)


def test_full_pipeline_shows_meaningful_compression_and_accuracy_tradeoff():
    with tempfile.TemporaryDirectory() as scratch:
        for sub in ("src", "data"):
            shutil.copytree(os.path.join(REPO_ROOT, sub), os.path.join(scratch, sub))

        result = subprocess.run(
            [sys.executable, os.path.join(scratch, "src", "run_pipeline.py")],
            cwd=scratch, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr

        with open(os.path.join(scratch, "compression_report.json")) as f:
            report = json.load(f)

        by_label = {r["label"]: r for r in report["results"]}

        # pruning "on paper" (dense storage) shouldn't save any bytes...
        assert by_label["pruned_dense_fp32"]["size_bytes"] == by_label["baseline_fp32"]["size_bytes"]
        # ...but the same pruning WITH sparse storage should be meaningfully smaller.
        assert by_label["pruned_sparse_coo"]["size_bytes"] < by_label["baseline_fp32"]["size_bytes"]
        # ...and equals value+index (8 bytes) per surviving weight.
        assert by_label["pruned_sparse_coo"]["size_bytes"] == 8 * report["n_nonzero_after_pruning"]
        # quantization should be exactly 4x smaller (4 bytes -> 1 byte per weight).
        assert by_label["quantized_int8"]["size_bytes"] * 4 == by_label["baseline_fp32"]["size_bytes"]
        # baseline should be near-perfect on data generated from these exact weights.
        assert by_label["baseline_fp32"]["accuracy"] > 0.95
