"""
A tiny, dependency-free model-compression benchmark — implements the two
techniques from this week's lecture (magnitude-based pruning, INT8
quantization) against a real (if small) set of weights, and measures the
three axes the lecture's Pareto-frontier framing cares about: size,
latency, and accuracy.

`data/weights.json` holds 256 weights for a linear classifier (a stand-in
for a detector's classification head) — deliberately over-parameterized:
only 40 of the 256 are actually informative, the rest are near-zero noise.
`data/eval_data.json` holds 300 labeled examples to measure accuracy
against.

Fill in the five functions marked # TODO. `dequantize_int8` and `predict`
are given.
"""
import time

# ---------------------------------------------------------------------------
# Given: prediction and dequantization
# ---------------------------------------------------------------------------


def predict(weights, features):
    """A trivial linear classifier: predicted label is 1 if the dot product
    of weights and features is positive, else 0. Given — used both to
    measure accuracy and to time inference."""
    score = sum(w * x for w, x in zip(weights, features))
    return 1 if score > 0 else 0


def dequantize_int8(quantized_weights, scale):
    """Reverse of quantize_int8: multiply each int back by `scale` to get
    an approximate float. Given."""
    return [q * scale for q in quantized_weights]


# ---------------------------------------------------------------------------
# Part 1 — Model size in bytes, given a per-weight byte width
# ---------------------------------------------------------------------------

def model_size_bytes(weights, dtype_bytes):
    """Return the storage size in bytes if every value in `weights` were
    stored using `dtype_bytes` bytes each (4 for FP32, 2 for FP16, 1 for
    INT8 — this week's "bits per parameter" framing, applied literally).
    """
    return len(weights) * dtype_bytes


# ---------------------------------------------------------------------------
# Part 2 — Magnitude-based unstructured pruning
# ---------------------------------------------------------------------------

def prune_weights(weights, sparsity):
    """Return a NEW list, same length as `weights`, where the fraction
    `sparsity` (e.g. 0.7 = 70%) of weights with the SMALLEST absolute value
    are set to 0.0, and the rest are unchanged. This is magnitude-based
    pruning — "removing weights that don't earn their byte," per the
    lecture.

    Steps:
      1. Compute how many weights to zero out: n_prune = int(sparsity * len(weights)).
      2. Find the indices of the n_prune weights with the smallest abs(weight).
      3. Return a copy of `weights` with those indices set to 0.0 (don't
         mutate the input list).
    """
    n_weights = len(weights)
    n_prune = int(sparsity * n_weights)
    if n_prune <= 0:
        return list(weights)
    if n_prune >= n_weights:
        return [0.0] * n_weights

    # Find indices of the n_prune weights with smallest absolute value
    sorted_indices = sorted(range(n_weights), key=lambda i: abs(weights[i]))
    prune_set = set(sorted_indices[:n_prune])

    return [0.0 if i in prune_set else weights[i] for i in range(n_weights)]


# ---------------------------------------------------------------------------
# Part 2b — Storage size of a SPARSE (COO-style) copy of the weights
# ---------------------------------------------------------------------------

def sparse_coo_size_bytes(weights, value_bytes=4, index_bytes=4):
    """Storage size in bytes if only the NON-ZERO weights were stored in a
    COO-style sparse format: every surviving weight pays for its value
    (value_bytes) AND its position (index_bytes).

    Return (number of entries where w != 0.0) * (value_bytes + index_bytes).
    All-zero (or empty) weights therefore cost 0 bytes.

    This is the lecture's pruning pitfall in one line: zeros stored densely
    cost the same as any other weight, and a sparse copy saves space only
    while enough weights are zero to pay for the extra index on every
    survivor.
    """
    n_nonzero = sum(1 for w in weights if w != 0.0)
    return n_nonzero * (value_bytes + index_bytes)


# ---------------------------------------------------------------------------
# Part 3 — Symmetric INT8 quantization
# ---------------------------------------------------------------------------

def quantize_int8(weights):
    """Symmetric linear quantization to signed 8-bit integers.

    Steps:
      1. scale = max(abs(w) for w in weights) / 127
      2. For each weight, compute round(w / scale), then CLAMP it to the
         range [-127, 127] (protects against a value that rounds just past
         the boundary due to floating point).
      3. Return (quantized_weights, scale) — quantized_weights is a list of
         ints, scale is the float you computed in step 1.

    Edge case: if every weight is 0.0, step 1 would divide by zero. In that
    case use scale = 1.0 and return all-zero integers.

    Note: with scale = max|w| / 127, no weight can round past +/-127, so
    the clamp in step 2 is only a floating-point safety net. What you can
    check is that the largest-magnitude weight maps to exactly +/-127.

    (dequantize_int8, given above, reverses this: w_approx = q * scale.)
    """
    if not weights:
        return ([], 1.0)

    max_abs = max(abs(w) for w in weights)
    if max_abs == 0.0:
        return ([0] * len(weights), 1.0)

    scale = max_abs / 127.0
    quantized_weights = []
    for w in weights:
        q = round(w / scale)
        q_clamped = max(-127, min(127, int(q)))
        quantized_weights.append(q_clamped)

    return (quantized_weights, scale)


# ---------------------------------------------------------------------------
# Part 4 — Benchmark: size, latency, accuracy for one weight set
# ---------------------------------------------------------------------------

def benchmark_model(label, weights, eval_examples, dtype_bytes, n_latency_reps=50):
    """Benchmark one weight set against the eval set.

    Return a dict:
      {
        "label": label,
        "size_bytes": model_size_bytes(weights, dtype_bytes),
        "accuracy": <fraction of eval_examples where predict(weights,
                     example["features"]) == example["label"]>, rounded to 4 dp,
        "latency_ms_per_pass": <average wall-clock milliseconds to run
                     predict() once over every example in eval_examples,
                     averaged over n_latency_reps repetitions>, rounded to 4 dp,
      }

    For latency: time n_latency_reps full passes over eval_examples (using
    time.perf_counter() before/after each pass), sum the per-pass times,
    divide by n_latency_reps, and convert seconds -> milliseconds.
    """
    size_bytes = model_size_bytes(weights, dtype_bytes)

    if eval_examples:
        correct = sum(
            1 for ex in eval_examples if predict(weights, ex["features"]) == ex["label"]
        )
        accuracy = round(correct / len(eval_examples), 4)
    else:
        accuracy = 0.0

    total_time_sec = 0.0
    for _ in range(n_latency_reps):
        t0 = time.perf_counter()
        for ex in eval_examples:
            predict(weights, ex["features"])
        t1 = time.perf_counter()
        total_time_sec += (t1 - t0)

    avg_ms = (total_time_sec / n_latency_reps * 1000.0) if n_latency_reps > 0 else 0.0
    latency_ms_per_pass = round(avg_ms, 4)

    return {
        "label": label,
        "size_bytes": size_bytes,
        "accuracy": accuracy,
        "latency_ms_per_pass": latency_ms_per_pass,
    }
