#!/usr/bin/env python3
"""
Run this FIRST, before anything else in this lab:

    python generate_for_student.py --student-id <your roll number or institute email>

Generates YOUR OWN copy of data/fixtures/ (20 synthetic images per camera +
_annotations.coco.json) — same structure as everyone else's (same file
names, same two camera profiles with deliberately different visual
statistics), but different actual pixel content, seeded deterministically
from your student ID.

The camera_A vs camera_B drift signal this lab is built around comes from
the camera PROFILES (brightness/noise/box-size), which are fixed — but with
only 20 images per camera, a purely random draw can occasionally land two
samples close enough together that PSI comes out near zero, which would
defeat the point of the lab. So this script checks the PSI of its own
candidate output (using the mock detector directly, not your
drift_monitor.py) and retries with a new sub-seed until the drift is at
least "moderate" — same idea as Week 5's f1-threshold guard, applied here to
keep the pedagogical signal intact under reseeding.

Record your --student-id in NOTES.md when you submit — the grader
regenerates data/ from it and diffs against what you committed.
"""
import argparse
import io
import json
import math
import os
import random
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_shared"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from student_seed import seed_from_student_id  # noqa: E402
from generate_synthetic_fixtures import CLASSES, CAMERA_PROFILES, draw_scene  # noqa: E402
import mock_detector as det  # noqa: E402

N_PER_CAMERA = 20
N_BINS = 10
PSI_MODERATE_THRESHOLD = 0.10
MAX_ATTEMPTS = 200


def _psi(reference_scores, live_scores, n_bins=N_BINS):
    def _bin_index(score):
        idx = int(score * n_bins)
        return min(max(idx, 0), n_bins - 1)

    ref_counts = [0] * n_bins
    for s in reference_scores:
        ref_counts[_bin_index(s)] += 1
    live_counts = [0] * n_bins
    for s in live_scores:
        live_counts[_bin_index(s)] += 1

    ref_total = len(reference_scores) or 1
    live_total = len(live_scores) or 1

    psi = 0.0
    for i in range(n_bins):
        ref_pct = max(ref_counts[i] / ref_total, 1e-4)
        live_pct = max(live_counts[i] / live_total, 1e-4)
        psi += (live_pct - ref_pct) * math.log(live_pct / ref_pct)
    return psi


def _generate_candidate(attempt_seed):
    rng = random.Random(attempt_seed)
    per_camera = {}
    for cam_name, profile in CAMERA_PROFILES.items():
        images = []
        for i in range(N_PER_CAMERA):
            img, boxes, cats = draw_scene(rng, profile)
            images.append((f"{i:03d}.jpg", img, boxes, cats))
        per_camera[cam_name] = images
    return per_camera


def _confidence_scores(images):
    """Score images after a JPEG round-trip, matching what
    extract_confidence_scores(camera_dir) sees when it loads the actual
    saved files from disk (JPEG compression shifts pixel values enough to
    change which pixels the mock detector's reddish-blob threshold picks
    up, so scoring the in-memory PIL image directly would silently
    disagree with the real report)."""
    scores = []
    for _, img, _, _ in images:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        reloaded = Image.open(buf).convert("RGB")
        for d in det.detect(reloaded):
            scores.append(d.score)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True, help="Your roll number or institute email")
    args = ap.parse_args()

    base_seed = seed_from_student_id(args.student_id, salt="week08")

    cam_names = list(CAMERA_PROFILES.keys())
    reference_cam, live_cam = cam_names[0], cam_names[1]

    chosen = None
    for attempt in range(MAX_ATTEMPTS):
        attempt_seed = base_seed + attempt
        per_camera = _generate_candidate(attempt_seed)
        ref_scores = _confidence_scores(per_camera[reference_cam])
        live_scores = _confidence_scores(per_camera[live_cam])
        if not ref_scores or not live_scores:
            continue
        psi = _psi(ref_scores, live_scores)
        if psi >= PSI_MODERATE_THRESHOLD:
            chosen = (attempt, per_camera, psi)
            break

    if chosen is None:
        raise SystemExit(
            f"Could not find a seed producing a moderate-or-greater PSI drift "
            f"signal in {MAX_ATTEMPTS} attempts for student_id={args.student_id!r}. "
            f"This should not happen — please report it."
        )

    attempt, per_camera, psi = chosen

    repo_root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(repo_root, "data", "fixtures")
    os.makedirs(out_dir, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": c} for i, c in enumerate(CLASSES)],
    }
    img_id, ann_id = 0, 0

    for cam_name in cam_names:
        cam_dir = os.path.join(out_dir, cam_name)
        os.makedirs(cam_dir, exist_ok=True)
        for fname, img, boxes, cats in per_camera[cam_name]:
            img.save(os.path.join(cam_dir, fname))
            coco["images"].append({
                "id": img_id, "file_name": f"{cam_name}/{fname}", "width": img.width, "height": img.height,
                "camera": cam_name,
            })
            for box, cat in zip(boxes, cats):
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id, "category_id": cat,
                    "bbox": box, "area": box[2] * box[3],
                })
                ann_id += 1
            img_id += 1

    with open(os.path.join(out_dir, "_annotations.coco.json"), "w") as f:
        json.dump(coco, f, indent=2)

    print(f"student_id: {args.student_id}")
    print(f"seed: {base_seed}")
    print(f"Wrote {img_id} images across {len(CAMERA_PROFILES)} camera profiles -> {out_dir}")
    print(f"Wrote {ann_id} annotations -> {out_dir}/_annotations.coco.json")
    print(f"Sanity-checked PSI (using the mock detector, not your drift_monitor.py): {psi:.4f}")
    print("\nRecord this seed in NOTES.md when you submit (see README.md).")


if __name__ == "__main__":
    main()
