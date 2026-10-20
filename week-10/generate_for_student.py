#!/usr/bin/env python3
"""
Run this FIRST, before anything else in this lab:

    python generate_for_student.py --student-id <your roll number or institute email>

Generates YOUR OWN copy of data/fixtures/ (30 synthetic images per camera +
_annotations.coco.json) — same structure as everyone else's (same file
names, same two camera profiles, same per-class size multipliers), but
different actual pixel content, seeded deterministically from your student
ID.

This lab's whole point is a recall disparity between small-vehicle classes
(Bicycle, Two-wheeler) and everything else. With only 30 images per camera,
a purely random draw can occasionally land close enough that the disparity
doesn't clearly show up (or, worse, flags an unrelated large class instead).
So this script checks its own candidate output with the actual detector +
IoU-matching logic (not your fairness_audit.py) and retries with a new
sub-seed until: both Bicycle and Two-wheeler recall fall below the 90%
threshold, AND no large class (Bus, Truck, Mini-bus, Tempo-traveller, Van)
does — same idea as Week 5's f1-threshold guard and Week 8's PSI guard,
applied here to keep this week's disparity signal intact under reseeding.
Other mid-sized classes (e.g. LCV, MUV) can still fall below 90% for some IDs
because each class has only ~5-30 boxes; that is sampling noise, not a bug.

Record your --student-id in NOTES.md when you submit — the grader
regenerates data/ from it and diffs against what you committed.
"""
import argparse
import io
import json
import os
import random
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_shared"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_shared"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from student_seed import seed_from_student_id  # noqa: E402
import mock_detector as det  # noqa: E402

N_PER_CAMERA = 30
IOU_THRESHOLD = 0.3
RECALL_THRESHOLD = 0.90
MAX_ATTEMPTS = 500

CLASSES = [
    "Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler",
    "Two-wheeler", "LCV", "Mini-bus", "Tempo-traveller", "Bicycle", "Van", "Other",
]

CLASS_SIZE_MULTIPLIER = {
    "Hatchback": 1.0, "Sedan": 1.1, "SUV": 1.25, "MUV": 1.2, "Bus": 2.2,
    "Truck": 2.0, "Three-wheeler": 0.7, "Two-wheeler": 0.45, "LCV": 1.5,
    "Mini-bus": 1.8, "Tempo-traveller": 1.9, "Bicycle": 0.35, "Van": 1.3, "Other": 1.0,
}

CAMERA_PROFILES = {
    "camera_A_daylight": {"bg": (170, 175, 180), "noise": 8, "box_scale": 1.0},
    "camera_B_lowlight": {"bg": (60, 62, 70), "noise": 22, "box_scale": 0.7},
}

SMALL_CLASSES = ["Bicycle", "Two-wheeler"]
LARGE_CLASSES = ["Bus", "Truck", "Mini-bus", "Tempo-traveller", "Van"]


def draw_scene(rng, profile, w=320, h=180):
    img = Image.new("RGB", (w, h), profile["bg"])
    draw = ImageDraw.Draw(img)
    n_boxes = rng.randint(2, 6)
    boxes, cats = [], []
    for _ in range(n_boxes):
        cat = rng.randrange(len(CLASSES))
        size_mult = CLASS_SIZE_MULTIPLIER[CLASSES[cat]]
        bw = max(6, int(rng.randint(18, 46) * profile["box_scale"] * size_mult))
        bh = max(6, int(rng.randint(12, 30) * profile["box_scale"] * size_mult))
        x = rng.randint(0, max(1, w - bw))
        y = rng.randint(0, max(1, h - bh))
        color = tuple(max(0, min(255, c + rng.randint(-60, 60))) for c in (200, 60, 60))
        draw.rectangle([x, y, x + bw, y + bh], fill=color, outline=(0, 0, 0))
        boxes.append([x, y, bw, bh])
        cats.append(cat)
    px = img.load()
    for _ in range(profile["noise"] * 50):
        rx, ry = rng.randint(0, w - 1), rng.randint(0, h - 1)
        r, g, b = px[rx, ry]
        d = rng.randint(-profile["noise"], profile["noise"])
        px[rx, ry] = (max(0, min(255, r + d)), max(0, min(255, g + d)), max(0, min(255, b + d)))
    return img, boxes, cats


def _iou(box_a, box_b):
    ax1, ay1 = box_a[0], box_a[1]
    ax2, ay2 = ax1 + box_a[2], ay1 + box_a[3]
    bx1, by1 = box_b[0], box_b[1]
    bx2, by2 = bx1 + box_b[2], by1 + box_b[3]
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - intersection
    return intersection / union if union else 0.0


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


def _class_recall(per_camera):
    totals = {}
    detected = {}
    for images in per_camera.values():
        for _, img, boxes, cats in images:
            # JPEG round-trip so the sanity check sees exactly what the
            # real pipeline sees once these images are saved to disk.
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)
            reloaded = Image.open(buf).convert("RGB")
            detections = det.detect(reloaded)
            for box, cat in zip(boxes, cats):
                name = CLASSES[cat]
                totals[name] = totals.get(name, 0) + 1
                matched = any(_iou(box, d.bbox) >= IOU_THRESHOLD for d in detections)
                if matched:
                    detected[name] = detected.get(name, 0) + 1
    return {name: detected.get(name, 0) / total for name, total in totals.items()}


def _signal_is_clean(recall):
    small_present = [c for c in SMALL_CLASSES if c in recall]
    if len(small_present) < len(SMALL_CLASSES):
        return False
    if not all(recall[c] < RECALL_THRESHOLD for c in small_present):
        return False
    large_present = [c for c in LARGE_CLASSES if c in recall]
    if not all(recall[c] >= RECALL_THRESHOLD for c in large_present):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True, help="Your roll number or institute email")
    args = ap.parse_args()

    base_seed = seed_from_student_id(args.student_id, salt="week10")

    chosen = None
    for attempt in range(MAX_ATTEMPTS):
        attempt_seed = base_seed + attempt
        per_camera = _generate_candidate(attempt_seed)
        recall = _class_recall(per_camera)
        if _signal_is_clean(recall):
            chosen = (attempt, per_camera, recall)
            break

    if chosen is None:
        raise SystemExit(
            f"Could not find a seed producing a clean small-vs-large recall "
            f"disparity in {MAX_ATTEMPTS} attempts for student_id={args.student_id!r}. "
            f"This should not happen — please report it."
        )

    attempt, per_camera, recall = chosen

    repo_root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(repo_root, "data", "fixtures")
    os.makedirs(out_dir, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": c} for i, c in enumerate(CLASSES)],
    }
    img_id, ann_id = 0, 0

    for cam_name in CAMERA_PROFILES:
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
    print("Sanity-checked recall (using the mock detector, not your fairness_audit.py):")
    for name in sorted(recall, key=lambda n: recall[n]):
        flag = " <-- below threshold" if recall[name] < RECALL_THRESHOLD else ""
        print(f"  {name:20s} {recall[name]*100:5.1f}%{flag}")
    print("\nRecord this seed in NOTES.md when you submit (see README.md).")


if __name__ == "__main__":
    main()
