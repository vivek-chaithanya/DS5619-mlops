#!/usr/bin/env python3
"""
Run this FIRST, before anything else in this lab:

    python generate_for_student.py --student-id <your roll number or institute email>

Generates YOUR OWN copy of data/fixtures/ (3 synthetic images per camera +
_annotations.coco.json) — same structure as everyone else's (same file
names, same two camera profiles, always at least one detectable "vehicle"
per image), but different actual pixel content, seeded deterministically
from your student ID.

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
from generate_synthetic_fixtures import CLASSES, CAMERA_PROFILES, draw_scene  # noqa: E402

N_PER_CAMERA = 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-id", required=True, help="Your roll number or institute email")
    args = ap.parse_args()

    seed = seed_from_student_id(args.student_id, salt="week07")
    rng = random.Random(seed)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(repo_root, "data", "fixtures")
    os.makedirs(out_dir, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": c} for i, c in enumerate(CLASSES)],
    }
    img_id, ann_id = 0, 0

    for cam_name, profile in CAMERA_PROFILES.items():
        cam_dir = os.path.join(out_dir, cam_name)
        os.makedirs(cam_dir, exist_ok=True)
        for i in range(N_PER_CAMERA):
            img, boxes, cats = draw_scene(rng, profile)
            fname = f"{i:03d}.jpg"
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
    print(f"seed: {seed}")
    print(f"Wrote {img_id} images across {len(CAMERA_PROFILES)} camera profiles -> {out_dir}")
    print(f"Wrote {ann_id} annotations -> {out_dir}/_annotations.coco.json")
    print("\nRecord this seed in NOTES.md when you submit (see README.md).")


if __name__ == "__main__":
    main()
