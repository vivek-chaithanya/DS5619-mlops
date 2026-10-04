#!/usr/bin/env python3
"""
Shared Track B offline test fixtures — DS5619 MLOps labs (Weeks 6-11).

Generates a handful of tiny SYNTHETIC images with drawn rectangles standing
in for vehicles, plus a matching COCO-format annotation file. These are NOT
BMD-45 data — they exist only so smoke_test.py in each starter repo can run
fast, offline, without a GPU, and without the 153GB dataset, on any laptop
or CI runner.

The real graded lab work runs against fetch_bmd45_subset.py output instead.

Two "camera" groups are generated with deliberately different visual
statistics (brightness/contrast/box-size distribution) to stand in for the
paper's real cross-camera domain-shift finding, which the Week 8 drift lab
is built around.

Usage:
    python generate_synthetic_fixtures.py --out ./fixtures --n-per-camera 12
"""
import argparse
import json
import os
import random

from PIL import Image, ImageDraw

CLASSES = [
    "Hatchback", "Sedan", "SUV", "MUV", "Bus", "Truck", "Three-wheeler",
    "Two-wheeler", "LCV", "Mini-bus", "Tempo-traveller", "Bicycle", "Van", "Other",
]

CAMERA_PROFILES = {
    "camera_A_daylight": {"bg": (170, 175, 180), "noise": 8, "box_scale": 1.0},
    "camera_B_lowlight": {"bg": (60, 62, 70), "noise": 22, "box_scale": 0.7},
}


def draw_scene(rng, profile, w=320, h=180):
    img = Image.new("RGB", (w, h), profile["bg"])
    draw = ImageDraw.Draw(img)
    n_boxes = rng.randint(2, 6)
    boxes, cats = [], []
    for _ in range(n_boxes):
        bw = int(rng.randint(18, 46) * profile["box_scale"])
        bh = int(rng.randint(12, 30) * profile["box_scale"])
        x = rng.randint(0, max(1, w - bw))
        y = rng.randint(0, max(1, h - bh))
        color = tuple(max(0, min(255, c + rng.randint(-60, 60))) for c in (200, 60, 60))
        draw.rectangle([x, y, x + bw, y + bh], fill=color, outline=(0, 0, 0))
        boxes.append([x, y, bw, bh])
        cats.append(rng.randrange(len(CLASSES)))
    # simple per-pixel noise to differentiate camera conditions
    px = img.load()
    for _ in range(profile["noise"] * 50):
        rx, ry = rng.randint(0, w - 1), rng.randint(0, h - 1)
        r, g, b = px[rx, ry]
        d = rng.randint(-profile["noise"], profile["noise"])
        px[rx, ry] = (max(0, min(255, r + d)), max(0, min(255, g + d)), max(0, min(255, b + d)))
    return img, boxes, cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./fixtures")
    ap.add_argument("--n-per-camera", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7619)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": c} for i, c in enumerate(CLASSES)],
    }
    img_id, ann_id = 0, 0

    for cam_name, profile in CAMERA_PROFILES.items():
        cam_dir = os.path.join(args.out, cam_name)
        os.makedirs(cam_dir, exist_ok=True)
        for i in range(args.n_per_camera):
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

    with open(os.path.join(args.out, "_annotations.coco.json"), "w") as f:
        json.dump(coco, f, indent=2)

    print(f"Wrote {img_id} synthetic images across {len(CAMERA_PROFILES)} camera profiles -> {args.out}")
    print(f"Wrote {ann_id} synthetic annotations -> {args.out}/_annotations.coco.json")
    print("\nThese are SYNTHETIC fixtures for offline smoke-testing only — not BMD-45 data.")


if __name__ == "__main__":
    main()
