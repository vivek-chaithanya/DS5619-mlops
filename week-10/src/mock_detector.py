"""
Shared Track B mock detector — DS5619 MLOps labs (Weeks 6-11).

This is a deliberately trivial, dependency-light stand-in for a real
object detector (YOLOv12 / RT-DETRv2 / D-FINE, as released alongside
BMD-45 at https://huggingface.co/iisc-aim/BMD-45, Apache 2.0).

WHY THIS EXISTS: torch + ultralytics/mmdetection + a GPU is a heavy,
slow, sometimes-flaky dependency chain to require just to prove a
Docker build, a CI pipeline, or a drift-monitoring script works. This
module lets every lab's smoke_test.py run in seconds on a laptop or a
GitHub Actions runner with zero GPU, so you can verify your PIPELINE is
correct before pointing it at the real model.

FOR THE GRADED SUBMISSION: swap this out for a real checkpoint. Every
lab's README.md tells you exactly where that swap happens (usually
one function, `load_model()` / `predict()`, behind the same interface
used here). Grading checks that your pipeline logic is correct; it does
not require you to have trained a detector from scratch.

Detection logic: thresholds reddish pixel blobs (matching the synthetic
fixtures from generate_synthetic_fixtures.py) and returns axis-aligned
boxes around each blob with a confidence score derived from blob
"redness" and size. It will NOT produce meaningful results on real
photographs — that's expected and fine for its purpose.
"""
from dataclasses import dataclass, field


@dataclass
class Detection:
    bbox: list  # [x, y, w, h]
    category_id: int
    score: float


def _is_reddish(pixel):
    r, g, b = pixel
    return r > 120 and r - g > 40 and r - b > 40


def detect(image, score_threshold: float = 0.3):
    """Run the mock detector over a PIL Image, return a list of Detection."""
    w, h = image.size
    px = image.load()
    visited = [[False] * w for _ in range(h)]
    detections = []

    def flood(sx, sy):
        stack = [(sx, sy)]
        pts = []
        while stack:
            x, y = stack.pop()
            if x < 0 or y < 0 or x >= w or y >= h or visited[y][x]:
                continue
            if not _is_reddish(px[x, y]):
                continue
            visited[y][x] = True
            pts.append((x, y))
            stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
        return pts

    for y in range(h):
        for x in range(w):
            if visited[y][x] or not _is_reddish(px[x, y]):
                continue
            pts = flood(x, y)
            if len(pts) < 20:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bx, by = min(xs), min(ys)
            bw, bh = max(xs) - bx + 1, max(ys) - by + 1
            score = min(0.98, 0.4 + len(pts) / (bw * bh + 1))
            if score < score_threshold:
                continue
            # deterministic pseudo-class assignment from box geometry, not a real classifier
            category_id = (bw * bh) % 14
            detections.append(Detection(bbox=[bx, by, bw, bh], category_id=category_id, score=round(score, 3)))

    return detections


def detections_to_coco(detections, image_id, ann_id_start=0):
    """Serialize a list of Detection into COCO-style annotation dicts."""
    out = []
    for i, d in enumerate(detections):
        out.append({
            "id": ann_id_start + i,
            "image_id": image_id,
            "category_id": d.category_id,
            "bbox": d.bbox,
            "score": d.score,
        })
    return out
