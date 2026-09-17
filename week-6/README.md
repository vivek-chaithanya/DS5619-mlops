# Week 6: Containerize and Serve a Detector

**Student ID:** 142301024  
**Seed:** 1265235759  
**Course:** DS5619 MLOps (Weeks 6-11 Track B)

---

## Overview

This lab implements a minimal inference API wrapping a mock vehicle detector, designed to be containerized with Docker. The API provides two endpoints:

- `GET /health` — Liveness check
- `POST /detect` — Accepts an uploaded image and returns vehicle detections in COCO format

The mock detector (`src/mock_detector.py`) stands in for a real checkpoint (YOLOv12-S / RT-DETRv2 / D-FINE) so the lab can run without GPU or the 153GB BMD-45 dataset.

---

## Algorithm & Logic

### 1. Mock Detector (`src/mock_detector.py`)

The mock detector uses a **flood-fill (connected component) algorithm** to find reddish pixel blobs in synthetic images, simulating vehicle detection.

#### Detection Pipeline

```
Input: PIL Image (RGB)
         │
         ▼
Convert to pixel access object
         │
         ▼
Scan every pixel (y=0..h, x=0..w)
         │
         ├──▶ If already visited OR not reddish → skip
         │
         ▼
Flood-fill from seed pixel (DFS with explicit stack)
         │
         ├──▶ Collect all connected reddish pixels
         │
         ▼
Filter by minimum blob size (≥ 20 pixels)
         │
         ▼
Compute bounding box: [min_x, min_y, width, height]
         │
         ▼
Compute confidence score: min(0.98, 0.4 + pixels / (box_area + 1))
         │
         ├──▶ Score < threshold (0.3) → discard
         │
         ▼
Assign pseudo-category_id: (box_width × box_height) % 14
         │
         ▼
Return list of Detection objects
```

#### Key Functions

**`_is_reddish(pixel)`** — Heuristic to identify "vehicle" pixels in synthetic fixtures:
```python
def _is_reddish(pixel):
    r, g, b = pixel
    return r > 120 and r - g > 40 and r - b > 40
```
This detects pixels where red channel dominates (synthetic vehicles are drawn as red rectangles).

**`flood(sx, sy)`** — Iterative DFS flood-fill:
- Uses explicit stack to avoid recursion limits
- 4-connected neighborhood (up, down, left, right)
- Marks visited pixels to avoid reprocessing
- Returns list of all connected reddish pixel coordinates

**`detect(image, score_threshold=0.3)`** — Main detection entry point:
- Scans image row by row
- Triggers flood-fill on unvisited reddish pixels
- Filters tiny noise blobs (< 20 pixels)
- Computes axis-aligned bounding box
- Derives confidence from blob density (pixels / box_area)
- Assigns deterministic pseudo-class from box geometry

**`detections_to_coco(detections, image_id, ann_id_start=0)`** — Serializes to COCO format:
```json
{
  "id": ann_id,
  "image_id": image_id,
  "category_id": 0-13,
  "bbox": [x, y, width, height],
  "score": 0.0-1.0
}
```

---

### 2. Flask API (`src/app.py`)

Three core functions were implemented:

#### `load_image_from_upload(file_storage)`
```python
def load_image_from_upload(file_storage):
    image_bytes = file_storage.read()
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")
```
- Reads raw bytes from Werkzeug FileStorage
- Wraps in BytesIO for PIL compatibility
- Converts to RGB (handles RGBA, palette mode, etc.)

#### `run_detection(image)`
```python
def run_detection(image):
    detections = det.detect(image)
    coco_detections = det.detections_to_coco(detections, image_id=0)
    return {"count": len(coco_detections), "detections": coco_detections}
```
- Delegates to mock detector
- Serializes to COCO format with image_id=0
- Returns count + detections list

#### `POST /detect` endpoint
```python
@app.post("/detect")
def detect():
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' file field"}), 400

    file_storage = request.files["image"]
    image = load_image_from_upload(file_storage)
    result = run_detection(image)
    return jsonify(result)
```
- Validates presence of "image" form field
- Returns 400 with error message if missing
- Processes image through pipeline
- Returns JSON response with 200 status

---

### 3. Synthetic Fixture Generation (`generate_for_student.py`)

Generates personalized test images seeded from student ID:

```
python generate_for_student.py --student-id 142301024
```

**Process:**
1. Derives deterministic seed from student ID + salt "week06"
2. Creates 2 camera profiles (`camera_A_daylight`, `camera_B_night`)
3. Generates 3 images per camera (6 total)
4. Each image: random scene with 1+ synthetic "vehicles" (red rectangles)
5. Outputs:
   - `data/fixtures/<camera>/000.jpg`, `001.jpg`, `002.jpg`
   - `data/fixtures/_annotations.coco.json` (ground truth)

**Seed for 142301024:** `1265235759`  
**Images generated:** 6 (3 per camera)  
**Annotations:** 23 vehicles across all images

---

## Test Results

All smoke tests pass:

```
tests/test_smoke.py::test_health_endpoint PASSED
tests/test_smoke.py::test_detect_missing_image_returns_400 PASSED
tests/test_smoke.py::test_detect_returns_detections_for_fixture_image PASSED
tests/test_smoke.py::test_run_detection_shape_directly PASSED

============================== 4 passed in 2.10s ===============================
```

### Test Coverage

| Test | Purpose |
|------|---------|
| `test_health_endpoint` | Verifies `/health` returns `{"status": "ok"}` |
| `test_detect_missing_image_returns_400` | Validates 400 error when no image field |
| `test_detect_returns_detections_for_fixture_image` | End-to-end test with real fixture image |
| `test_run_detection_shape_directly` | Unit test for `run_detection()` function |

### Sample Detection Output

For a fixture image, `/detect` returns:
```json
{
  "count": 4,
  "detections": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 7,
      "bbox": [123, 45, 67, 89],
      "score": 0.723
    },
    {
      "id": 1,
      "image_id": 0,
      "category_id": 2,
      "bbox": [200, 150, 45, 32],
      "score": 0.611
    },
    ...
  ]
}
```

---

## Docker Support

The repository includes a `Dockerfile` for containerization:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
EXPOSE 8080
CMD ["python", "src/app.py"]
```

To build and run:
```bash
docker build -t week6-detector .
docker run -p 8080:8080 week6-detector
```

Test the running container:
```bash
curl http://localhost:8080/health
curl -F "image=@data/fixtures/camera_A_daylight/000.jpg" http://localhost:8080/detect
```

---

## Swapping for a Real Model

For production deployment, replace `src/mock_detector.py` with a real detector:

1. **Keep the same interface** — `detect(image)` returning `List[Detection]`
2. **Add model loading** — `load_model()` called once at startup
3. **Update Dockerfile** — Add torch, ultralytics/mmdetection dependencies
4. **GPU support** — Use `nvidia/cuda` base image, `--gpus all` at runtime

**Biggest Dockerfile change:** Switching from `python:3.12-slim` to `nvidia/cuda:12.x-runtime-ubuntu22.04` and installing PyTorch + detection framework, increasing image size from ~200MB to 5-10GB+ and build time from seconds to minutes.

---

## Files Modified

| File | Changes |
|------|---------|
| `src/app.py` | Implemented `load_image_from_upload()`, `run_detection()`, `POST /detect` |
| `NOTES.md` | Recorded student ID: 142301024 |
| `README.md` | This documentation (new file) |

---

## Generated Artifacts

```
week-6/
├── data/fixtures/
│   ├── camera_A_daylight/000.jpg, 001.jpg, 002.jpg
│   ├── camera_B_night/000.jpg, 001.jpg, 002.jpg
│   └── _annotations.coco.json
├── src/
│   ├── app.py          # Flask API (implemented)
│   └── mock_detector.py # Mock detector (provided)
├── tests/test_smoke.py  # All 4 tests passing
├── requirements.txt
├── Dockerfile
└── NOTES.md
```

---

*Generated for DS5619 MLOps Week 6 lab — Student ID: 142301024*