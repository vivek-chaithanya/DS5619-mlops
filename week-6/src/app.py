"""
Minimal inference API wrapping the (mock) vehicle detector, ready to be
containerized. This is Track B's first lab (Weeks 6-11, built around the
BMD-45 CCTV vehicle-detection dataset).

Endpoints:
  GET  /health   - liveness check, given.
  POST /detect   - accepts an uploaded image under form field "image",
                   returns JSON detections. Marked # TODO.

`mock_detector.py` (in this same directory, complete, don't edit) stands in
for a real checkpoint (YOLOv12-S etc., released alongside BMD-45) so this
lab and its container build don't require torch/a GPU/a 153GB dataset
download in a 110-minute session. For the graded submission in a real
deployment, `run_detection()` is the one function you'd point at a real
model instead — everything else (the API, the Dockerfile) stays the same.
"""
import io
import os
import sys

from flask import Flask, jsonify, request
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mock_detector as det


def load_image_from_upload(file_storage):
    """`file_storage` is a werkzeug FileStorage (what request.files["image"]
    gives you). Read its bytes and return a PIL Image opened in RGB mode.

    Hint: file_storage.read() gives raw bytes; wrap them in io.BytesIO(...)
    and pass that to Image.open(...); call .convert("RGB") on the result
    (uploaded images may be non-RGB, e.g. RGBA or palette mode, and the
    detector expects RGB pixel tuples).
    """
    image_bytes = file_storage.read()
    image = Image.open(io.BytesIO(image_bytes))
    return image.convert("RGB")


def run_detection(image):
    """Run the detector on a PIL Image and return a JSON-serializable dict:

      {"count": <int>, "detections": [<coco-style annotation dict>, ...]}

    Use det.detect(image) to get a list of Detection objects, then
    det.detections_to_coco(detections, image_id=0) to serialize them.
    (image_id=0 is fine — this endpoint handles one image per request, it
    doesn't need a real dataset-wide id.)
    """
    detections = det.detect(image)
    coco_detections = det.detections_to_coco(detections, image_id=0)
    return {"count": len(coco_detections), "detections": coco_detections}


def create_app():
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/detect")
    def detect():
        """TODO: wire this up.

        - If request.files does not contain a file under the key "image",
          return (jsonify({"error": "missing 'image' file field"}), 400).
        - Otherwise, load the image with load_image_from_upload(...), run
          run_detection(...) on it, and return jsonify(<that result>) with
          the default 200 status.
        """
        if "image" not in request.files:
            return jsonify({"error": "missing 'image' file field"}), 400

        file_storage = request.files["image"]
        image = load_image_from_upload(file_storage)
        result = run_detection(image)
        return jsonify(result)

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
