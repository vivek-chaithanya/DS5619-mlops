"""
Inference API wrapping the (mock) vehicle detector. This is the SAME app
from Week 6 — already complete here, because this week's lab is about CI/CD,
not re-implementing it. Don't edit this file; your work this week is the
GitHub Actions workflow and the integration test script that exercise it.
"""
import io
import os
import sys

from flask import Flask, jsonify, request
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mock_detector as det


def load_image_from_upload(file_storage):
    data = file_storage.read()
    return Image.open(io.BytesIO(data)).convert("RGB")


def run_detection(image):
    detections = det.detect(image)
    coco = det.detections_to_coco(detections, image_id=0)
    return {"count": len(coco), "detections": coco}


def create_app():
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/detect")
    def detect():
        if "image" not in request.files:
            return jsonify({"error": "missing 'image' file field"}), 400
        image = load_image_from_upload(request.files["image"])
        result = run_detection(image)
        return jsonify(result)

    return app


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
