"""
Self-check for the Week 6 lab. Not the grader — see README.md.

This tests the Flask app directly (via its test client), which is enough to
verify your API logic without needing Docker in this test environment. The
Docker build/run itself is verified separately — see DOCKER_VERIFICATION.md.

Run with: pytest tests/ -q
"""
import io
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
import app as app_module  # noqa: E402

FIXTURE_IMAGE = os.path.join(REPO_ROOT, "data", "fixtures", "camera_A_daylight", "000.jpg")


def _client():
    return app_module.create_app().test_client()


def test_health_endpoint():
    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_detect_missing_image_returns_400():
    resp = _client().post("/detect", data={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_detect_returns_detections_for_fixture_image():
    with open(FIXTURE_IMAGE, "rb") as f:
        data = {"image": (io.BytesIO(f.read()), "000.jpg")}
        resp = _client().post("/detect", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    body = resp.get_json()
    assert "count" in body and "detections" in body
    assert body["count"] == len(body["detections"])
    assert body["count"] > 0  # the synthetic fixture has drawn "vehicles" on it

    for d in body["detections"]:
        assert set(["bbox", "category_id", "score", "image_id", "id"]).issubset(d.keys())
        assert len(d["bbox"]) == 4
        assert 0.0 <= d["score"] <= 1.0


def test_run_detection_shape_directly():
    from PIL import Image
    image = Image.open(FIXTURE_IMAGE).convert("RGB")
    result = app_module.run_detection(image)
    assert isinstance(result["detections"], list)
    assert result["count"] == len(result["detections"])
