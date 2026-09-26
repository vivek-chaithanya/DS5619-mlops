"""
Unit tests for the (already-complete) inference API — this is exactly what
the "unit-test" CI job should run. No TODOs here; if these fail, something
is wrong with your environment, not your CI config.
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


def test_detect_returns_detections_for_fixture_image():
    with open(FIXTURE_IMAGE, "rb") as f:
        data = {"image": (io.BytesIO(f.read()), "000.jpg")}
        resp = _client().post("/detect", data=data, content_type="multipart/form-data")

    assert resp.status_code == 200
    body = resp.get_json()
    assert "detections" in body
    assert body["count"] == len(body["detections"])
