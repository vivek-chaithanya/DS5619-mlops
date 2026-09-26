#!/usr/bin/env bash
# Week 7 integration test: build the Docker image, run it, hit both
# endpoints over HTTP, then tear it down. This is what the "integration-test"
# CI job runs — it's testing the CONTAINER, not the Python code directly
# (that's what tests/test_smoke.py + the "unit-test" job already cover).
#
# Fill in each TODO. Exits non-zero (via `set -e`) on any failure, which is
# what makes this useful as a CI gate.
set -euo pipefail

IMAGE_NAME="week7-detector"
CONTAINER_NAME="week7-detector-ci"
PORT="8080"

cleanup() {
  echo "Tearing down container (if running)..."
  docker rm -f "$CONTAINER_NAME" > /dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Building image..."
# TODO: docker build, tag it $IMAGE_NAME, build context is the repo root (.)
docker build -t "$IMAGE_NAME" .

echo "Starting container..."
# TODO: docker run, detached (-d), map host port $PORT to container port 8080,
# name it $CONTAINER_NAME, remove automatically on stop (--rm), image $IMAGE_NAME
docker run -d -p "$PORT:8080" --name "$CONTAINER_NAME" --rm "$IMAGE_NAME"

echo "Waiting for /health to respond..."
ready=0
for i in $(seq 1 20); do
  if curl -sf "http://localhost:${PORT}/health" > /dev/null; then
    ready=1
    break
  fi
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  echo "Container never became healthy" >&2
  docker logs "$CONTAINER_NAME" || true
  exit 1
fi
echo "Health check passed."

echo "Testing /detect with a sample fixture image..."
# TODO: curl -sf -F "image=@data/fixtures/camera_A_daylight/000.jpg" against
# http://localhost:${PORT}/detect, save the response body to a variable, and
# fail (exit 1) if it doesn't contain the string "detections"
response=$(curl -sf -F "image=@data/fixtures/camera_A_daylight/000.jpg" "http://localhost:${PORT}/detect")
if ! echo "$response" | grep -q "detections"; then
  echo "Unexpected /detect response: $response" >&2
  exit 1
fi
echo "/detect check passed."

echo "Integration test passed."
