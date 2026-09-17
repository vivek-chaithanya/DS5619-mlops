# Docker verification

Fill this in after you build and run your container (see README.md,
"Part 2 — Dockerfile"). This is how we confirm your container actually works, since an
automated grader running in a sandbox may not always have Docker-in-Docker
available.

## Build

Paste the command you ran and its final output line (the one showing the
built image ID/tag):

```
TODO
```

## Run

Paste the command you used to start the container (should map a host port
to the container's 8080):

```
TODO
```

## Verify

Paste the exact `curl` commands and their JSON output for both endpoints,
run against the running container (not against `python src/app.py` directly
— the point is to prove the *container* works):

```
TODO: curl http://localhost:PORT/health
TODO: <paste JSON response>

TODO: curl -F "image=@data/fixtures/camera_A_daylight/000.jpg" http://localhost:PORT/detect
TODO: <paste JSON response>
```
