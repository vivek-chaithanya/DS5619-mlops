# Week 6 — containerize the inference API in src/app.py.
#
# Fill in each TODO. This should build a small, working image that serves
# the Flask app from src/app.py on the port it reads from the PORT env var
# (default 8080, see the bottom of app.py).

# TODO: pick a small official Python base image. A "slim" variant is enough
# here — we have no torch/GPU dependency, just Flask + Pillow.
FROM ???

WORKDIR /app

# TODO: copy requirements.txt into the image and install it. Use
# --no-cache-dir to keep the image smaller (pip's download cache adds
# nothing useful inside a container that's rebuilt from scratch each time).
COPY requirements.txt .
RUN ???

# TODO: copy the rest of the application source (the src/ directory) into
# the image.
COPY ??? ???

# Document which port the container listens on (this doesn't publish the
# port to the host by itself — `docker run -p` does that — but it's how
# anyone reading this Dockerfile, or another tool, knows what to expect).
EXPOSE 8080

# TODO: the command that starts the server when the container runs. Look at
# the bottom of src/app.py to see how it's invoked directly with `python`.
CMD [???]
