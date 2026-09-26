# Already complete (this is Week 6's solved Dockerfile) — this week's work
# is the CI pipeline that builds and tests this image, not the Dockerfile
# itself. Don't edit this file.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/

EXPOSE 8080

CMD ["python", "src/app.py"]
