# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

# System deps (optional: add build tools if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app/ ./app/

# MLflow URI can be overridden in docker-compose.yml
ENV MLFLOW_TRACKING_URI=http://mlflow:5001
ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
