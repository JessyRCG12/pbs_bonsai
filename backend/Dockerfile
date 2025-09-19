FROM mcr.microsoft.com/devcontainers/python:3.11-bookworm

WORKDIR /app

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app/ ./app/

# Env
ENV MLFLOW_TRACKING_URI=http://mlflow:5001
ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8080"]