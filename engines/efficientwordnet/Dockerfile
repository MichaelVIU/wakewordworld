# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg portaudio19-dev gcc \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install "/app[engines-ewn]" "EfficientWord-Net==1.0.5"
# Enrolment references are mounted under /models at run time and passed via --config
# '{"hotword": "hey michael", "reference_file": "/models/hey_michael_ref.json"}'.
RUN useradd --create-home --uid 10001 bench && chown -R bench:bench /app
USER bench
ENV WWW_DATA_ROOT=/data
VOLUME ["/data", "/models"]
ENTRYPOINT ["wakewordworld", "eval", "run", "efficientwordnet"]
