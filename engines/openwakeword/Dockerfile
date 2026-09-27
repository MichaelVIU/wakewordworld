# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install "/app[engines-oww]" "openwakeword==0.6.0" "onnxruntime>=1.18"
# Pre-download built-in models so the container can run without network access.
RUN python -c "from openwakeword.utils import download_models; download_models(model_names=['alexa','hey_jarvis','hey_mycroft'])"
RUN useradd --create-home --uid 10001 bench && chown -R bench:bench /app
USER bench
ENV WWW_DATA_ROOT=/data
VOLUME ["/data"]
ENTRYPOINT ["wakewordworld", "eval", "run", "openwakeword"]
