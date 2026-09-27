# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install "/app[engines-sherpa]" "sherpa-onnx==1.13.8" sentencepiece pypinyin
# Pre-download the pinned model archive so the container runs offline.
RUN WWW_DATA_ROOT=/opt/wwwcache python -c "from wakewordworld.engines.sherpa_kws import ensure_sherpa_model; print(ensure_sherpa_model())"
RUN useradd --create-home --uid 10001 bench && chown -R bench:bench /app /opt/wwwcache
USER bench
ENV WWW_DATA_ROOT=/data WWW_MODEL_CACHE=/opt/wwwcache
VOLUME ["/data"]
ENTRYPOINT ["wakewordworld", "eval", "run", "sherpa_kws"]
