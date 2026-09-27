# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg unzip \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install "/app[engines-vosk]" "vosk==0.3.44"
ENV WWW_DATA_ROOT=/data
# Pre-download the pinned English model into the benchmark cache (checksum verified).
RUN python -c "from wakewordworld.engines.vosk import ensure_vosk_model; ensure_vosk_model('vosk-model-small-en-us-0.15', expected_sha256='30f26242c4eb449f948e42cb302dd7a686cb29a3423a8367f99ff41780942498')"
RUN useradd --create-home --uid 10001 bench && chown -R bench:bench /app /data
USER bench
VOLUME ["/data"]
ENTRYPOINT ["wakewordworld", "eval", "run", "vosk"]
