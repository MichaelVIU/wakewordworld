# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
# openwakeword 0.6.0 declares tflite-runtime on Linux, which has no wheels for
# Python 3.11+; install it without dependencies and provide the ONNX path instead.
RUN pip install "/app" \
    && pip install --no-deps "openwakeword==0.6.0" \
    && pip install "onnxruntime>=1.18" "scipy>=1.11" "scikit-learn>=1.3" "tqdm" "requests"
# Pre-download built-in models so the container can run without network access.
RUN python -c "from openwakeword.utils import download_models; download_models(model_names=['alexa','hey_jarvis','hey_mycroft'])"
RUN useradd --create-home --uid 10001 bench && chown -R bench:bench /app
USER bench
ENV WWW_DATA_ROOT=/data
VOLUME ["/data"]
ENTRYPOINT ["wakewordworld", "eval", "run", "openwakeword"]
