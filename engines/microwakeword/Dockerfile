# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN pip install "/app[engines-mww]" "pymicro-wakeword==2.5.0"
# Models are bundled in the wheel; verify they load.
RUN python -c "import pymicro_wakeword as pw; [pw.MicroWakeWord.from_builtin(m) for m in pw.Model]"
RUN useradd --create-home --uid 10001 bench && chown -R bench:bench /app
USER bench
ENV WWW_DATA_ROOT=/data
VOLUME ["/data"]
ENTRYPOINT ["wakewordworld", "eval", "run", "microwakeword"]
