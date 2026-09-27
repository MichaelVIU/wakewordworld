"""Acoustic fingerprints with Chromaprint (``fpcalc``)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from wakewordworld.ingest.normalize import FfmpegError, find_tool

__all__ = ["fingerprint"]


def fingerprint(path: Path, *, length_s: int = 120, timeout: float = 300.0) -> str:
    """Return the compressed Chromaprint fingerprint of the first ``length_s`` seconds."""
    cmd = [find_tool("fpcalc"), "-json", "-length", str(length_s), str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        msg = f"fpcalc timed out on {path}"
        raise FfmpegError(msg) from exc
    if proc.returncode != 0:
        msg = f"fpcalc failed on {path}: {proc.stderr.strip()[-300:]}"
        raise FfmpegError(msg)
    data = json.loads(proc.stdout)
    fp = data.get("fingerprint")
    if not isinstance(fp, str) or not fp:
        msg = f"fpcalc returned no fingerprint for {path}"
        raise FfmpegError(msg)
    return fp
