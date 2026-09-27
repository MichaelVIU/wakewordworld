"""Hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = ["sha256_bytes", "sha256_file", "short_id"]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """SHA-256 hex digest of a byte string."""
    return hashlib.sha256(data).hexdigest()


def short_id(*parts: str, length: int = 16) -> str:
    """Stable short identifier derived from string parts."""
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:length]
