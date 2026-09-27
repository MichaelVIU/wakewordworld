"""Streaming, resumable, content-addressed downloads of fetched items."""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from wakewordworld.licences import LicenceTier
from wakewordworld.manifest.schema import FetchedItem
from wakewordworld.util.paths import DataRoot

__all__ = [
    "DiskSpaceError",
    "ForbiddenItemError",
    "check_free_space",
    "download_item",
    "guess_extension",
]

log = logging.getLogger(__name__)

_CONTENT_TYPE_EXT: dict[str, str] = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mpeg3": "mp3",
    "audio/ogg": "ogg",
    "audio/vorbis": "ogg",
    "audio/opus": "opus",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/aac": "aac",
    "audio/webm": "webm",
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/x-matroska": "mkv",
    "video/quicktime": "mov",
    "application/x-parquet": "parquet",
    "application/vnd.apache.parquet": "parquet",
    "application/zip": "zip",
    "application/x-zip-compressed": "zip",
    "application/x-tar": "tar",
    "application/gzip": "tar.gz",
    "application/x-gzip": "tar.gz",
}

_KNOWN_EXTS: frozenset[str] = frozenset(
    {
        "mp3",
        "ogg",
        "oga",
        "opus",
        "flac",
        "wav",
        "m4a",
        "aac",
        "webm",
        "mp4",
        "mkv",
        "mov",
        "parquet",
        "zip",
        "tar",
        "tgz",
        "tar.gz",
    }
)


class DiskSpaceError(RuntimeError):
    """Raised when the data root has less free space than the configured minimum."""


class ForbiddenItemError(RuntimeError):
    """Raised when an item's licence tier forbids any download."""


def check_free_space(root: Path, min_free_gb: float) -> None:
    """Raise :class:`DiskSpaceError` if free space under ``root`` is below the minimum."""
    probe = root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    if free_gb < min_free_gb:
        msg = f"only {free_gb:.1f} GB free under {root}, minimum is {min_free_gb:.1f} GB"
        raise DiskSpaceError(msg)


def guess_extension(url: str, content_type: str | None, media_type: str | None) -> str:
    """Pick a file extension from the URL path, then Content-Type, then item media type."""
    path = unquote(urlparse(url).path).lower()
    for ext in sorted(_KNOWN_EXTS, key=len, reverse=True):
        if path.endswith(f".{ext}"):
            return ext
    for ct in (content_type, media_type):
        if not ct:
            continue
        base = ct.split(";", 1)[0].strip().lower()
        if base in _CONTENT_TYPE_EXT:
            return _CONTENT_TYPE_EXT[base]
        guessed = mimetypes.guess_extension(base)
        if guessed:
            return guessed.lstrip(".")
    return "bin"


def _part_path(data_root: DataRoot, item: FetchedItem) -> Path:
    return data_root.cache / "downloads" / f"{item.item_id}.part"


def download_item(
    item: FetchedItem,
    data_root: DataRoot,
    *,
    client: httpx.Client,
    max_bytes: int | None = None,
    min_free_gb: float = 5.0,
) -> tuple[Path, str, int]:
    """Download an item into the content-addressed originals store.

    The download streams to ``cache/downloads/<item_id>.part``; if a partial file
    exists a ``Range`` request resumes it. The SHA-256 is computed while streaming
    (re-hashing the existing prefix on resume) and the file is moved to
    ``originals/<sha[:2]>/<sha>.<ext>`` when complete.

    Args:
        item: The fetched item (must not be tier ``forbidden``).
        data_root: Data directory.
        client: HTTP client to use.
        max_bytes: Abort when more than this many bytes would be written.
        min_free_gb: Minimum free disk space required before starting.

    Returns:
        ``(path, sha256, size_bytes)`` of the stored original.

    Raises:
        ForbiddenItemError: for items whose licence forbids use.
        DiskSpaceError: when free disk space is below ``min_free_gb``.
        httpx.HTTPError: on network failures.
        ValueError: when ``max_bytes`` is exceeded.
    """
    if item.licence.tier is LicenceTier.FORBIDDEN:
        msg = f"item {item.item_id} ({item.url}) has a forbidden licence tier"
        raise ForbiddenItemError(msg)
    check_free_space(data_root.root, min_free_gb)

    part = _part_path(data_root, item)
    part.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    offset = 0
    if part.exists():
        with part.open("rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
                offset += len(block)

    headers: dict[str, str] = {}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    with client.stream("GET", item.url, headers=headers) as resp:
        if offset and resp.status_code == 200:
            # Server ignored the range: start over.
            log.info("server ignored Range for %s; restarting download", item.item_id)
            digest = hashlib.sha256()
            offset = 0
            mode = "wb"
        elif offset and resp.status_code == 206:
            mode = "ab"
        else:
            resp.raise_for_status()
            mode = "wb"
        resp.raise_for_status()
        content_type = resp.headers.get("content-type")
        written = offset
        with part.open(mode) as fh:
            for block in resp.iter_bytes(1 << 20):
                if not block:
                    continue
                written += len(block)
                if max_bytes is not None and written > max_bytes:
                    part.unlink(missing_ok=True)
                    msg = f"item {item.item_id} exceeds max_bytes={max_bytes}"
                    raise ValueError(msg)
                digest.update(block)
                fh.write(block)

    sha = digest.hexdigest()
    ext = guess_extension(item.url, content_type, item.media_type)
    final = data_root.original_path(sha, ext)
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        part.unlink(missing_ok=True)
    else:
        part.replace(final)
    return final, sha, written
