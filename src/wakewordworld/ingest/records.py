"""Records produced by ingestion: normalised files and evaluation chunks.

``FileRecord`` describes one normalised working copy (16 kHz mono FLAC) derived from
one fetched item. ``ChunkRecord`` describes one evaluation chunk cut from a file.
Both are stored as JSONL next to the audio (``audio/<source_id>/files.jsonl`` and
``chunks/<source_id>/chunks.jsonl``).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from wakewordworld.util.hashing import short_id

__all__ = ["ChunkRecord", "FileRecord", "chunk_id_for", "file_id_for", "read_jsonl", "write_jsonl"]

TARGET_SAMPLE_RATE = 16_000
T = TypeVar("T", bound=BaseModel)


def file_id_for(item_id: str, original_sha256: str) -> str:
    """Stable file id from the item id and the original's checksum."""
    return short_id(item_id, original_sha256)


def chunk_id_for(file_id: str, start_ms: int) -> str:
    """Stable chunk id from the parent file id and the chunk offset."""
    return short_id(file_id, str(start_ms))


class FileRecord(BaseModel):
    """One normalised audio file."""

    model_config = ConfigDict(extra="forbid")

    file_id: str
    item_id: str
    source_id: str
    original_sha256: str
    original_ext: str
    original_bytes: int
    audio_path: str = Field(description="Path relative to the data root.")
    audio_sha256: str
    duration_s: float = Field(ge=0)
    sample_rate: int = TARGET_SAMPLE_RATE
    channels_original: int | None = None
    sample_rate_original: int | None = None
    loudness_lufs: float | None = None
    peak_dbfs: float | None = None
    clipping_ratio: float | None = None
    fingerprint: str | None = Field(default=None, description="Chromaprint fingerprint (base64).")
    duplicate_of: str | None = Field(default=None, description="file_id of the kept duplicate.")


class ChunkRecord(BaseModel):
    """One evaluation chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    file_id: str
    source_id: str
    start_s: float = Field(ge=0)
    duration_s: float = Field(gt=0)
    audio_path: str
    audio_sha256: str
    cut_reason: str = Field(description="silence | max_length | file_end")


def read_jsonl(path: Path, model: type[T]) -> Iterator[T]:
    """Yield validated models from a JSONL file (empty if missing)."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield model.model_validate_json(line)


def write_jsonl(path: Path, rows: Iterable[T], *, key: str) -> int:
    """Merge rows into a JSONL file by ``key`` attribute; returns row count."""
    existing: dict[str, T] = {}
    model: type[T] | None = None
    rows = list(rows)
    if rows:
        model = type(rows[0])
    if path.exists() and model is not None:
        existing = {getattr(r, key): r for r in read_jsonl(path, model)}
    for r in rows:
        existing[getattr(r, key)] = r
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in sorted(existing.values(), key=lambda x: getattr(x, key)):
            fh.write(r.model_dump_json())
            fh.write("\n")
    tmp.replace(path)
    return len(existing)
