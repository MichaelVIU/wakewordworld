"""Duplicate detection across normalised files.

Version 0 treats two files as duplicates when their normalised audio checksums or
their Chromaprint fingerprints are identical. ``similar`` is the hook for a
tolerant comparison (bit-error rate over the fingerprint integers) in a later version.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from wakewordworld.ingest.records import FileRecord

__all__ = ["mark_duplicates", "similar_exact"]


def similar_exact(a: str, b: str) -> bool:
    """Exact fingerprint equality (the v0 similarity function)."""
    return a == b


def mark_duplicates(
    records: Iterable[FileRecord],
    *,
    similar: Callable[[str, str], bool] = similar_exact,
) -> list[FileRecord]:
    """Return copies of ``records`` with ``duplicate_of`` set for later duplicates.

    Records are processed in ``file_id`` order so the kept copy is deterministic. A
    record that already carries ``duplicate_of`` keeps it.
    """
    ordered = sorted(records, key=lambda r: r.file_id)
    by_sha: dict[str, str] = {}
    kept_fps: list[tuple[str, str]] = []  # (fingerprint, file_id)
    out: list[FileRecord] = []
    for rec in ordered:
        dup: str | None = rec.duplicate_of
        if dup is None:
            if rec.audio_sha256 in by_sha:
                dup = by_sha[rec.audio_sha256]
            elif rec.fingerprint:
                if similar is similar_exact:
                    for fp, fid in kept_fps:
                        if fp == rec.fingerprint:
                            dup = fid
                            break
                else:
                    for fp, fid in kept_fps:
                        if similar(fp, rec.fingerprint):
                            dup = fid
                            break
        if dup is None:
            by_sha.setdefault(rec.audio_sha256, rec.file_id)
            if rec.fingerprint:
                kept_fps.append((rec.fingerprint, rec.file_id))
            out.append(rec)
        else:
            out.append(rec.model_copy(update={"duplicate_of": dup}))
    return out
