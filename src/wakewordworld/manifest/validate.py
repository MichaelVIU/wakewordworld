"""Validate a frozen manifest release directory."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import ValidationError

from wakewordworld.licences import LicenceTier
from wakewordworld.manifest.schema import ChunkRow, ManifestRelease

__all__ = ["validate_dir"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_dir(path: Path) -> list[str]:
    """Return a list of problems (empty when the release is valid)."""
    problems: list[str] = []
    chunks = path / "chunks.jsonl"
    release_path = path / "release.json"
    if not chunks.exists():
        return [f"{chunks} missing"]
    if not release_path.exists():
        problems.append(f"{release_path} missing")

    seen: set[str] = set()
    files: set[str] = set()
    n = 0
    with chunks.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            n += 1
            try:
                row = ChunkRow.model_validate_json(line)
            except ValidationError as exc:
                problems.append(f"chunks.jsonl:{lineno}: {exc.errors()[0]['msg']}")
                continue
            if row.chunk_id in seen:
                problems.append(f"chunks.jsonl:{lineno}: duplicate chunk id {row.chunk_id}")
            seen.add(row.chunk_id)
            files.add(row.file_id)
            if row.licence_tier is LicenceTier.FORBIDDEN:
                problems.append(f"chunks.jsonl:{lineno}: forbidden tier")
            if row.licence_tier is LicenceTier.B and row.audio_sha256:
                problems.append(f"chunks.jsonl:{lineno}: tier B row carries audio_sha256")

    if release_path.exists():
        try:
            release = ManifestRelease.model_validate_json(release_path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            problems.append(f"release.json: {exc.errors()[0]['msg']}")
        else:
            if release.n_chunks != n:
                problems.append(f"release.n_chunks={release.n_chunks} but {n} rows")
            if release.n_files != len(files):
                problems.append(f"release.n_files={release.n_files} but {len(files)} files")

    sums = path / "SHA256SUMS"
    if sums.exists():
        for line in sums.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, _, name = line.partition("  ")
            target = path / name
            if not target.exists():
                problems.append(f"SHA256SUMS: {name} missing")
            elif _sha256(target) != digest:
                problems.append(f"SHA256SUMS: {name} checksum mismatch")
    else:
        problems.append("SHA256SUMS missing")
    return problems
