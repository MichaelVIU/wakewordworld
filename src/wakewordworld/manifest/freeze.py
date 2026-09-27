"""Write a manifest release to ``manifests/<version>/``."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from wakewordworld.manifest.schema import ChunkRow, ManifestRelease

__all__ = ["freeze", "write_rows"]

_SEALED_README = """# Sealed split

This directory holds the sealed (held-out) rows of the release. It must never be
committed or published. Only maintainer-run evaluation jobs read it.
"""


def write_rows(rows: Sequence[ChunkRow], out_dir: Path) -> list[Path]:
    """Write ``chunks.jsonl`` and ``chunks.parquet``; return the written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: r.chunk_id)
    jsonl = out_dir / "chunks.jsonl"
    with jsonl.open("w", encoding="utf-8") as fh:
        for r in ordered:
            fh.write(r.model_dump_json())
            fh.write("\n")
    parquet = out_dir / "chunks.parquet"
    pl.DataFrame([r.model_dump(mode="json") for r in ordered]).write_parquet(
        parquet, compression="zstd"
    )
    return [jsonl, parquet]


def _sources_md(rows: Sequence[ChunkRow]) -> str:
    hours: dict[str, float] = defaultdict(float)
    licences: dict[str, set[str]] = defaultdict(set)
    tiers: dict[str, set[str]] = defaultdict(set)
    example: dict[str, str] = {}
    for r in rows:
        hours[r.source_id] += r.duration_s / 3600.0
        licences[r.source_id].add(r.licence_spdx)
        tiers[r.source_id].add(r.licence_tier.value)
        example.setdefault(r.source_id, r.page_url or r.item_url)
    lines = [
        "# Sources in this release",
        "",
        "| source | licence(s) | tier | hours | example item |",
        "|---|---|---|---|---|",
    ]
    for sid in sorted(hours):
        lines.append(
            f"| {sid} | {', '.join(sorted(licences[sid]))} | {', '.join(sorted(tiers[sid]))} | "
            f"{hours[sid]:.2f} | {example[sid]} |"
        )
    lines.append("")
    lines.append("Per-item attribution is in the `attribution` column of `chunks.jsonl`.")
    return "\n".join(lines) + "\n"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def freeze(
    rows: Sequence[ChunkRow],
    release: ManifestRelease,
    out_dir: Path,
    *,
    sealed_rows: Sequence[ChunkRow] = (),
    force: bool = False,
) -> Path:
    """Write a complete release directory and return it.

    Refuses to overwrite an existing directory unless ``force`` is set.
    """
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        msg = f"{out_dir} exists; pass force=True to overwrite"
        raise FileExistsError(msg)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = write_rows(rows, out_dir)
    release_path = out_dir / "release.json"
    release_path.write_text(release.model_dump_json(indent=2), encoding="utf-8")
    sources_path = out_dir / "sources.md"
    sources_path.write_text(_sources_md(rows), encoding="utf-8")
    written += [release_path, sources_path]
    sums = out_dir / "SHA256SUMS"
    sums.write_text(
        "".join(f"{_sha256(p)}  {p.name}\n" for p in sorted(written, key=lambda p: p.name)),
        encoding="utf-8",
    )
    if sealed_rows:
        sealed_dir = out_dir / "sealed"
        write_rows(sealed_rows, sealed_dir)
        (sealed_dir / "README.md").write_text(_SEALED_README, encoding="utf-8")
    return out_dir
