"""Build and upload the public dataset release.

Layout of ``out_dir``::

    README.md                      dataset card
    metadata.parquet               every public manifest row (tier A and tier B)
    release.json, SHA256SUMS, sources.md   copied from the frozen manifest
    data/<family>/shard-00000.parquet      tier A audio, one licence family per folder
    index/<source_id>.parquet      word index rows of tier A chunks
    transcripts/<source_id>.jsonl  transcripts of tier A chunks

Tier B rows appear in ``metadata.parquet`` only. Their audio, transcripts and word
index rows are derivative works of a recording we may not redistribute, so they stay
with the maintainers; results computed on them are published separately.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from wakewordworld.hub.dataset_card import licence_family, load_manifest, render_dataset_card
from wakewordworld.licences import LicenceTier
from wakewordworld.manifest.schema import ChunkRow
from wakewordworld.util.paths import DataRoot

__all__ = ["ReleaseBuild", "build_release_files", "upload_dataset"]

log = logging.getLogger(__name__)

MAX_SHARD_BYTES = 500 * 1024 * 1024
MAX_FOLDER_ENTRIES = 10_000
_COPIED = ("release.json", "SHA256SUMS", "sources.md")


@dataclass
class ReleaseBuild:
    """What ``build_release_files`` produced."""

    out_dir: Path
    n_rows: int
    n_audio: int
    families: dict[str, int] = field(default_factory=dict)
    missing_audio: list[str] = field(default_factory=list)


def _audio_schema(sample: dict[str, object]) -> pa.Schema:
    fields = [pa.field("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())]))]
    for k, v in sample.items():
        if k == "audio":
            continue
        typ: pa.DataType
        if isinstance(v, bool):
            typ = pa.bool_()
        elif isinstance(v, int):
            typ = pa.int64()
        elif isinstance(v, float):
            typ = pa.float64()
        elif isinstance(v, list):
            typ = pa.list_(pa.string())
        else:
            typ = pa.string()
        fields.append(pa.field(k, typ))
    return pa.schema(fields)


class _ShardWriter:
    """Writes rows into size-bounded Parquet shards for one licence family."""

    def __init__(self, folder: Path) -> None:
        self._folder = folder
        self._rows: list[dict[str, object]] = []
        self._bytes = 0
        self._n = 0
        self.count = 0

    def add(self, row: dict[str, object], nbytes: int) -> None:
        if self._rows and self._bytes + nbytes > MAX_SHARD_BYTES:
            self.flush()
        self._rows.append(row)
        self._bytes += nbytes
        self.count += 1

    def flush(self) -> None:
        if not self._rows:
            return
        if self._n >= MAX_FOLDER_ENTRIES:
            msg = f"{self._folder}: more than {MAX_FOLDER_ENTRIES} shards; split the release"
            raise RuntimeError(msg)
        self._folder.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(self._rows, schema=_audio_schema(self._rows[0]))
        pq.write_table(
            table,
            self._folder / f"shard-{self._n:05d}.parquet",
            compression="zstd",
            row_group_size=64,
        )
        self._n += 1
        self._rows = []
        self._bytes = 0


def _metadata_row(row: ChunkRow) -> dict[str, object]:
    d = row.model_dump(mode="json")
    d["licence_tier"] = row.licence_tier.value
    return d


def _write_index_and_transcripts(
    rows: Sequence[ChunkRow], data_root: DataRoot, out_dir: Path
) -> None:
    public_ids = {r.chunk_id for r in rows}
    by_source: dict[str, set[str]] = defaultdict(set)
    files_by_source: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_source[r.source_id].add(r.chunk_id)
        files_by_source[r.source_id].add(r.file_id)
    for sid, chunk_ids in sorted(by_source.items()):
        idx = data_root.index / f"{sid}.parquet"
        if idx.exists():
            df = pl.read_parquet(idx).filter(pl.col("chunk_id").is_in(sorted(chunk_ids)))
            if df.height:
                (out_dir / "index").mkdir(parents=True, exist_ok=True)
                df.write_parquet(out_dir / "index" / f"{sid}.parquet", compression="zstd")
        tdir = data_root.transcripts / sid
        if tdir.exists():
            lines: list[str] = []
            for fid in sorted(files_by_source[sid]):
                p = tdir / f"{fid}.json"
                if p.exists():
                    lines.append(p.read_text(encoding="utf-8").strip())
            if lines:
                (out_dir / "transcripts").mkdir(parents=True, exist_ok=True)
                (out_dir / "transcripts" / f"{sid}.jsonl").write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
    log.info("index/transcripts written for %d public chunks", len(public_ids))


def build_release_files(
    manifest_dir: Path,
    data_root: DataRoot,
    out_dir: Path,
    *,
    repo_id: str = "wakewordworld/benchmark",
    include_audio: bool = True,
) -> ReleaseBuild:
    """Assemble the public dataset directory from a frozen manifest and the data root."""
    rows, _release = load_manifest(manifest_dir)
    if any(r.sealed for r in rows):
        msg = "public manifest contains sealed rows; rebuild with --public"
        raise ValueError(msg)
    if any(r.licence_tier is LicenceTier.FORBIDDEN for r in rows):
        msg = "public manifest contains forbidden rows"
        raise ValueError(msg)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in _COPIED:
        src = manifest_dir / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    pl.DataFrame([_metadata_row(r) for r in rows]).write_parquet(
        out_dir / "metadata.parquet", compression="zstd"
    )

    tier_a = [r for r in rows if r.licence_tier is LicenceTier.A]
    writers: dict[str, _ShardWriter] = {}
    missing: list[str] = []
    n_audio = 0
    if include_audio:
        for r in tier_a:
            audio = data_root.chunks / r.source_id / f"{r.chunk_id}.flac"
            if not audio.exists():
                missing.append(r.chunk_id)
                continue
            fam = licence_family(r.licence_spdx)
            w = writers.setdefault(fam, _ShardWriter(out_dir / "data" / fam))
            data = audio.read_bytes()
            row: dict[str, object] = {
                "audio": {"bytes": data, "path": f"{r.chunk_id}.flac"},
                **_metadata_row(r),
            }
            w.add(row, len(data))
            n_audio += 1
        for w in writers.values():
            w.flush()
        _write_index_and_transcripts(
            [r for r in tier_a if r.chunk_id not in set(missing)], data_root, out_dir
        )
    families = {fam: w.count for fam, w in writers.items()}
    (out_dir / "README.md").write_text(
        render_dataset_card(manifest_dir, repo_id=repo_id, families_present=families or None),
        encoding="utf-8",
    )
    (out_dir / "build.json").write_text(
        json.dumps(
            {
                "n_rows": len(rows),
                "n_audio": n_audio,
                "families": families,
                "missing_audio": missing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if missing:
        log.warning("%d tier A chunks had no audio on disk", len(missing))
    return ReleaseBuild(
        out_dir=out_dir, n_rows=len(rows), n_audio=n_audio, families=families, missing_audio=missing
    )


def upload_dataset(
    out_dir: Path,
    repo_id: str,
    *,
    token: str | None,
    private: bool = True,
    revision_tag: str | None = None,
) -> str:
    """Create (if needed) a gated dataset repo, upload ``out_dir`` and tag the commit.

    Returns the repository URL. Requires ``huggingface_hub`` (``uv sync --extra hub``).
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        msg = "huggingface_hub is not installed; run `uv sync --extra hub`"
        raise RuntimeError(msg) from exc
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.update_repo_settings(repo_id, repo_type="dataset", gated="auto")
    message = f"release {revision_tag}" if revision_tag else "release"
    if hasattr(api, "upload_large_folder"):
        api.upload_large_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(out_dir))
    else:  # pragma: no cover - older clients
        api.upload_folder(
            repo_id=repo_id, repo_type="dataset", folder_path=str(out_dir), commit_message=message
        )
    if revision_tag:
        api.create_tag(
            repo_id, tag=revision_tag, repo_type="dataset", tag_message=message, exist_ok=True
        )
    return f"https://huggingface.co/datasets/{repo_id}"
