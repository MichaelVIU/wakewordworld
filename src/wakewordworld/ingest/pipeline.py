"""Ingestion pipeline: fetch -> download -> normalise -> fingerprint/dedupe -> chunk.

State lives in JSONL files under the data root so every stage is resumable:

* ``items/<source_id>.jsonl``            fetched items (:class:`FetchedItem`)
* ``items/<source_id>.downloads.jsonl``  downloaded originals (:class:`DownloadRecord`)
* ``items/<source_id>.failures.jsonl``   per-item failures (:class:`FailureRecord`)
* ``audio/<source_id>/files.jsonl``      normalised files (:class:`FileRecord`)
* ``chunks/<source_id>/chunks.jsonl``    evaluation chunks (:class:`ChunkRecord`)
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict

from wakewordworld.ingest.archive import expand_archive, is_archive
from wakewordworld.ingest.chunking import ChunkPolicy, write_chunks
from wakewordworld.ingest.dedupe import mark_duplicates
from wakewordworld.ingest.download import DiskSpaceError, ForbiddenItemError, download_item
from wakewordworld.ingest.fingerprint import fingerprint
from wakewordworld.ingest.items import ItemStore
from wakewordworld.ingest.normalize import FfmpegError, normalize_file
from wakewordworld.ingest.parquet_audio import expand_parquet
from wakewordworld.ingest.records import ChunkRecord, FileRecord, read_jsonl, write_jsonl
from wakewordworld.licences import LicenceTier
from wakewordworld.manifest.schema import FetchedItem
from wakewordworld.sources.base import http_client
from wakewordworld.sources.spec import HttpArchiveAccess, HuggingFaceAccess, SourceSpec
from wakewordworld.util.paths import DataRoot

__all__ = [
    "DownloadRecord",
    "FailureRecord",
    "IngestOptions",
    "IngestPipeline",
    "SourceStatus",
]

log = logging.getLogger(__name__)

# Chromaprint returns an empty fingerprint for very short clips; skip those.
MIN_FINGERPRINT_S = 3.0


class DownloadRecord(BaseModel):
    """A stored original for an item."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    source_id: str
    sha256: str
    path: str
    bytes: int
    ext: str


class FailureRecord(BaseModel):
    """A per-item failure at some stage."""

    model_config = ConfigDict(extra="forbid")

    item_id: str
    source_id: str
    stage: str
    error: str
    at: datetime


@dataclass(frozen=True)
class IngestOptions:
    """Budgets and limits for a pipeline run."""

    max_items: int | None = None
    max_hours: float | None = None
    max_rows_per_item: int | None = 500
    min_free_gb: float = 5.0
    max_bytes_per_item: int | None = None
    measure_quality: bool = True
    chunk_policy: ChunkPolicy = field(default_factory=ChunkPolicy)


@dataclass(frozen=True)
class SourceStatus:
    """Counts for one source."""

    source_id: str
    items: int
    downloaded: int
    files: int
    duplicates: int
    hours: float
    chunks: int
    failures: int


class IngestPipeline:
    """Runs ingestion stages for a set of source specs."""

    def __init__(
        self,
        data_root: DataRoot,
        specs: Iterable[SourceSpec],
        *,
        options: IngestOptions | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.data_root = data_root
        self.specs = {s.id: s for s in specs}
        self.options = options or IngestOptions()
        self._client = client
        self.items = ItemStore(data_root)
        data_root.ensure()

    # ------------------------------------------------------------------ paths
    def _downloads_path(self, source_id: str) -> Path:
        return self.data_root.items / f"{source_id}.downloads.jsonl"

    def _failures_path(self, source_id: str) -> Path:
        return self.data_root.items / f"{source_id}.failures.jsonl"

    def _files_path(self, source_id: str) -> Path:
        return self.data_root.audio / source_id / "files.jsonl"

    def _chunks_path(self, source_id: str) -> Path:
        return self.data_root.chunks / source_id / "chunks.jsonl"

    def _spec(self, source_id: str) -> SourceSpec:
        try:
            return self.specs[source_id]
        except KeyError as exc:
            msg = f"unknown source {source_id!r}"
            raise LookupError(msg) from exc

    # ---------------------------------------------------------------- records
    def downloads(self, source_id: str) -> list[DownloadRecord]:
        """Stored download records."""
        return list(read_jsonl(self._downloads_path(source_id), DownloadRecord))

    def files(self, source_id: str) -> list[FileRecord]:
        """Stored file records."""
        return list(read_jsonl(self._files_path(source_id), FileRecord))

    def chunks(self, source_id: str) -> list[ChunkRecord]:
        """Stored chunk records."""
        return list(read_jsonl(self._chunks_path(source_id), ChunkRecord))

    def failures(self, source_id: str) -> list[FailureRecord]:
        """Stored failure records."""
        return list(read_jsonl(self._failures_path(source_id), FailureRecord))

    def _fail(self, source_id: str, item_id: str, stage: str, exc: BaseException) -> None:
        log.warning("%s/%s failed at %s: %s", source_id, item_id, stage, exc)
        log.debug("%s", "".join(traceback.format_exception(exc)))
        rec = FailureRecord(
            item_id=item_id,
            source_id=source_id,
            stage=stage,
            error=f"{type(exc).__name__}: {exc}"[:2000],
            at=datetime.now(UTC),
        )
        path = self._failures_path(source_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(rec.model_dump_json())
            fh.write("\n")

    def _hours(self, source_id: str) -> float:
        return sum(f.duration_s for f in self.files(source_id) if f.duplicate_of is None) / 3600

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = http_client()
        return self._client

    # ------------------------------------------------------------------ fetch
    def fetch(self, source_id: str) -> int:
        """Discover items for a source and merge them into the item store."""
        spec = self._spec(source_id)
        try:
            from wakewordworld.sources.fetchers import fetch_source
        except ImportError as exc:  # pragma: no cover - depends on sibling module
            msg = (
                "wakewordworld.sources.fetchers.fetch_source is not available; "
                "the fetchers module has not been implemented or installed"
            )
            raise RuntimeError(msg) from exc
        items: list[FetchedItem] = []
        for item in fetch_source(spec, client=self._http(), max_items=self.options.max_items):
            if item.licence.tier is LicenceTier.FORBIDDEN:
                continue
            items.append(item)
            if self.options.max_items is not None and len(items) >= self.options.max_items:
                break
        return self.items.write(source_id, items)

    # --------------------------------------------------------------- download
    def _select_items(self, source_id: str) -> list[FetchedItem]:
        items = list(self.items.read(source_id))
        if self.options.max_items is not None:
            items = items[: self.options.max_items]
        return items

    def download(self, source_id: str) -> int:
        """Download originals for stored items; returns the number of new downloads."""
        spec = self._spec(source_id)
        done = {d.item_id for d in self.downloads(source_id)}
        new: list[DownloadRecord] = []
        for item in self._select_items(source_id):
            if item.item_id in done:
                continue
            if (
                self.options.max_hours is not None
                and self._hours(source_id) >= self.options.max_hours
            ):
                break
            try:
                path, sha, size = download_item(
                    item,
                    self.data_root,
                    client=self._http(),
                    max_bytes=self.options.max_bytes_per_item,
                    min_free_gb=self.options.min_free_gb,
                )
            except DiskSpaceError:
                raise
            except (ForbiddenItemError, httpx.HTTPError, ValueError, OSError) as exc:
                self._fail(source_id, item.item_id, "download", exc)
                continue
            rec = DownloadRecord(
                item_id=item.item_id,
                source_id=spec.id,
                sha256=sha,
                path=str(path.relative_to(self.data_root.root)),
                bytes=size,
                ext=path.name.split(".", 1)[1] if "." in path.name else "bin",
            )
            new.append(rec)
            write_jsonl(self._downloads_path(source_id), [rec], key="item_id")
        return len(new)

    # -------------------------------------------------------------- normalise
    def _originals_for(
        self, spec: SourceSpec, item: FetchedItem, dl: DownloadRecord
    ) -> Iterator[tuple[Path, str, str | None]]:
        """Yield ``(path, sha256, member_path)`` for every source file behind an item."""
        original = self.data_root.root / dl.path
        access = spec.access
        if dl.ext == "parquet" or (item.media_type or "").endswith("parquet"):
            audio_col = access.audio_column if isinstance(access, HuggingFaceAccess) else "audio"
            text_col = access.text_column if isinstance(access, HuggingFaceAccess) else None
            for row in expand_parquet(
                original,
                item_id=item.item_id,
                source_id=spec.id,
                data_root=self.data_root,
                audio_column=audio_col,
                text_column=text_col,
                max_rows=self.options.max_rows_per_item,
            ):
                yield row.path, row.sha256, row.member_path
            return
        if is_archive(original):
            glob = item.extra.get("audio_glob")
            if not glob and isinstance(access, HttpArchiveAccess):
                glob = access.audio_glob
            for member in expand_archive(
                original,
                item_id=item.item_id,
                data_root=self.data_root,
                audio_glob=glob or "**/*.wav",
            ):
                yield member.path, member.sha256, member.member_path
            return
        yield original, dl.sha256, None

    def normalize(self, source_id: str) -> int:
        """Transcode downloaded originals to working copies; returns new file count."""
        spec = self._spec(source_id)
        items = {it.item_id: it for it in self.items.read(source_id)}
        existing = {f.original_sha256 for f in self.files(source_id)}
        count = 0
        for dl in self.downloads(source_id):
            item = items.get(dl.item_id)
            if item is None:
                continue
            if (
                self.options.max_hours is not None
                and self._hours(source_id) >= self.options.max_hours
            ):
                break
            try:
                for path, sha, member in self._originals_for(spec, item, dl):
                    if sha in existing:
                        continue
                    if (
                        self.options.max_hours is not None
                        and self._hours(source_id) >= self.options.max_hours
                    ):
                        break
                    rec = normalize_file(
                        path,
                        item_id=item.item_id,
                        source_id=spec.id,
                        original_sha256=sha,
                        data_root=self.data_root,
                        member_path=member,
                        measure=self.options.measure_quality,
                    )
                    write_jsonl(self._files_path(source_id), [rec], key="file_id")
                    text = item.extra.get("text")
                    if text and member is None:
                        ref = self.data_root.cache / "reftext" / spec.id / f"{rec.file_id}.txt"
                        ref.parent.mkdir(parents=True, exist_ok=True)
                        ref.write_text(text.strip() + "\n", encoding="utf-8")
                    existing.add(sha)
                    count += 1
            except (FfmpegError, OSError, ValueError) as exc:
                self._fail(source_id, dl.item_id, "normalize", exc)
        return count

    # ------------------------------------------------------ fingerprint/dedupe
    def dedupe(self, source_id: str) -> int:
        """Fingerprint files lacking one and mark duplicates; returns duplicate count."""
        records = self.files(source_id)
        updated: list[FileRecord] = []
        for rec in records:
            if rec.fingerprint is None and rec.duration_s >= MIN_FINGERPRINT_S:
                try:
                    fp = fingerprint(self.data_root.root / rec.audio_path)
                    rec = rec.model_copy(update={"fingerprint": fp})
                except FfmpegError as exc:
                    self._fail(source_id, rec.item_id, "fingerprint", exc)
            updated.append(rec)
        marked = mark_duplicates(updated)
        write_jsonl(self._files_path(source_id), marked, key="file_id")
        return sum(1 for r in marked if r.duplicate_of is not None)

    # ------------------------------------------------------------------ chunk
    def chunk(self, source_id: str) -> int:
        """Cut non-duplicate files into evaluation chunks; returns new chunk count."""
        done = {c.file_id for c in self.chunks(source_id)}
        count = 0
        for rec in self.files(source_id):
            if rec.duplicate_of is not None or rec.file_id in done:
                continue
            try:
                chunks = write_chunks(
                    self.data_root.root / rec.audio_path,
                    file_id=rec.file_id,
                    source_id=source_id,
                    data_root=self.data_root,
                    policy=self.options.chunk_policy,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                self._fail(source_id, rec.item_id, "chunk", exc)
                continue
            write_jsonl(self._chunks_path(source_id), chunks, key="chunk_id")
            count += len(chunks)
        return count

    # -------------------------------------------------------------------- run
    def run(self, source_id: str, *, do_fetch: bool = True) -> SourceStatus:
        """Run all stages for one source."""
        if do_fetch:
            self.fetch(source_id)
        self.download(source_id)
        self.normalize(source_id)
        self.dedupe(source_id)
        self.chunk(source_id)
        return self.status(source_id)

    def status(self, source_id: str) -> SourceStatus:
        """Summarise the state of one source."""
        self._spec(source_id)
        files = self.files(source_id)
        return SourceStatus(
            source_id=source_id,
            items=sum(1 for _ in self.items.read(source_id)),
            downloaded=len(self.downloads(source_id)),
            files=len(files),
            duplicates=sum(1 for f in files if f.duplicate_of is not None),
            hours=self._hours(source_id),
            chunks=len(self.chunks(source_id)),
            failures=len(self.failures(source_id)),
        )
