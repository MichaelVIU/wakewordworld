"""Transcription pipeline: reference-first, ASR otherwise.

For every normalised file of a source:

1. Skip duplicates and files that already have a transcript (unless ``force``).
2. Look for a reference transcript (see :func:`find_reference`). If found, parse it,
   attach word timings with the requested aligner, and save with origin
   ``reference_aligned`` (MMS_FA) or ``reference`` (uniform / none).
3. Otherwise run the ASR backend and save with origin ``asr``.
4. Tag music segments heuristically.

Failures are appended to ``transcripts/<source_id>/failures.jsonl`` and never abort
the run.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import soundfile as sf
from pydantic import BaseModel, ConfigDict

from wakewordworld.ingest.records import FileRecord, read_jsonl
from wakewordworld.sources.spec import HttpArchiveAccess, Language, SourceSpec
from wakewordworld.transcribe.align import align_reference, align_uniform
from wakewordworld.transcribe.backends.base import AsrBackend, BackendUnavailable
from wakewordworld.transcribe.music import tag_music
from wakewordworld.transcribe.reference import ReferenceFormat, parse_reference, sniff_format
from wakewordworld.transcribe.schema import Segment, Transcript
from wakewordworld.util.paths import DataRoot

__all__ = [
    "AlignBackend",
    "ReferenceHit",
    "TranscribeFailure",
    "TranscribeSummary",
    "find_reference",
    "iter_file_records",
    "transcribe_file",
    "transcribe_source",
    "transcript_path",
]

AlignBackend = Literal["mms_fa", "uniform", "none"]
_REF_EXTS: tuple[str, ...] = (".mrt", ".vtt", ".srt", ".txt")


class TranscribeFailure(BaseModel):
    """One failed file."""

    model_config = ConfigDict(extra="forbid")

    file_id: str
    source_id: str
    stage: str
    error: str
    at: datetime


@dataclass
class ReferenceHit:
    """A located reference transcript."""

    path: Path
    fmt: ReferenceFormat
    how: str


@dataclass
class TranscribeSummary:
    """Counts for one source run."""

    source_id: str
    done: int = 0
    skipped: int = 0
    failed: int = 0
    hours: float = 0.0
    origins: dict[str, int] = field(default_factory=dict)

    def bump(self, origin: str) -> None:
        """Count a completed transcript by origin."""
        self.origins[origin] = self.origins.get(origin, 0) + 1


def transcript_path(data_root: DataRoot, source_id: str, file_id: str) -> Path:
    """Where the transcript JSON of a file lives."""
    return data_root.transcripts / source_id / f"{file_id}.json"


def iter_file_records(data_root: DataRoot, source_id: str) -> Iterator[FileRecord]:
    """Yield non-duplicate file records of a source."""
    for rec in read_jsonl(data_root.audio / source_id / "files.jsonl", FileRecord):
        if rec.duplicate_of is None:
            yield rec


def find_reference(data_root: DataRoot, spec: SourceSpec, rec: FileRecord) -> ReferenceHit | None:
    """Locate a reference transcript for a file, if any.

    Lookup order:

    1. ``cache/reftext/<source_id>/<file_id>.txt`` (sentence text stored by the
       ingester for dataset rows such as Common Voice or VoxPopuli).
    2. ``cache/refs/<source_id>/<file_id>.<ext>`` for ext in mrt/vtt/srt/txt.
    3. ``cache/refs/<item_id>.<ext>`` (downloaded from ``FetchedItem.transcript_url``).
    4. For archive sources with ``transcript_glob``: a file under
       ``cache/archives/<source_id>/`` matching the glob whose stem equals the stem of
       the record's ``member_path`` (if the record carries one) or of the audio file.
    """
    cache = data_root.cache
    p = cache / "reftext" / spec.id / f"{rec.file_id}.txt"
    if p.exists():
        return ReferenceHit(p, "txt", "reftext")
    for ext in _REF_EXTS:
        p = cache / "refs" / spec.id / f"{rec.file_id}{ext}"
        if p.exists():
            fmt = sniff_format(p)
            if fmt:
                return ReferenceHit(p, fmt, "refs/file_id")
    for ext in _REF_EXTS:
        p = cache / "refs" / f"{rec.item_id}{ext}"
        if p.exists():
            fmt = sniff_format(p)
            if fmt:
                return ReferenceHit(p, fmt, "refs/item_id")
    access = spec.access
    if isinstance(access, HttpArchiveAccess) and access.transcript_glob:
        member = getattr(rec, "member_path", None)
        stem = Path(str(member)).stem if member else Path(rec.audio_path).stem
        root = cache / "archives" / spec.id
        if root.exists():
            for cand in root.glob(access.transcript_glob):
                if cand.stem == stem or cand.stem.split(".")[0] == stem.split(".")[0]:
                    declared = access.transcript_format
                    fmt_arch: ReferenceFormat | None = (
                        declared if declared in ("icsi_mrt", "txt") else None
                    )
                    fmt_arch = fmt_arch or sniff_format(cand)
                    if fmt_arch:
                        return ReferenceHit(cand, fmt_arch, "archive")
    return None


def _language_for(spec: SourceSpec, rec: FileRecord) -> Language | None:
    if len(spec.languages) == 1:
        return spec.languages[0]
    return None


def _language_or_default(spec: SourceSpec, rec: FileRecord) -> Language:
    return _language_for(spec, rec) or spec.languages[0]


def transcribe_file(
    data_root: DataRoot,
    spec: SourceSpec,
    rec: FileRecord,
    *,
    backend: AsrBackend | None,
    align_backend: AlignBackend = "mms_fa",
    tag: bool = True,
) -> Transcript:
    """Produce the transcript of one file (reference-first, ASR otherwise)."""
    audio_path = data_root.root / rec.audio_path
    duration = rec.duration_s or float(sf.info(str(audio_path)).duration)
    language = _language_for(spec, rec)
    ref = find_reference(data_root, spec, rec)
    if ref is not None:
        segments = parse_reference(ref.path, ref.fmt, duration)
        segments, origin, backend_name = _align(audio_path, segments, language, align_backend)
        transcript = Transcript(
            file_id=rec.file_id,
            source_id=spec.id,
            language=language or spec.languages[0],
            backend=backend_name,
            origin=origin,
            duration_s=duration,
            segments=segments,
            notes=f"reference from {ref.how}: {ref.path.name}",
        )
    else:
        if backend is None:
            msg = f"{rec.file_id}: no reference transcript and no ASR backend configured"
            raise BackendUnavailable(msg)
        asr = backend.transcribe(audio_path, language)
        transcript = asr.model_copy(
            update={
                "file_id": rec.file_id,
                "source_id": spec.id,
                "language": language or asr.language,
                "duration_s": max(duration, asr.duration_s),
            }
        )
    if tag:
        tag_music(audio_path, transcript.segments)
    return transcript


def _align(
    audio_path: Path,
    segments: list[Segment],
    language: Language | None,
    align_backend: AlignBackend,
) -> tuple[list[Segment], Literal["reference", "reference_aligned"], str]:
    if align_backend == "mms_fa":
        return align_reference(audio_path, segments, language), "reference_aligned", "mms_fa"
    if align_backend == "uniform":
        return align_uniform(segments), "reference", "uniform"
    return segments, "reference", "none"


def transcribe_source(
    data_root: DataRoot,
    spec: SourceSpec,
    *,
    backend: AsrBackend | None,
    force: bool = False,
    limit: int | None = None,
    align_backend: AlignBackend = "mms_fa",
    tag: bool = True,
    progress: bool = False,
) -> TranscribeSummary:
    """Transcribe every pending file of a source."""
    summary = TranscribeSummary(source_id=spec.id)
    failures_path = data_root.transcripts / spec.id / "failures.jsonl"
    records = list(iter_file_records(data_root, spec.id))
    if limit is not None:
        records = records[:limit]
    iterator = records
    if progress:
        from tqdm import tqdm

        iterator = tqdm(records, desc=f"transcribe {spec.id}", unit="file")
    for rec in iterator:
        out = transcript_path(data_root, spec.id, rec.file_id)
        if out.exists() and not force:
            summary.skipped += 1
            continue
        t0 = time.monotonic()
        try:
            transcript = transcribe_file(
                data_root, spec, rec, backend=backend, align_backend=align_backend, tag=tag
            )
            transcript.save(out)
        except Exception as exc:
            summary.failed += 1
            _record_failure(failures_path, rec, exc)
            continue
        summary.done += 1
        summary.hours += transcript.duration_s / 3600.0
        summary.bump(transcript.origin)
        del t0
    return summary


def _record_failure(path: Path, rec: FileRecord, exc: BaseException) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = "asr" if isinstance(exc, BackendUnavailable) else "pipeline"
    err = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    row = TranscribeFailure(
        file_id=rec.file_id,
        source_id=rec.source_id,
        stage=stage,
        error=err[:2000],
        at=datetime.now(UTC),
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(row.model_dump_json())
        fh.write("\n")
