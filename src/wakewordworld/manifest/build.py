"""Assemble release manifest rows from ingestion records, items and source specs."""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime

from wakewordworld.ingest.records import ChunkRecord, FileRecord, read_jsonl
from wakewordworld.licences import LicenceTier
from wakewordworld.manifest.schema import ChunkRow, FetchedItem, ManifestRelease
from wakewordworld.sources.spec import Language, SourceSpec
from wakewordworld.transcribe.schema import Transcript
from wakewordworld.util.paths import DataRoot

__all__ = ["build_manifest", "is_sealed", "render_attribution"]

_FALLBACK_TEMPLATE = "{name}, {licence}, {url}"


class _Missing(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return ""


def render_attribution(spec: SourceSpec, item: FetchedItem, licence: str) -> str:
    """Render the source's attribution template for an item.

    Unknown placeholders render empty; a source without a template falls back to
    ``"{name}, {licence}, {url}"``.
    """
    template = spec.licence.attribution or _FALLBACK_TEMPLATE
    values = _Missing(
        title=item.title,
        author=item.author or "",
        url=item.page_url or item.url,
        licence=licence,
        event=item.extra.get("event", ""),
        name=spec.name,
    )
    text = template.format_map(values)
    # Collapse artefacts of empty placeholders such as ": " or ", ,".
    while ", ," in text:
        text = text.replace(", ,", ",")
    return " ".join(text.split()).strip(" ,:")


def is_sealed(file_id: str, *, fraction: float, seed: str) -> bool:
    """Deterministically decide whether a file belongs to the sealed subset."""
    if fraction <= 0:
        return False
    digest = hashlib.sha256(f"{seed}{file_id}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return value < fraction


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _language_for(spec: SourceSpec, item: FetchedItem) -> Language:
    if item.language is not None:
        return item.language
    if len(spec.languages) == 1:
        return spec.languages[0]
    msg = f"item {item.item_id} of multi-language source {spec.id} has no language"
    raise ValueError(msg)


def _transcript_info(data_root: DataRoot, source_id: str, file_id: str) -> tuple[bool, str | None]:
    p = data_root.transcripts / source_id / f"{file_id}.json"
    if not p.exists():
        return False, None
    t = Transcript.load(p)
    return t.origin != "asr", t.backend


def build_manifest(
    data_root: DataRoot,
    specs: Sequence[SourceSpec],
    *,
    version: str,
    public: bool,
    sealed_fraction: float = 0.10,
    seed: str = "wakewordworld",
    notes: str | None = None,
) -> tuple[list[ChunkRow], list[ChunkRow], ManifestRelease]:
    """Build manifest rows for the given sources.

    Returns ``(rows, sealed_rows, release)``. In public mode forbidden rows are
    dropped, tier B rows lose their audio checksum, and sealed rows are returned
    separately instead of in ``rows``. In internal mode ``sealed_rows`` is empty
    and every row (including sealed ones, flagged) is in ``rows``.
    """
    rows: list[ChunkRow] = []
    sealed_rows: list[ChunkRow] = []
    file_ids: set[str] = set()
    hours_lang: dict[str, float] = {}
    hours_tier: dict[str, float] = {}
    used_sources: list[str] = []

    for spec in specs:
        chunks = list(read_jsonl(data_root.chunks / spec.id / "chunks.jsonl", ChunkRecord))
        if not chunks:
            continue
        files = {
            f.file_id: f for f in read_jsonl(data_root.audio / spec.id / "files.jsonl", FileRecord)
        }
        items = {
            i.item_id: i for i in read_jsonl(data_root.items / f"{spec.id}.jsonl", FetchedItem)
        }
        used_sources.append(spec.id)
        tinfo: dict[str, tuple[bool, str | None]] = {}
        for chunk in chunks:
            frec = files.get(chunk.file_id)
            if frec is None:
                msg = f"chunk {chunk.chunk_id}: no file record {chunk.file_id}"
                raise ValueError(msg)
            item = items.get(frec.item_id)
            if item is None:
                msg = f"file {frec.file_id}: no item record {frec.item_id}"
                raise ValueError(msg)
            if frec.duplicate_of:
                continue
            tier = item.licence.tier
            if public and tier is LicenceTier.FORBIDDEN:
                continue
            if chunk.file_id not in tinfo:
                tinfo[chunk.file_id] = _transcript_info(data_root, spec.id, chunk.file_id)
            has_ref, backend = tinfo[chunk.file_id]
            sealed = is_sealed(chunk.file_id, fraction=sealed_fraction, seed=seed)
            language = _language_for(spec, item)
            row = ChunkRow(
                chunk_id=chunk.chunk_id,
                file_id=chunk.file_id,
                item_id=item.item_id,
                source_id=spec.id,
                language=language,
                licence_spdx=item.licence.spdx,
                licence_tier=tier,
                attribution=render_attribution(spec, item, item.licence.spdx),
                item_url=item.url,
                page_url=item.page_url,
                start_s=chunk.start_s,
                duration_s=chunk.duration_s,
                domain=spec.domain,
                microphone=spec.microphone,
                background_tags=list(spec.background_tags),
                speakers_est=None,
                has_reference_transcript=has_ref,
                transcript_backend=backend,
                audio_sha256=None if (public and tier is not LicenceTier.A) else chunk.audio_sha256,
                engine_training_overlap=list(spec.engine_training_overlap),
                sealed=sealed,
            )
            file_ids.add(chunk.file_id)
            h = chunk.duration_s / 3600.0
            hours_lang[language.value] = hours_lang.get(language.value, 0.0) + h
            hours_tier[tier.value] = hours_tier.get(tier.value, 0.0) + h
            if public and sealed:
                sealed_rows.append(row)
            else:
                rows.append(row)

    rows.sort(key=lambda r: r.chunk_id)
    sealed_rows.sort(key=lambda r: r.chunk_id)
    release = ManifestRelease(
        version=version,
        created_at=datetime.now(UTC),
        git_commit=_git_commit(),
        n_chunks=len(rows),
        n_files=len({r.file_id for r in rows}),
        hours_by_language={k: round(v, 4) for k, v in sorted(hours_lang.items())},
        hours_by_tier={k: round(v, 4) for k, v in sorted(hours_tier.items())},
        sources=sorted(used_sources),
        notes=notes,
    )
    return rows, sealed_rows, release
