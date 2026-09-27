"""Build the word index of a source from its transcripts and chunk records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import polars as pl

from wakewordworld.index.normalize import normalise_token, split_elision
from wakewordworld.ingest.records import ChunkRecord, FileRecord, read_jsonl
from wakewordworld.manifest.schema import WordOccurrence
from wakewordworld.sources.spec import Language, SourceSpec
from wakewordworld.transcribe.schema import Transcript
from wakewordworld.util.paths import DataRoot

__all__ = ["INDEX_COLUMNS", "build_index", "load_index", "occurrences_for_chunk"]

INDEX_COLUMNS: dict[str, pl.DataType] = {
    "chunk_id": pl.Utf8(),
    "word": pl.Utf8(),
    "raw": pl.Utf8(),
    "start_s": pl.Float64(),
    "end_s": pl.Float64(),
    "confidence": pl.Float64(),
    "speaker": pl.Utf8(),
    "utterance_start": pl.Boolean(),
    "utterance_end": pl.Boolean(),
    "backend": pl.Utf8(),
    "source_id": pl.Utf8(),
    "language": pl.Utf8(),
}


def _chunk_language(spec: SourceSpec, item_language: Language | None) -> Language:
    if item_language is not None:
        return item_language
    if len(spec.languages) == 1:
        return spec.languages[0]
    msg = f"source {spec.id} has several languages and the item carries none"
    raise ValueError(msg)


def occurrences_for_chunk(
    transcript: Transcript, chunk: ChunkRecord, language: Language
) -> list[WordOccurrence]:
    """Word occurrences of one chunk, with chunk-relative, clipped times.

    Words are selected by their *start* time falling inside the chunk. Words in
    segments flagged as music are skipped. Elided French/Spanish tokens produce an
    extra occurrence for the part after the apostrophe (same timing).
    """
    lo = chunk.start_s
    hi = chunk.start_s + chunk.duration_s
    out: list[WordOccurrence] = []
    for seg in transcript.segments:
        if seg.is_music or not seg.words:
            continue
        last = len(seg.words) - 1
        for i, w in enumerate(seg.words):
            if not (lo <= w.start < hi):
                continue
            tok = normalise_token(w.text, language)
            if tok is None:
                continue
            start = max(0.0, w.start - lo)
            end = min(chunk.duration_s, max(w.end, w.start) - lo)
            if end <= start:
                end = min(chunk.duration_s, start + 1e-3)
            forms = [tok, *split_elision(tok, language)]
            for form in forms:
                out.append(
                    WordOccurrence(
                        chunk_id=chunk.chunk_id,
                        word=form,
                        raw=w.text,
                        start_s=start,
                        end_s=end,
                        confidence=w.confidence,
                        speaker=w.speaker or seg.speaker,
                        utterance_start=i == 0,
                        utterance_end=i == last,
                        backend=transcript.backend,
                    )
                )
    return out


def _frame(rows: Iterable[WordOccurrence], source_id: str, language: Language) -> pl.DataFrame:
    dicts = [{**r.model_dump(), "source_id": source_id, "language": language.value} for r in rows]
    return pl.DataFrame(dicts, schema=INDEX_COLUMNS)


def build_index(data_root: DataRoot, spec: SourceSpec) -> Path:
    """Build ``index/<source_id>.parquet`` for a source and return its path.

    Requires ``chunks/<source_id>/chunks.jsonl``, ``audio/<source_id>/files.jsonl``,
    ``items/<source_id>.jsonl`` and one transcript per file. Files without a
    transcript are skipped and counted in the meta file.
    """
    chunks = list(read_jsonl(data_root.chunks / spec.id / "chunks.jsonl", ChunkRecord))
    files = {
        f.file_id: f for f in read_jsonl(data_root.audio / spec.id / "files.jsonl", FileRecord)
    }
    item_lang: dict[str, Language | None] = {}
    items_path = data_root.items / f"{spec.id}.jsonl"
    if items_path.exists():
        from wakewordworld.manifest.schema import FetchedItem

        for it in read_jsonl(items_path, FetchedItem):
            item_lang[it.item_id] = it.language

    frames: list[pl.DataFrame] = []
    missing_transcripts: set[str] = set()
    n_chunks_indexed = 0
    seconds_indexed = 0.0
    transcripts: dict[str, Transcript] = {}
    for chunk in chunks:
        tpath = data_root.transcripts / spec.id / f"{chunk.file_id}.json"
        if chunk.file_id not in transcripts:
            if not tpath.exists():
                missing_transcripts.add(chunk.file_id)
                continue
            transcripts[chunk.file_id] = Transcript.load(tpath)
        transcript = transcripts[chunk.file_id]
        frec = files.get(chunk.file_id)
        language = _chunk_language(spec, item_lang.get(frec.item_id) if frec else None)
        rows = occurrences_for_chunk(transcript, chunk, language)
        frames.append(_frame(rows, spec.id, language))
        n_chunks_indexed += 1
        seconds_indexed += chunk.duration_s

    df = pl.concat(frames) if frames else pl.DataFrame(schema=INDEX_COLUMNS)
    data_root.index.mkdir(parents=True, exist_ok=True)
    out = data_root.index / f"{spec.id}.parquet"
    df.write_parquet(out, compression="zstd")
    meta = {
        "source_id": spec.id,
        "n_chunks": len(chunks),
        "n_chunks_indexed": n_chunks_indexed,
        "n_files_missing_transcript": len(missing_transcripts),
        "hours_indexed": round(seconds_indexed / 3600.0, 4),
        "n_words": df.height,
        "n_distinct_words": df.get_column("word").n_unique() if df.height else 0,
    }
    (data_root.index / f"{spec.id}.meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return out


def load_index(data_root: DataRoot, source_ids: Sequence[str] | None = None) -> pl.DataFrame:
    """Load and concatenate index parquet files (all sources by default)."""
    paths = (
        [data_root.index / f"{s}.parquet" for s in source_ids]
        if source_ids is not None
        else sorted(data_root.index.glob("*.parquet"))
    )
    frames = [pl.read_parquet(p) for p in paths if p.exists()]
    if not frames:
        return pl.DataFrame(schema=INDEX_COLUMNS)
    return pl.concat(frames, how="vertical_relaxed")
