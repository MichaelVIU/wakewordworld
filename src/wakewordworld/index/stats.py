"""Statistics over the word index: name tables, word queries, candidate selection."""

from __future__ import annotations

import polars as pl

from wakewordworld.index.lexicon import NameLexicon
from wakewordworld.index.vocative import flag_vocatives
from wakewordworld.ingest.records import ChunkRecord, read_jsonl
from wakewordworld.sources.spec import Language
from wakewordworld.util.paths import DataRoot

__all__ = ["candidate_wake_words", "chunk_file_map", "name_table", "word_query"]


def chunk_file_map(data_root: DataRoot) -> pl.DataFrame:
    """``chunk_id -> file_id`` for all sources with chunk records."""
    rows: list[dict[str, str]] = []
    for p in sorted(data_root.chunks.glob("*/chunks.jsonl")):
        rows.extend(
            {"chunk_id": c.chunk_id, "file_id": c.file_id} for c in read_jsonl(p, ChunkRecord)
        )
    return pl.DataFrame(rows, schema={"chunk_id": pl.Utf8(), "file_id": pl.Utf8()})


def name_table(
    df: pl.DataFrame,
    language: Language,
    lexicon: NameLexicon | None = None,
    *,
    chunk_files: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Per-name occurrence statistics for one language.

    Columns: ``name, occurrences, vocatives, distinct_chunks, distinct_files,
    distinct_sources``. ``distinct_files`` needs ``chunk_files`` (see
    :func:`chunk_file_map`); without it the column equals ``distinct_chunks``.
    """
    lang_df = df.filter(pl.col("language") == language.value)
    flagged = flag_vocatives(lang_df, lexicon=lexicon).filter(pl.col("is_name"))
    if chunk_files is not None and chunk_files.height:
        flagged = flagged.join(chunk_files, on="chunk_id", how="left").with_columns(
            pl.col("file_id").fill_null(pl.col("chunk_id"))
        )
    else:
        flagged = flagged.with_columns(pl.col("chunk_id").alias("file_id"))
    if flagged.height == 0:
        return pl.DataFrame(
            schema={
                "name": pl.Utf8(),
                "occurrences": pl.UInt32(),
                "vocatives": pl.UInt32(),
                "distinct_chunks": pl.UInt32(),
                "distinct_files": pl.UInt32(),
                "distinct_sources": pl.UInt32(),
            }
        )
    return (
        flagged.group_by("word")
        .agg(
            pl.len().alias("occurrences"),
            pl.col("vocative").sum().cast(pl.UInt32).alias("vocatives"),
            pl.col("chunk_id").n_unique().alias("distinct_chunks"),
            pl.col("file_id").n_unique().alias("distinct_files"),
            pl.col("source_id").n_unique().alias("distinct_sources"),
        )
        .rename({"word": "name"})
        .with_columns(pl.col("occurrences").cast(pl.UInt32))
        .sort(["occurrences", "name"], descending=[True, False])
    )


def word_query(df: pl.DataFrame, word: str, *, language: Language | None = None) -> pl.DataFrame:
    """All occurrences of a normalised word, with source, chunk and timing."""
    q = df.filter(pl.col("word") == word)
    if language is not None:
        q = q.filter(pl.col("language") == language.value)
    return q.select(
        "source_id", "language", "chunk_id", "start_s", "end_s", "raw", "speaker", "confidence"
    ).sort(["source_id", "chunk_id", "start_s"])


def candidate_wake_words(
    table: pl.DataFrame, *, min_occurrences: int = 200, min_files: int = 100
) -> pl.DataFrame:
    """Names that meet the size targets of the plan (occurrences and distinct files)."""
    return table.filter(
        (pl.col("occurrences") >= min_occurrences) & (pl.col("distinct_files") >= min_files)
    )
