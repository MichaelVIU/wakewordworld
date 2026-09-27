"""Find target occurrences of a wake word (or phrase) in the word index.

A wake word is a space-separated sequence of normalised tokens ("hey jarvis"). A
phrase matches consecutive index tokens in the same chunk when the gap between tokens
is at most ``max_gap_s``. Near-miss words (from the phonetic index) are returned as
confusable occurrences so false accepts near them can be categorised.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import polars as pl

from wakewordworld.eval.detect import Occurrence

__all__ = ["find_occurrences", "occurrences_by_chunk"]


def _match_phrase(
    words: list[str], starts: list[float], ends: list[float], phrase: list[str], max_gap_s: float
) -> list[tuple[float, float]]:
    n = len(phrase)
    out: list[tuple[float, float]] = []
    for i in range(len(words) - n + 1):
        if words[i] != phrase[0]:
            continue
        ok = True
        for k in range(1, n):
            if words[i + k] != phrase[k] or starts[i + k] - ends[i + k - 1] > max_gap_s:
                ok = False
                break
        if ok:
            out.append((starts[i], ends[i + n - 1]))
    return out


def find_occurrences(
    index: pl.DataFrame,
    wake_word: str,
    *,
    confusables: Iterable[str] = (),
    max_gap_s: float = 0.5,
) -> pl.DataFrame:
    """Return a frame with columns chunk_id, start_s, end_s, confusable.

    ``index`` needs columns chunk_id, word, start_s, end_s (chunk-relative seconds).
    """
    phrase = wake_word.lower().split()
    conf_set = {c.lower() for c in confusables} - {wake_word.lower()}
    rows: list[tuple[str, float, float, bool]] = []
    ordered = index.sort(["chunk_id", "start_s"])
    for (chunk_id,), sub in ordered.group_by("chunk_id", maintain_order=True):
        words = sub.get_column("word").to_list()
        starts = sub.get_column("start_s").to_list()
        ends = sub.get_column("end_s").to_list()
        for s, e in _match_phrase(words, starts, ends, phrase, max_gap_s):
            rows.append((str(chunk_id), float(s), float(e), False))
        if conf_set and len(phrase) == 1:
            for w, s, e in zip(words, starts, ends, strict=True):
                if w in conf_set:
                    rows.append((str(chunk_id), float(s), float(e), True))
    return pl.DataFrame(
        rows,
        schema={
            "chunk_id": pl.Utf8,
            "start_s": pl.Float64,
            "end_s": pl.Float64,
            "confusable": pl.Boolean,
        },
        orient="row",
    )


def occurrences_by_chunk(df: pl.DataFrame) -> Mapping[str, list[Occurrence]]:
    """Group an occurrences frame into per-chunk lists."""
    out: dict[str, list[Occurrence]] = {}
    for row in df.iter_rows(named=True):
        out.setdefault(row["chunk_id"], []).append(
            Occurrence(start_s=row["start_s"], end_s=row["end_s"], confusable=row["confusable"])
        )
    return out
