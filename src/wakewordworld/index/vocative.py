"""Heuristic detection of vocative name use (a name spoken as an address)."""

from __future__ import annotations

import polars as pl

from wakewordworld.index.lexicon import NameLexicon
from wakewordworld.sources.spec import Language

__all__ = ["flag_vocatives"]


def flag_vocatives(
    df: pl.DataFrame, *, lexicon: NameLexicon | None = None, pause_s: float = 0.3
) -> pl.DataFrame:
    """Add ``is_name`` and ``vocative`` boolean columns.

    A token is flagged vocative when it is a known first name in the row's language
    and it sits at an utterance edge or is separated from its neighbour by at
    least ``pause_s`` seconds on either side. Neighbours are computed within a
    chunk in time order.
    """
    lex = lexicon or NameLexicon()
    if df.height == 0:
        return df.with_columns(pl.lit(False).alias("is_name"), pl.lit(False).alias("vocative"))
    name_sets = {lang.value: lex.names(lang) for lang in Language}
    is_name = pl.struct(["word", "language"]).map_elements(
        lambda s: s["word"] in name_sets.get(s["language"], frozenset()),
        return_dtype=pl.Boolean,
    )
    ordered = (
        df.with_row_index("_row")
        .sort(["chunk_id", "start_s", "_row"])
        .with_columns(
            (pl.col("start_s") - pl.col("end_s").shift(1).over("chunk_id")).alias("_gap_prev"),
            (pl.col("start_s").shift(-1).over("chunk_id") - pl.col("end_s")).alias("_gap_next"),
        )
        .with_columns(is_name.alias("is_name"))
        .with_columns(
            (
                pl.col("is_name")
                & (
                    pl.col("utterance_start")
                    | pl.col("utterance_end")
                    | pl.col("_gap_prev").fill_null(pause_s).ge(pause_s)
                    | pl.col("_gap_next").fill_null(pause_s).ge(pause_s)
                )
            ).alias("vocative")
        )
        .sort("_row")
        .drop(["_row", "_gap_prev", "_gap_next"])
    )
    return ordered
