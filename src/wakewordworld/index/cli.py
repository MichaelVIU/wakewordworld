"""``wakewordworld index`` commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from wakewordworld.index.build import build_index, load_index
from wakewordworld.index.lexicon import NameLexicon
from wakewordworld.index.phonetic import EspeakMissingError, near_misses
from wakewordworld.index.stats import candidate_wake_words, chunk_file_map, name_table, word_query
from wakewordworld.sources.spec import Language, load_source_specs
from wakewordworld.util.paths import DataRoot, repo_root

index_app = typer.Typer(help="Word index, name lexicon and statistics.", no_args_is_help=True)
console = Console()

DataRootOpt = Annotated[
    Path | None, typer.Option("--data-root", envvar="WWW_DATA_ROOT", help="Data directory.")
]
LangOpt = Annotated[Language | None, typer.Option("--language", "-l", help="Language filter.")]


def _print_df(df: pl.DataFrame, title: str, limit: int | None = None) -> None:
    table = Table(title=title)
    for col in df.columns:
        table.add_column(col)
    rows = df.head(limit) if limit else df
    for row in rows.iter_rows():
        table.add_row(*("" if v is None else str(v) for v in row))
    console.print(table)


@index_app.command("build")
def build(
    source_ids: Annotated[list[str] | None, typer.Argument(help="Source ids.")] = None,
    all_sources: Annotated[bool, typer.Option("--all", help="Build every source.")] = False,
    data_root: DataRootOpt = None,
) -> None:
    """Build the word index for one or more sources."""
    root = DataRoot.resolve(data_root)
    specs = {s.id: s for s in load_source_specs(repo_root() / "sources")}
    ids = list(specs) if all_sources else list(source_ids or [])
    if not ids:
        console.print("[red]give source ids or --all[/red]")
        raise typer.Exit(code=2)
    for sid in ids:
        if sid not in specs:
            console.print(f"[red]unknown source {sid}[/red]")
            raise typer.Exit(code=2)
        out = build_index(root, specs[sid])
        meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
        console.print(
            f"[green]{sid}[/green]: {meta['n_words']} words, {meta['hours_indexed']} h, "
            f"{meta['n_files_missing_transcript']} file(s) without transcript"
        )


@index_app.command("names")
def names(
    language: Annotated[Language, typer.Option("--language", "-l")],
    min_occurrences: Annotated[int, typer.Option("--min-occurrences")] = 1,
    top: Annotated[int, typer.Option("--top")] = 50,
    data_root: DataRootOpt = None,
) -> None:
    """Name frequency table for a language."""
    root = DataRoot.resolve(data_root)
    df = load_index(root)
    table = name_table(df, language, NameLexicon(root), chunk_files=chunk_file_map(root))
    table = table.filter(pl.col("occurrences") >= min_occurrences)
    _print_df(table, f"Names ({language.value})", limit=top)
    cands = candidate_wake_words(table)
    if cands.height:
        console.print(
            f"{cands.height} name(s) meet the v0.1 size targets: {cands['name'].to_list()}"
        )


@index_app.command("query")
def query(
    word: Annotated[str, typer.Argument()],
    language: LangOpt = None,
    top: Annotated[int, typer.Option("--top")] = 50,
    data_root: DataRootOpt = None,
) -> None:
    """List occurrences of a word."""
    root = DataRoot.resolve(data_root)
    res = word_query(load_index(root), word.casefold(), language=language)
    console.print(f"{res.height} occurrence(s) of {word!r}")
    _print_df(res, f"Occurrences of {word!r}", limit=top)


@index_app.command("near-miss")
def near_miss(
    word: Annotated[str, typer.Argument()],
    language: Annotated[Language, typer.Option("--language", "-l")],
    max_distance: Annotated[float, typer.Option("--max-distance")] = 0.34,
    top: Annotated[int, typer.Option("--top")] = 50,
    data_root: DataRootOpt = None,
) -> None:
    """Phonetically close words in the index (confusable negatives)."""
    root = DataRoot.resolve(data_root)
    df = load_index(root).filter(pl.col("language") == language.value)
    vocab = df.get_column("word").unique().to_list() if df.height else []
    try:
        hits = near_misses(
            word.casefold(), vocab, language, max_distance=max_distance, top=top, data_root=root
        )
    except EspeakMissingError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    table = Table(title=f"Near misses of {word!r} ({language.value})")
    table.add_column("word")
    table.add_column("distance")
    for w, d in hits:
        table.add_row(w, f"{d:.2f}")
    console.print(table)


@index_app.command("summary")
def summary(data_root: DataRootOpt = None) -> None:
    """Rows, hours and distinct words per language."""
    root = DataRoot.resolve(data_root)
    df = load_index(root)
    hours: dict[str, float] = {}
    for p in sorted(root.index.glob("*.meta.json")):
        meta = json.loads(p.read_text(encoding="utf-8"))
        hours[meta["source_id"]] = meta["hours_indexed"]
    if df.height == 0:
        console.print("index is empty")
        return
    per_lang = (
        df.group_by("language")
        .agg(
            pl.len().alias("tokens"),
            pl.col("word").n_unique().alias("distinct_words"),
            pl.col("chunk_id").n_unique().alias("chunks"),
            pl.col("source_id").n_unique().alias("sources"),
        )
        .sort("language")
    )
    _print_df(per_lang, "Index summary")
    console.print(f"hours indexed per source: {hours}; total {sum(hours.values()):.2f} h")
