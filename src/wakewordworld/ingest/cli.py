"""``wakewordworld ingest`` commands."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from wakewordworld.ingest.download import DiskSpaceError
from wakewordworld.ingest.pipeline import IngestOptions, IngestPipeline, SourceStatus
from wakewordworld.sources.spec import SourceSpec, load_source_specs
from wakewordworld.util.paths import DataRoot, repo_root

ingest_app = typer.Typer(help="Fetch, download and normalise audio.", no_args_is_help=True)
console = Console()

SourceIdsArg = Annotated[list[str] | None, typer.Argument(help="Source ids (default: --all).")]
AllOpt = Annotated[bool, typer.Option("--all", help="Run for every source spec.")]
DataRootOpt = Annotated[
    Path | None,
    typer.Option("--data-root", envvar="WWW_DATA_ROOT", help="Data directory (default ./data)."),
]
SourcesDirOpt = Annotated[
    Path | None, typer.Option("--sources-dir", help="Directory with source YAML files.")
]
MaxItemsOpt = Annotated[int | None, typer.Option("--max-items", help="Cap items per source.")]
MaxHoursOpt = Annotated[
    float | None, typer.Option("--max-hours", help="Stop normalising once this many hours exist.")
]
MaxRowsOpt = Annotated[
    int | None, typer.Option("--max-rows-per-item", help="Row cap per parquet item.")
]
MinFreeOpt = Annotated[float, typer.Option("--min-free-gb", help="Refuse to download below.")]
KeepGoingOpt = Annotated[bool, typer.Option("--keep-going", help="Exit 0 despite failures.")]
NoMeasureOpt = Annotated[
    bool, typer.Option("--no-measure", help="Skip loudness/clipping analysis.")
]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=False)],
        force=True,
    )


def _select(source_ids: list[str] | None, all_: bool, sources_dir: Path | None) -> list[SourceSpec]:
    directory = sources_dir if sources_dir is not None else repo_root() / "sources"
    specs = load_source_specs(directory)
    if all_ or not source_ids:
        if not all_ and not source_ids:
            console.print("[yellow]no source ids given; use --all to run every source[/yellow]")
            raise typer.Exit(code=2)
        return specs
    by_id = {s.id: s for s in specs}
    missing = [s for s in source_ids if s not in by_id]
    if missing:
        console.print(f"[red]unknown source id(s):[/red] {', '.join(missing)}")
        raise typer.Exit(code=2)
    return [by_id[s] for s in source_ids]


def _pipeline(
    specs: list[SourceSpec],
    data_root: Path | None,
    *,
    max_items: int | None = None,
    max_hours: float | None = None,
    max_rows: int | None = 500,
    min_free_gb: float = 5.0,
    measure: bool = True,
) -> IngestPipeline:
    return IngestPipeline(
        DataRoot.resolve(data_root),
        specs,
        options=IngestOptions(
            max_items=max_items,
            max_hours=max_hours,
            max_rows_per_item=max_rows,
            min_free_gb=min_free_gb,
            measure_quality=measure,
        ),
    )


def _print_status(rows: list[SourceStatus]) -> None:
    table = Table(title="Ingest status")
    for col in ("source", "items", "downloaded", "files", "dups", "hours", "chunks", "failures"):
        table.add_column(col, justify="right" if col != "source" else "left")
    for s in rows:
        table.add_row(
            s.source_id,
            str(s.items),
            str(s.downloaded),
            str(s.files),
            str(s.duplicates),
            f"{s.hours:.2f}",
            str(s.chunks),
            f"[red]{s.failures}[/red]" if s.failures else "0",
        )
    console.print(table)


def _finish(rows: list[SourceStatus], keep_going: bool) -> None:
    _print_status(rows)
    if any(r.failures for r in rows) and not keep_going:
        raise typer.Exit(code=1)


def _stage(
    name: str,
    source_ids: list[str] | None,
    all_: bool,
    sources_dir: Path | None,
    data_root: Path | None,
    keep_going: bool,
    *,
    verbose: bool = False,
    **opts: object,
) -> None:
    _setup_logging(verbose)
    specs = _select(source_ids, all_, sources_dir)
    pipe = _pipeline(specs, data_root, **opts)  # type: ignore[arg-type]
    rows: list[SourceStatus] = []
    for spec in specs:
        console.rule(f"{name}: {spec.id}")
        try:
            getattr(pipe, name)(spec.id)
        except DiskSpaceError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=3) from exc
        rows.append(pipe.status(spec.id))
    _finish(rows, keep_going)


@ingest_app.command("fetch")
def fetch(
    source_ids: SourceIdsArg = None,
    all_: AllOpt = False,
    max_items: MaxItemsOpt = None,
    data_root: DataRootOpt = None,
    sources_dir: SourcesDirOpt = None,
    keep_going: KeepGoingOpt = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Discover items (no media download)."""
    _stage(
        "fetch",
        source_ids,
        all_,
        sources_dir,
        data_root,
        keep_going,
        verbose=verbose,
        max_items=max_items,
    )


@ingest_app.command("download")
def download(
    source_ids: SourceIdsArg = None,
    all_: AllOpt = False,
    max_items: MaxItemsOpt = None,
    max_hours: MaxHoursOpt = None,
    min_free_gb: MinFreeOpt = 5.0,
    data_root: DataRootOpt = None,
    sources_dir: SourcesDirOpt = None,
    keep_going: KeepGoingOpt = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Download originals for fetched items."""
    _stage(
        "download",
        source_ids,
        all_,
        sources_dir,
        data_root,
        keep_going,
        verbose=verbose,
        max_items=max_items,
        max_hours=max_hours,
        min_free_gb=min_free_gb,
    )


@ingest_app.command("normalize")
def normalize(
    source_ids: SourceIdsArg = None,
    all_: AllOpt = False,
    max_hours: MaxHoursOpt = None,
    max_rows: MaxRowsOpt = 500,
    no_measure: NoMeasureOpt = False,
    data_root: DataRootOpt = None,
    sources_dir: SourcesDirOpt = None,
    keep_going: KeepGoingOpt = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Transcode originals to 16 kHz mono FLAC, then fingerprint and mark duplicates."""
    _setup_logging(verbose)
    specs = _select(source_ids, all_, sources_dir)
    pipe = _pipeline(
        specs, data_root, max_hours=max_hours, max_rows=max_rows, measure=not no_measure
    )
    rows: list[SourceStatus] = []
    for spec in specs:
        console.rule(f"normalize: {spec.id}")
        pipe.normalize(spec.id)
        pipe.dedupe(spec.id)
        rows.append(pipe.status(spec.id))
    _finish(rows, keep_going)


@ingest_app.command("chunk")
def chunk(
    source_ids: SourceIdsArg = None,
    all_: AllOpt = False,
    data_root: DataRootOpt = None,
    sources_dir: SourcesDirOpt = None,
    keep_going: KeepGoingOpt = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Cut normalised files into evaluation chunks."""
    _stage("chunk", source_ids, all_, sources_dir, data_root, keep_going, verbose=verbose)


@ingest_app.command("run")
def run(
    source_ids: SourceIdsArg = None,
    all_: AllOpt = False,
    max_items: MaxItemsOpt = None,
    max_hours: MaxHoursOpt = None,
    max_rows: MaxRowsOpt = 500,
    min_free_gb: MinFreeOpt = 5.0,
    no_measure: NoMeasureOpt = False,
    no_fetch: Annotated[bool, typer.Option("--no-fetch", help="Skip discovery.")] = False,
    data_root: DataRootOpt = None,
    sources_dir: SourcesDirOpt = None,
    keep_going: KeepGoingOpt = False,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    """Run every stage: fetch, download, normalise, dedupe, chunk."""
    _setup_logging(verbose)
    specs = _select(source_ids, all_, sources_dir)
    pipe = _pipeline(
        specs,
        data_root,
        max_items=max_items,
        max_hours=max_hours,
        max_rows=max_rows,
        min_free_gb=min_free_gb,
        measure=not no_measure,
    )
    rows: list[SourceStatus] = []
    for spec in specs:
        console.rule(f"ingest: {spec.id}")
        try:
            rows.append(pipe.run(spec.id, do_fetch=not no_fetch))
        except DiskSpaceError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=3) from exc
    _finish(rows, keep_going)


@ingest_app.command("status")
def status(
    source_ids: SourceIdsArg = None,
    all_: AllOpt = False,
    data_root: DataRootOpt = None,
    sources_dir: SourcesDirOpt = None,
) -> None:
    """Show per-source counts."""
    specs = _select(source_ids, all_, sources_dir)
    pipe = _pipeline(specs, data_root)
    _print_status([pipe.status(s.id) for s in specs])
