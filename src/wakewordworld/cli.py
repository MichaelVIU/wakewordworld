"""Command line interface.

Sub-commands mirror the pipeline stages: ``sources`` -> ``ingest`` -> ``transcribe``
-> ``index`` -> ``manifest`` -> ``eval`` -> ``report``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from wakewordworld import __version__
from wakewordworld.licences import LicenceTier
from wakewordworld.sources.spec import load_source_specs
from wakewordworld.util.paths import DataRoot, repo_root

app = typer.Typer(
    name="wakewordworld",
    help="Independent wake word benchmark: source registry, ingestion, evaluation, reports.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
sources_app = typer.Typer(help="Source registry commands.", no_args_is_help=True)
app.add_typer(sources_app, name="sources")
console = Console()

DataRootOpt = Annotated[
    Path | None,
    typer.Option("--data-root", envvar="WWW_DATA_ROOT", help="Data directory (default ./data)."),
]
SourcesDirOpt = Annotated[
    Path | None, typer.Option("--sources-dir", help="Directory with source YAML files.")
]


def _sources_dir(value: Path | None) -> Path:
    return value if value is not None else repo_root() / "sources"


@app.callback(invoke_without_command=True)
def _version_callback(
    version: Annotated[
        bool, typer.Option("--version", help="Show version and exit.", is_eager=True)
    ] = False,
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit


@sources_app.command("validate")
def sources_validate(sources_dir: SourcesDirOpt = None) -> None:
    """Validate all source specs; exit non-zero on the first error."""
    directory = _sources_dir(sources_dir)
    try:
        specs = load_source_specs(directory)
    except Exception as exc:
        console.print(f"[red]invalid source spec:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    n_a = sum(1 for s in specs if s.tier is LicenceTier.A)
    console.print(f"[green]{len(specs)} source specs valid[/green] ({n_a} tier A)")


@sources_app.command("list")
def sources_list(sources_dir: SourcesDirOpt = None) -> None:
    """Print a table of sources with licence tier and expected hours."""
    specs = load_source_specs(_sources_dir(sources_dir))
    table = Table(title="Sources")
    for col in ("id", "langs", "domain", "licence", "tier", "access", "hours"):
        table.add_column(col)
    for s in specs:
        table.add_row(
            s.id,
            ",".join(s.languages),
            s.domain,
            s.licence.spdx,
            s.tier,
            s.access.type,
            f"{s.expected_hours:.0f}" if s.expected_hours else "?",
        )
    console.print(table)


@app.command("init-data")
def init_data(data_root: DataRootOpt = None) -> None:
    """Create the data directory layout."""
    root = DataRoot.resolve(data_root)
    root.ensure()
    console.print(f"data root ready at [bold]{root.root}[/bold]")


def _register_stage_commands() -> None:
    """Attach stage sub-apps lazily so optional deps are imported only when used."""
    from wakewordworld.eval.cli import eval_app
    from wakewordworld.index.cli import index_app
    from wakewordworld.ingest.cli import ingest_app
    from wakewordworld.manifest.cli import manifest_app
    from wakewordworld.report.cli import report_app
    from wakewordworld.transcribe.cli import transcribe_app

    app.add_typer(ingest_app, name="ingest")
    app.add_typer(transcribe_app, name="transcribe")
    app.add_typer(index_app, name="index")
    app.add_typer(manifest_app, name="manifest")
    app.add_typer(eval_app, name="eval")
    app.add_typer(report_app, name="report")


_register_stage_commands()
