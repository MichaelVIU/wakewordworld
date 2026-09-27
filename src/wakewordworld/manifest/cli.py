"""``wakewordworld manifest`` commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from wakewordworld.manifest.build import build_manifest
from wakewordworld.manifest.freeze import freeze
from wakewordworld.manifest.validate import validate_dir
from wakewordworld.sources.spec import load_source_specs
from wakewordworld.util.paths import DataRoot, repo_root

manifest_app = typer.Typer(
    help="Build, validate and freeze release manifests.", no_args_is_help=True
)
console = Console()

DataRootOpt = Annotated[
    Path | None, typer.Option("--data-root", envvar="WWW_DATA_ROOT", help="Data directory.")
]


@manifest_app.command("build")
def build(
    version: Annotated[str, typer.Option("--version", help="Release version, e.g. 0.1.0")],
    public: Annotated[
        bool, typer.Option("--public/--internal", help="Public release masks tier B audio.")
    ] = True,
    sealed_fraction: Annotated[float, typer.Option("--sealed-fraction")] = 0.10,
    force: Annotated[bool, typer.Option("--force")] = False,
    out: Annotated[
        Path | None, typer.Option("--out", help="Output dir (default manifests/).")
    ] = None,
    data_root: DataRootOpt = None,
) -> None:
    """Build and freeze a release manifest from the ingested data."""
    root = DataRoot.resolve(data_root)
    specs = load_source_specs(repo_root() / "sources")
    rows, sealed, release = build_manifest(
        root, specs, version=version, public=public, sealed_fraction=sealed_fraction
    )
    base = out if out is not None else repo_root() / "manifests"
    out_dir = base / (version if public else f"{version}-internal")
    try:
        freeze(rows, release, out_dir, sealed_rows=sealed, force=force)
    except FileExistsError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]wrote {out_dir}[/green]: {release.n_chunks} chunks, {release.n_files} files, "
        f"{len(sealed)} sealed; hours by language {release.hours_by_language}"
    )


@manifest_app.command("validate")
def validate(directory: Annotated[Path, typer.Argument()]) -> None:
    """Validate a frozen release directory."""
    problems = validate_dir(directory)
    for p in problems:
        console.print(f"[red]{p}[/red]")
    if problems:
        raise typer.Exit(code=1)
    console.print(f"[green]{directory} valid[/green]")


@manifest_app.command("stats")
def stats(directory: Annotated[Path, typer.Argument()]) -> None:
    """Print the release header and hours by language and tier."""
    release = json.loads((directory / "release.json").read_text(encoding="utf-8"))
    table = Table(title=f"Release {release['version']}")
    table.add_column("field")
    table.add_column("value")
    for key in ("created_at", "git_commit", "n_chunks", "n_files", "sources"):
        table.add_row(key, str(release.get(key)))
    for lang, h in release["hours_by_language"].items():
        table.add_row(f"hours[{lang}]", f"{h:.2f}")
    for tier, h in release["hours_by_tier"].items():
        table.add_row(f"hours[tier {tier}]", f"{h:.2f}")
    console.print(table)
