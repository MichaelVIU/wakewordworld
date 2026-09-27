"""``wakewordworld report`` commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from wakewordworld.util.paths import repo_root

report_app = typer.Typer(help="Build reports from results.", no_args_is_help=True)
console = Console()

ResultsOpt = Annotated[
    Path | None, typer.Option("--results", help="Results root (default results/).")
]


def _results(value: Path | None) -> Path:
    return value if value is not None else repo_root() / "results"


@report_app.command("build")
def build(
    out: Annotated[Path, typer.Option("--out", help="Output HTML file.")] = Path("site/index.html"),
    results: ResultsOpt = None,
    title: Annotated[str, typer.Option("--title")] = "WakeWordWorld results",
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Also write a machine-readable leaderboard.")
    ] = None,
    fa_budget: Annotated[float, typer.Option("--fa-budget", help="Headline budget (FA/h).")] = 0.5,
) -> None:
    """Render the static HTML report from all runs under the results root."""
    from wakewordworld.report.build import export_json, load_runs
    from wakewordworld.report.render import render_report

    runs = load_runs(_results(results))
    if not runs:
        console.print("[yellow]no runs found[/yellow]")
    path = render_report(runs, title=title, out_path=out, fa_budget=fa_budget)
    console.print(f"report written to [bold]{path}[/bold] ({len(runs)} runs)")
    if json_out is not None:
        export_json(runs, json_out, fa_budget=fa_budget)
        console.print(f"leaderboard JSON written to [bold]{json_out}[/bold]")


@report_app.command("list")
def list_runs(results: ResultsOpt = None) -> None:
    """List runs under the results root."""
    from wakewordworld.report.build import load_runs

    runs = load_runs(_results(results))
    table = Table(title="Runs")
    for col in ("run", "engine", "manifest", "created", "chunks", "audio h", "verified"):
        table.add_column(col)
    for r in runs:
        table.add_row(
            r.run_id,
            r.engine_label,
            str(r.meta.get("manifest_version", "?")),
            str(r.meta.get("created_at", "?")),
            str(r.meta.get("n_chunks", "?")),
            f"{float(r.meta.get('audio_hours', 0.0)):.1f}",  # type: ignore[arg-type]
            "yes" if r.verified else "no",
        )
    console.print(table)
