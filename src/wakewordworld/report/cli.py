"""``wakewordworld report`` commands (implemented in M3)."""

from __future__ import annotations

import typer

report_app = typer.Typer(help="Build reports from results.", no_args_is_help=True)
