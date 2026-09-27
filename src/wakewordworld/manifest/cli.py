"""``wakewordworld manifest`` commands (implemented in M1/M3)."""

from __future__ import annotations

import typer

manifest_app = typer.Typer(
    help="Build, validate and freeze release manifests.", no_args_is_help=True
)
