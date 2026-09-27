"""``wakewordworld ingest`` commands (implemented in M1)."""

from __future__ import annotations

import typer

ingest_app = typer.Typer(help="Fetch, download and normalise audio.", no_args_is_help=True)
