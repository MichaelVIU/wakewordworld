"""``wakewordworld eval`` commands (implemented in M2)."""

from __future__ import annotations

import typer

eval_app = typer.Typer(help="Run engines against the benchmark.", no_args_is_help=True)
