"""``wakewordworld transcribe`` commands (implemented in M1)."""

from __future__ import annotations

import typer

transcribe_app = typer.Typer(help="Word-level transcription and alignment.", no_args_is_help=True)
