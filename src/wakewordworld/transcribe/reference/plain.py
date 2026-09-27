"""Plain-text references: the whole file is one utterance (Common Voice, VoxPopuli)."""

from __future__ import annotations

import re
from pathlib import Path

from wakewordworld.transcribe.schema import Segment

__all__ = ["parse", "parse_text"]

_WS = re.compile(r"\s+")


def parse_text(text: str, duration_s: float) -> list[Segment]:
    """One segment spanning ``[0, duration_s]`` with the given text."""
    clean = _WS.sub(" ", text).strip()
    if not clean:
        return []
    return [Segment(start=0.0, end=max(duration_s, 0.0), text=clean)]


def parse(path: Path, duration_s: float) -> list[Segment]:
    """Read a text file and wrap it in a single segment."""
    return parse_text(path.read_text(encoding="utf-8", errors="replace"), duration_s)
