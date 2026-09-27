"""Parsers for reference transcripts shipped with a source."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args

from wakewordworld.transcribe.reference import icsi_mrt, plain, webvtt
from wakewordworld.transcribe.schema import Segment

__all__ = ["ReferenceFormat", "parse_reference", "sniff_format"]

ReferenceFormat = Literal["icsi_mrt", "vtt", "srt", "txt"]

_BY_EXT: dict[str, ReferenceFormat] = {
    ".mrt": "icsi_mrt",
    ".vtt": "vtt",
    ".srt": "srt",
    ".txt": "txt",
}


def sniff_format(path: Path) -> ReferenceFormat | None:
    """Guess the reference format from the file extension."""
    return _BY_EXT.get(path.suffix.lower())


def parse_reference(
    path: Path, fmt: ReferenceFormat | None = None, duration_s: float | None = None
) -> list[Segment]:
    """Parse a reference transcript into utterance segments without word timings.

    Args:
        path: The transcript file.
        fmt: Explicit format; sniffed from the extension when ``None``.
        duration_s: Needed for ``txt`` (one segment covering the whole file).
    """
    fmt = fmt or sniff_format(path)
    if fmt is None:
        msg = f"cannot determine reference format of {path.name}"
        raise ValueError(msg)
    if fmt not in get_args(ReferenceFormat):
        msg = f"unknown reference format {fmt!r}"
        raise ValueError(msg)
    if fmt == "icsi_mrt":
        return icsi_mrt.parse(path)
    if fmt in ("vtt", "srt"):
        return webvtt.parse(path)
    if duration_s is None:
        msg = "duration_s is required for plain-text references"
        raise ValueError(msg)
    return plain.parse(path, duration_s)
