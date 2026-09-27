"""WebVTT and SubRip parsers.

Handles cue timing lines (``HH:MM:SS.mmm --> HH:MM:SS.mmm`` with optional settings),
inline tags (``<c>``, ``<c.colour>``, ``<i>``, ``<b>``, ``<u>``, ``<v Name>``), HTML
entities and rolling captions (YouTube style, where each cue repeats the previous
line). Consecutive cues with identical text are merged into one segment.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from wakewordworld.transcribe.schema import Segment

__all__ = ["parse"]

_TIME_RE = re.compile(
    r"(?P<start>(?:\d{1,2}:)?\d{2}:\d{2}[.,]\d{3})\s*-->\s*(?P<end>(?:\d{1,2}:)?\d{2}:\d{2}[.,]\d{3})"
)
_VOICE_RE = re.compile(r"<v(?:\.[^\s>]+)?\s+([^>]+)>")
_TAG_RE = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _to_seconds(stamp: str) -> float:
    stamp = stamp.replace(",", ".")
    parts = stamp.split(":")
    if len(parts) == 2:
        parts = ["0", *parts]
    h, m, s = parts
    return int(h) * 3600 + int(m) * 60 + float(s)


def _clean(text: str) -> tuple[str, str | None]:
    speaker: str | None = None
    m = _VOICE_RE.search(text)
    if m:
        speaker = m.group(1).strip() or None
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = _WS.sub(" ", text).strip()
    return text, speaker


def parse(path: Path) -> list[Segment]:
    """Parse ``.vtt`` or ``.srt`` into utterance segments."""
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = raw.splitlines()
    cues: list[tuple[float, float, str, str | None]] = []
    i = 0
    n = len(lines)
    while i < n:
        m = _TIME_RE.search(lines[i])
        if not m:
            i += 1
            continue
        start = _to_seconds(m.group("start"))
        end = _to_seconds(m.group("end"))
        i += 1
        body: list[str] = []
        while i < n and lines[i].strip():
            body.append(lines[i])
            i += 1
        text, speaker = _clean(" ".join(body))
        if text and end >= start:
            cues.append((start, end, text, speaker))
    return _merge_rolling(cues)


def _merge_rolling(cues: list[tuple[float, float, str, str | None]]) -> list[Segment]:
    """Collapse repeated text from rolling captions into single segments.

    YouTube auto-captions emit cue A = "line1", cue B = "line1 line2", cue C = "line2".
    We drop text already emitted at the start of the next cue and merge identical
    consecutive cues by extending the previous segment's end time.
    """
    segments: list[Segment] = []
    prev_text = ""
    for start, end, text, speaker in cues:
        if segments and text == prev_text:
            last = segments[-1]
            segments[-1] = last.model_copy(update={"end": max(last.end, end)})
            continue
        new_text = text
        if prev_text and text.startswith(prev_text + " "):
            new_text = text[len(prev_text) :].strip()
        elif prev_text and prev_text.endswith(text):
            # Trailing repeat of the previous line only.
            prev_text = text
            continue
        if new_text:
            segments.append(Segment(start=start, end=end, text=new_text, speaker=speaker))
        prev_text = text
    return segments
