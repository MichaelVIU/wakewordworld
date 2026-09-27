"""ICSI Meeting Corpus ``.mrt`` transcript parser.

The format is XML: ``<Meeting><Transcript><Segment StartTime=".." EndTime=".."
Participant="..">text with <VocalSound .../> <Comment .../> <Pause/>
children</Segment>...``. Non-speech child elements are dropped, their tails kept.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from wakewordworld.transcribe.schema import Segment

__all__ = ["parse"]

_WS = re.compile(r"\s+")


def parse(path: Path) -> list[Segment]:
    """Return utterance segments in time order, skipping empty ones."""
    tree = ET.parse(path)
    segments: list[Segment] = []
    for el in tree.iter("Segment"):
        start = _float_attr(el, "StartTime")
        end = _float_attr(el, "EndTime")
        if start is None or end is None or end < start:
            continue
        text = _text_of(el)
        if not text:
            continue
        speaker = el.get("Participant") or None
        segments.append(Segment(start=start, end=end, text=text, speaker=speaker))
    segments.sort(key=lambda s: (s.start, s.end))
    return segments


def _text_of(el: ET.Element) -> str:
    parts: list[str] = [el.text or ""]
    for child in el:
        # Drop the child's own text (VocalSound descriptions etc.), keep its tail.
        parts.append(child.tail or "")
    return _WS.sub(" ", "".join(parts)).strip()


def _float_attr(el: ET.Element, name: str) -> float | None:
    raw = el.get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None
