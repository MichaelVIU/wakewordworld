"""Transcript file format (``transcripts/<source_id>/<file_id>.json``).

One transcript per *file* (not per chunk); chunk-relative word timings are derived at
index time from the chunk offsets. Times are seconds from the start of the normalised
file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from wakewordworld.sources.spec import Language

__all__ = ["Segment", "Transcript", "TranscriptOrigin", "Word"]

TranscriptOrigin = Literal["asr", "reference", "reference_aligned"]


class Word(BaseModel):
    """A word token with timing."""

    model_config = ConfigDict(extra="forbid")

    text: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    speaker: str | None = None


class Segment(BaseModel):
    """An utterance-like unit (ASR segment or reference utterance)."""

    model_config = ConfigDict(extra="forbid")

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    speaker: str | None = None
    words: list[Word] = Field(default_factory=list)
    is_music: bool = False


class Transcript(BaseModel):
    """Word-level transcript of one file."""

    model_config = ConfigDict(extra="forbid")

    file_id: str
    source_id: str
    language: Language
    backend: str = Field(description="e.g. faster-whisper:large-v3-turbo, parakeet-mlx:v3, mfa:3.2")
    origin: TranscriptOrigin
    duration_s: float = Field(ge=0)
    segments: list[Segment]
    notes: str | None = None

    def words(self) -> list[Word]:
        """All words in file order."""
        return [w for seg in self.segments for w in seg.words]

    @classmethod
    def load(cls, path: Path) -> Transcript:
        """Read from JSON."""
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        """Write JSON atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(self.model_dump_json(indent=None), encoding="utf-8")
        tmp.replace(path)
