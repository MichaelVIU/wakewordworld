"""Row models for items, chunks and word occurrences.

These models are the contract between pipeline stages. They are serialised to JSONL
(one object per line, for diffs in git) and to Parquet (for analysis). Field names
are stable; additions bump ``MANIFEST_SCHEMA_VERSION``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from wakewordworld.licences import LicenceTier, normalise_spdx
from wakewordworld.sources.spec import Domain, Language, Microphone

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ChunkRow",
    "FetchedItem",
    "ItemLicence",
    "ManifestRelease",
    "WordOccurrence",
]

MANIFEST_SCHEMA_VERSION = 1

# A canary string embedded in published metadata. Its presence in a model's training
# data is evidence that benchmark material was used for training.
CANARY = "WWW-CANARY-7f3a9c2e-4b1d-4e8a-9f6b-2c5d8e1a3b7f"


class ItemLicence(BaseModel):
    """Licence as resolved for one item (source-level or item-level)."""

    model_config = ConfigDict(extra="forbid")

    spdx: str
    tier: LicenceTier
    origin: Literal["source", "item"]
    evidence_quote: str | None = None
    evidence_url: str | None = None

    @field_validator("spdx")
    @classmethod
    def _normalise(cls, value: str) -> str:
        return normalise_spdx(value)


class FetchedItem(BaseModel):
    """One downloadable recording as discovered by a fetcher (before download)."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(description="Stable id: short hash of source_id + item url.")
    source_id: str
    url: str = Field(description="Direct media URL.")
    page_url: str | None = Field(default=None, description="Human-facing page for attribution.")
    title: str
    author: str | None = None
    published: datetime | date | None = None
    duration_s: float | None = None
    language: Language | None = None
    licence: ItemLicence
    media_type: str | None = None
    transcript_url: str | None = None
    extra: dict[str, str] = Field(default_factory=dict)


class ChunkRow(BaseModel):
    """One evaluation chunk in a release manifest."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    file_id: str
    item_id: str
    source_id: str
    language: Language
    licence_spdx: str
    licence_tier: LicenceTier
    attribution: str
    item_url: str
    page_url: str | None = None
    start_s: float = Field(ge=0, description="Offset of the chunk within the parent file.")
    duration_s: float = Field(gt=0)
    domain: Domain
    microphone: Microphone
    background_tags: list[str] = Field(default_factory=list)
    speakers_est: int | None = None
    has_reference_transcript: bool = False
    transcript_backend: str | None = None
    audio_sha256: str | None = Field(
        default=None, description="Checksum of the chunk FLAC; None for tier B in public manifests."
    )
    engine_training_overlap: list[str] = Field(default_factory=list)
    sealed: bool = False
    canary: str = CANARY


class WordOccurrence(BaseModel):
    """One word token with timing, as stored in the word index."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    word: str = Field(description="Normalised (lower-case, no punctuation) token.")
    raw: str
    start_s: float = Field(ge=0, description="Relative to the chunk start.")
    end_s: float = Field(gt=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    speaker: str | None = None
    utterance_start: bool = False
    utterance_end: bool = False
    backend: str


class ManifestRelease(BaseModel):
    """Release header stored next to the chunk table."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    schema_version: int = MANIFEST_SCHEMA_VERSION
    created_at: datetime
    git_commit: str | None = None
    n_chunks: int
    n_files: int
    hours_by_language: dict[str, float]
    hours_by_tier: dict[str, float]
    sources: list[str]
    notes: str | None = None
