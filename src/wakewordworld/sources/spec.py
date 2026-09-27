"""Pydantic models for source specifications (``sources/*.yaml``).

A source spec is the unit of provenance: it says where audio comes from, how it is
accessed, under which licence, and which evidence backs that licence claim. Every
ingested item inherits the source licence unless the fetcher finds item-level
licence metadata (for example a per-episode ``podcast:license`` tag or an Internet
Archive ``licenseurl``), in which case the item-level value wins and is recorded
separately.
"""

from __future__ import annotations

import re
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from wakewordworld.licences import LicenceTier, normalise_spdx, tier_for

__all__ = [
    "AccessSpec",
    "CccAccess",
    "CommonsAccess",
    "Domain",
    "HttpArchiveAccess",
    "HuggingFaceAccess",
    "InternetArchiveAccess",
    "Language",
    "LicenceEvidence",
    "LicenceSpec",
    "Microphone",
    "PeerTubeAccess",
    "RssAccess",
    "SourceFilters",
    "SourceSpec",
    "load_source_spec",
    "load_source_specs",
]

_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{1,63}$")


class Language(StrEnum):
    """ISO 639-1 codes of the languages the benchmark covers."""

    EN = "en"
    DE = "de"
    FR = "fr"
    ES = "es"


class Domain(StrEnum):
    """Coarse recording domain used for slicing results."""

    PODCAST = "podcast"
    RADIO = "radio"
    MEETING = "meeting"
    CONFERENCE = "conference"
    PARLIAMENT = "parliament"
    DINNER = "dinner"
    READ = "read"
    INTERVIEW = "interview"
    OTHER = "other"


class Microphone(StrEnum):
    """Microphone / capture condition."""

    CLOSE = "close"
    FAR = "far"
    ARRAY = "array"
    PHONE = "phone"
    BROADCAST = "broadcast"
    UNKNOWN = "unknown"


class LicenceEvidence(BaseModel):
    """Where and how the licence was verified.

    ``quote`` is the verbatim text found (feed tag content, page sentence, API
    field), ``captured_at`` the date it was read. A source without evidence cannot
    be tier A.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["feed_tag", "page", "api_field", "file", "email", "dataset_card"]
    url: HttpUrl | None = None
    quote: str = Field(min_length=3, max_length=2000)
    captured_at: date
    notes: str | None = None


class LicenceSpec(BaseModel):
    """Licence claim for a source, with derived tier."""

    model_config = ConfigDict(extra="forbid")

    spdx: str = Field(description="SPDX-like identifier, e.g. CC-BY-SA-4.0")
    evidence: LicenceEvidence | None = None
    attribution: str | None = Field(
        default=None,
        description="Attribution line template. May use {title}, {author}, {url}, {licence}.",
    )
    per_item: bool = Field(
        default=False,
        description="True when the fetcher reads a licence per item and the source-level "
        "value is only a fallback.",
    )
    tier_override: LicenceTier | None = Field(
        default=None,
        description="Force a tier (only downwards, e.g. to B for PDM-on-own-work).",
    )

    @field_validator("spdx")
    @classmethod
    def _normalise(cls, value: str) -> str:
        return normalise_spdx(value)

    @property
    def tier(self) -> LicenceTier:
        """Effective tier: computed from the identifier, then optionally downgraded."""
        computed = tier_for(self.spdx)
        if self.evidence is None and computed is LicenceTier.A:
            computed = LicenceTier.B
        if self.tier_override is None:
            return computed
        order = {LicenceTier.A: 0, LicenceTier.B: 1, LicenceTier.FORBIDDEN: 2}
        return max(computed, self.tier_override, key=lambda t: order[t])


class RssAccess(BaseModel):
    """A podcast / RSS feed."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["rss"]
    url: HttpUrl
    read_item_licence: bool = Field(
        default=True, description="Parse podcast:license / creativeCommons:license per item."
    )


class PeerTubeAccess(BaseModel):
    """A PeerTube instance (optionally a channel or account)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["peertube"]
    base_url: HttpUrl
    channel: str | None = None
    account: str | None = None
    licence_ids: list[int] = Field(
        default_factory=lambda: [1, 2, 7],
        description="PeerTube licence ids to accept: 1 BY, 2 BY-SA, 3 BY-ND, 4 BY-NC, "
        "5 BY-NC-SA, 6 BY-NC-ND, 7 public domain.",
    )
    language_ids: list[str] = Field(default_factory=list)
    local_only: bool = True


class InternetArchiveAccess(BaseModel):
    """An Internet Archive advancedsearch query."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["internet_archive"]
    query: str = Field(description="Lucene query, e.g. collection:hackerpublicradio")
    licence_urls: list[str] = Field(
        default_factory=list,
        description="Accepted licenseurl values; empty means use the source licence.",
    )
    formats: list[str] = Field(default_factory=lambda: ["VBR MP3", "MP3", "Ogg Vorbis", "FLAC"])
    sort: str = Field(
        default="identifier asc",
        description="advancedsearch sort expression, e.g. 'identifier desc' or 'date desc'.",
    )


class CccAccess(BaseModel):
    """media.ccc.de public API."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["ccc"]
    conferences: list[str] = Field(
        default_factory=list, description="Conference acronyms, e.g. 38c3; empty = all."
    )
    languages: list[str] = Field(default_factory=list, description="e.g. deu, eng")
    prefer_audio_only: bool = True


class HuggingFaceAccess(BaseModel):
    """A Hugging Face dataset repository."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["huggingface"]
    repo_id: str
    config: str | None = None
    split: str | None = None
    revision: str | None = None
    audio_column: str = "audio"
    text_column: str | None = None
    filter_expr: str | None = Field(
        default=None, description="Optional polars expression string applied to metadata."
    )


class HttpArchiveAccess(BaseModel):
    """A tarball / zip published at a fixed URL (OpenSLR, Zenodo, university pages)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["http_archive"]
    urls: list[HttpUrl]
    sha256: dict[str, str] = Field(
        default_factory=dict, description="Expected checksums keyed by file name."
    )
    audio_glob: str = "**/*.wav"
    transcript_glob: str | None = None
    transcript_format: Literal["icsi_mrt", "ami_nxt", "stm", "json", "txt", "none"] = "none"


class CommonsAccess(BaseModel):
    """Wikimedia Commons search."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["commons"]
    search: str = Field(description="srsearch string, e.g. 'filetype:video incategory:...'")
    licence_qids: list[str] = Field(
        default_factory=lambda: ["Q20007257", "Q18199165", "Q6938433", "Q14946043"],
        description="Wikidata QIDs of accepted licences (P275).",
    )
    language_qid: str | None = Field(default=None, description="P407 language QID.")


AccessSpec = Annotated[
    RssAccess
    | PeerTubeAccess
    | InternetArchiveAccess
    | CccAccess
    | HuggingFaceAccess
    | HttpArchiveAccess
    | CommonsAccess,
    Field(discriminator="type"),
]


class SourceFilters(BaseModel):
    """Item-level filters applied at fetch time."""

    model_config = ConfigDict(extra="forbid")

    min_duration_s: float = 60.0
    max_duration_s: float | None = None
    include_title_regex: str | None = None
    exclude_title_regex: str | None = None
    max_items: int | None = Field(default=None, description="Cap for development slices.")
    published_after: date | None = None


class SourceSpec(BaseModel):
    """One source of benchmark audio."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    languages: list[Language] = Field(min_length=1)
    domain: Domain
    microphone: Microphone = Microphone.UNKNOWN
    access: AccessSpec
    licence: LicenceSpec
    filters: SourceFilters = Field(default_factory=SourceFilters)
    expected_hours: float | None = None
    has_transcripts: bool = False
    background_tags: list[str] = Field(
        default_factory=list,
        description="Background typically present: music, audience, kitchen, street, tv, none.",
    )
    engine_training_overlap: list[str] = Field(
        default_factory=list,
        description="Engine ids known to have trained on this source (from their docs).",
    )
    homepage: HttpUrl | None = None
    notes: str | None = None

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not _SOURCE_ID_RE.match(value):
            msg = f"source id {value!r} must match {_SOURCE_ID_RE.pattern}"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _tier_a_needs_evidence(self) -> SourceSpec:
        # A tier-A licence without evidence silently drops to tier B; require an
        # explanatory note so the demotion is visible in the spec itself.
        if (
            self.licence.evidence is None
            and tier_for(self.licence.spdx) is LicenceTier.A
            and not self.notes
        ):
            msg = (
                f"source {self.id}: tier-A licence {self.licence.spdx} without evidence "
                "is demoted to tier B; add evidence or a note explaining why"
            )
            raise ValueError(msg)
        return self

    @property
    def tier(self) -> LicenceTier:
        """Effective licence tier of the source."""
        return self.licence.tier


def load_source_spec(path: Path) -> SourceSpec:
    """Load and validate a single source spec file."""
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        msg = f"{path}: top level must be a mapping"
        raise ValueError(msg)
    spec = SourceSpec.model_validate(raw)
    if spec.id != path.stem:
        msg = f"{path}: id {spec.id!r} must equal file stem {path.stem!r}"
        raise ValueError(msg)
    return spec


def load_source_specs(directory: Path) -> list[SourceSpec]:
    """Load all ``*.yaml`` specs in a directory, sorted by id."""
    specs = [load_source_spec(p) for p in sorted(directory.glob("*.yaml"))]
    seen: set[str] = set()
    for spec in specs:
        if spec.id in seen:
            msg = f"duplicate source id {spec.id}"
            raise ValueError(msg)
        seen.add(spec.id)
    return specs
