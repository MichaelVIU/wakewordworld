"""Internet Archive fetcher (advancedsearch + per-item metadata)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx

from wakewordworld.manifest.schema import FetchedItem
from wakewordworld.sources.base import (
    item_id_for,
    licence_from_url,
    passes_filters,
    registry,
    resolve_item_licence,
)
from wakewordworld.sources.fetchers import get_json, parse_duration, parse_when, spec_language
from wakewordworld.sources.spec import InternetArchiveAccess, SourceSpec

__all__ = ["InternetArchiveFetcher"]

SEARCH_URL = "https://archive.org/advancedsearch.php"
METADATA_URL = "https://archive.org/metadata/{identifier}"
DOWNLOAD_URL = "https://archive.org/download/{identifier}/{name}"
DETAILS_URL = "https://archive.org/details/{identifier}"
ROWS = 500
_DERIVATIVE_MARKERS = ("_spectrogram", "_thumb", ".afpk", "_files.xml", "_meta.xml")


def _first(value: Any) -> str | None:
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value) if value is not None else None


def _pick_file(files: list[dict[str, Any]], formats: list[str]) -> dict[str, Any] | None:
    for fmt in formats:
        for f in files:
            name = str(f.get("name", ""))
            if f.get("format") == fmt and not any(m in name for m in _DERIVATIVE_MARKERS):
                return f
    return None


@registry.register("internet_archive")
class InternetArchiveFetcher:
    """Fetch audio items matching an advancedsearch query."""

    def fetch(self, spec: SourceSpec, *, client: httpx.Client) -> Iterator[FetchedItem]:
        """Page through search results and resolve one media file per item."""
        access = spec.access
        assert isinstance(access, InternetArchiveAccess)
        language = spec_language(spec)
        page = 1
        while True:
            params: list[tuple[str, str | int]] = [
                ("q", access.query),
                ("rows", ROWS),
                ("page", page),
                ("output", "json"),
                ("sort[]", "identifier asc"),
            ]
            params += [
                ("fl[]", f)
                for f in (
                    "identifier",
                    "title",
                    "creator",
                    "date",
                    "licenseurl",
                    "language",
                    "runtime",
                    "mediatype",
                )
            ]
            result = get_json(client, SEARCH_URL, params=params)
            docs = (result.get("response") or {}).get("docs") or []
            for doc in docs:
                identifier = doc.get("identifier")
                if not identifier:
                    continue
                title = _first(doc.get("title")) or str(identifier)
                published = parse_when(_first(doc.get("date")))
                runtime = parse_duration(_first(doc.get("runtime")))
                if not passes_filters(
                    spec.filters, title=title, duration_s=runtime, published=published
                ):
                    continue
                meta = get_json(client, METADATA_URL.format(identifier=identifier))
                files = meta.get("files") or []
                chosen = _pick_file(files, access.formats)
                if chosen is None:
                    continue
                duration = parse_duration(chosen.get("length")) or runtime
                if not passes_filters(
                    spec.filters, title=title, duration_s=duration, published=published
                ):
                    continue
                item_meta = meta.get("metadata") or {}
                licence_url = _first(item_meta.get("licenseurl")) or _first(doc.get("licenseurl"))
                spdx = licence_from_url(licence_url)
                licence = resolve_item_licence(
                    spec,
                    spdx,
                    evidence_quote=f"licenseurl: {licence_url}" if licence_url else None,
                    evidence_url=METADATA_URL.format(identifier=identifier),
                )
                media_url = DOWNLOAD_URL.format(identifier=identifier, name=chosen["name"])
                yield FetchedItem(
                    item_id=item_id_for(spec.id, media_url),
                    source_id=spec.id,
                    url=media_url,
                    page_url=DETAILS_URL.format(identifier=identifier),
                    title=title,
                    author=_first(item_meta.get("creator")) or _first(doc.get("creator")),
                    published=published,
                    duration_s=duration,
                    language=language,
                    licence=licence,
                    media_type=_mime_for(str(chosen.get("format", ""))),
                    extra={
                        "identifier": str(identifier),
                        "format": str(chosen.get("format", "")),
                        "ia_language": _first(item_meta.get("language")) or "",
                    },
                )
            if len(docs) < ROWS:
                break
            page += 1


def _mime_for(fmt: str) -> str | None:
    fmt_l = fmt.lower()
    if "mp3" in fmt_l:
        return "audio/mpeg"
    if "opus" in fmt_l:
        return "audio/opus"
    if "vorbis" in fmt_l or "ogg" in fmt_l:
        return "audio/ogg"
    if "flac" in fmt_l:
        return "audio/flac"
    if "wav" in fmt_l:
        return "audio/wav"
    return None
