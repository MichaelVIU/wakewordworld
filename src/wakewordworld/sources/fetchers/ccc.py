"""media.ccc.de fetcher (public JSON API)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx

from wakewordworld.manifest.schema import FetchedItem
from wakewordworld.sources.base import (
    item_id_for,
    passes_filters,
    registry,
    resolve_item_licence,
)
from wakewordworld.sources.fetchers import get_json, parse_duration, parse_when, spec_language
from wakewordworld.sources.spec import CccAccess, SourceSpec

__all__ = ["CccFetcher"]

API = "https://api.media.ccc.de/public"
AUDIO_MIMES = ("audio/opus", "audio/mpeg", "audio/ogg")
SUBTITLE_MIMES = ("application/x-subrip", "text/vtt")


def _pick_recording(
    recordings: list[dict[str, Any]], prefer_audio: bool, original_language: str | None = None
) -> dict[str, Any] | None:
    """Pick the recording to download.

    Congress talks ship translated audio tracks as separate recordings, each tagged
    with a ``language``; only recordings in the event's original language (or with no
    language tag) are considered.
    """
    if original_language:
        in_lang = [
            r
            for r in recordings
            if not r.get("language") or str(r.get("language")) == original_language
        ]
        if in_lang:
            recordings = in_lang
    if prefer_audio:
        for mime in AUDIO_MIMES:
            for r in recordings:
                if r.get("mime_type") == mime and r.get("recording_url"):
                    return r
    videos = [
        r
        for r in recordings
        if str(r.get("mime_type", "")).startswith("video/") and r.get("recording_url")
    ]
    if not videos:
        return None
    return min(videos, key=lambda r: (int(r.get("size") or 0), int(r.get("height") or 0)))


@registry.register("ccc")
class CccFetcher:
    """Fetch talks (audio-only when available) from media.ccc.de."""

    def fetch(self, spec: SourceSpec, *, client: httpx.Client) -> Iterator[FetchedItem]:
        """Iterate conferences, then events, then recordings."""
        access = spec.access
        assert isinstance(access, CccAccess)
        language = spec_language(spec)
        acronyms = list(access.conferences)
        if not acronyms:
            listing = get_json(client, f"{API}/conferences")
            acronyms = [c["acronym"] for c in listing.get("conferences", []) if c.get("acronym")]
        for acronym in acronyms:
            conf = get_json(client, f"{API}/conferences/{acronym}")
            for event in conf.get("events") or []:
                if access.languages and event.get("original_language") not in access.languages:
                    continue
                title = str(event.get("title") or event.get("guid"))
                duration = parse_duration(event.get("length"))
                published = parse_when(event.get("date") or event.get("release_date"))
                if not passes_filters(
                    spec.filters, title=title, duration_s=duration, published=published
                ):
                    continue
                guid = event.get("guid")
                if not guid:
                    continue
                details_url = f"{API}/events/{guid}"
                details = get_json(client, details_url)
                recordings = details.get("recordings") or []
                chosen = _pick_recording(
                    recordings, access.prefer_audio_only, event.get("original_language")
                )
                if chosen is None:
                    continue
                media_url = str(chosen["recording_url"])
                licence_text = details.get("license") or event.get("license")
                licence = resolve_item_licence(
                    spec,
                    str(licence_text) if licence_text else None,
                    evidence_quote=f"license: {licence_text}" if licence_text else None,
                    evidence_url=details_url if licence_text else None,
                )
                transcript_url = next(
                    (
                        str(r["recording_url"])
                        for r in recordings
                        if r.get("mime_type") in SUBTITLE_MIMES and r.get("recording_url")
                    ),
                    None,
                )
                persons = details.get("persons") or event.get("persons") or []
                yield FetchedItem(
                    item_id=item_id_for(spec.id, media_url),
                    source_id=spec.id,
                    url=media_url,
                    page_url=details.get("frontend_link") or event.get("frontend_link"),
                    title=title,
                    author=", ".join(str(p) for p in persons) or None,
                    published=published,
                    duration_s=duration,
                    language=language,
                    licence=licence,
                    media_type=str(chosen.get("mime_type") or "") or None,
                    transcript_url=transcript_url,
                    extra={
                        "event": str(acronym),
                        "guid": str(guid),
                        "original_language": str(event.get("original_language") or ""),
                    },
                )
