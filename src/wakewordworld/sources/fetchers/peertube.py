"""PeerTube instance fetcher (REST API v1)."""

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
from wakewordworld.sources.fetchers import FetchError, get_json, parse_when, spec_language
from wakewordworld.sources.spec import PeerTubeAccess, SourceSpec

__all__ = ["PEERTUBE_LICENCES", "PeerTubeFetcher"]

# PeerTube licence ids and the CC 4.0 licences the UI links them to.
PEERTUBE_LICENCES: dict[int, tuple[str, str]] = {
    1: ("Attribution", "CC-BY-4.0"),
    2: ("Attribution - Share Alike", "CC-BY-SA-4.0"),
    3: ("Attribution - No Derivatives", "CC-BY-ND-4.0"),
    4: ("Attribution - Non Commercial", "CC-BY-NC-4.0"),
    5: ("Attribution - Non Commercial - Share Alike", "CC-BY-NC-SA-4.0"),
    6: ("Attribution - Non Commercial - No Derivatives", "CC-BY-NC-ND-4.0"),
    7: ("Public Domain Dedication", "CC0-1.0"),
}

PAGE_SIZE = 100


def _pick_file(details: dict[str, Any]) -> tuple[str | None, str | None]:
    """Smallest playable file: lowest resolution id > 0, from files[] or the HLS playlist."""
    candidates: list[dict[str, Any]] = list(details.get("files") or [])
    for playlist in details.get("streamingPlaylists") or []:
        candidates.extend(playlist.get("files") or [])
    best: dict[str, Any] | None = None
    for f in candidates:
        res = (f.get("resolution") or {}).get("id")
        if not isinstance(res, int) or res <= 0 or not f.get("fileDownloadUrl"):
            continue
        if best is None or res < (best["resolution"]["id"]):
            best = f
    if best is None:
        # Audio-only files have resolution id 0 on some instances; accept them last.
        for f in candidates:
            if (f.get("resolution") or {}).get("id") == 0 and f.get("fileDownloadUrl"):
                best = f
                break
    if best is None:
        return None, None
    url = str(best["fileDownloadUrl"])
    return url, "video/mp4" if url.endswith(".mp4") else None


@registry.register("peertube")
class PeerTubeFetcher:
    """Fetch videos with an accepted licence from a PeerTube instance."""

    def _list_url(self, access: PeerTubeAccess) -> str:
        base = str(access.base_url).rstrip("/")
        if access.channel:
            return f"{base}/api/v1/video-channels/{access.channel}/videos"
        if access.account:
            return f"{base}/api/v1/accounts/{access.account}/videos"
        return f"{base}/api/v1/videos"

    def fetch(self, spec: SourceSpec, *, client: httpx.Client) -> Iterator[FetchedItem]:
        """Page through the listing, then fetch details for accepted videos."""
        access = spec.access
        assert isinstance(access, PeerTubeAccess)
        base = str(access.base_url).rstrip("/")
        list_url = self._list_url(access)
        language = spec_language(spec)
        start = 0
        while True:
            params: dict[str, str | int] = {
                "count": PAGE_SIZE,
                "start": start,
                "sort": "-publishedAt",
            }
            if access.local_only:
                params["isLocal"] = "true"
            page = get_json(client, list_url, params=params)
            data = page.get("data") or []
            for video in data:
                lic_id = (video.get("licence") or {}).get("id")
                if lic_id not in access.licence_ids:
                    continue
                lang_id = (video.get("language") or {}).get("id")
                if access.language_ids and lang_id not in access.language_ids:
                    continue
                title = str(video.get("name") or video.get("uuid"))
                duration = float(video["duration"]) if video.get("duration") is not None else None
                published = parse_when(video.get("publishedAt"))
                if not passes_filters(
                    spec.filters, title=title, duration_s=duration, published=published
                ):
                    continue
                uuid = video.get("uuid")
                if not uuid:
                    continue
                details_url = f"{base}/api/v1/videos/{uuid}"
                details = get_json(client, details_url)
                media_url, media_type = _pick_file(details)
                if media_url is None:
                    continue
                label, spdx = PEERTUBE_LICENCES.get(lic_id, (str(lic_id), ""))
                if not spdx:
                    raise FetchError(details_url, f"unknown PeerTube licence id {lic_id}")
                licence = resolve_item_licence(
                    spec,
                    spdx,
                    evidence_quote=f"licence.id={lic_id} ({label})",
                    evidence_url=details_url,
                )
                transcript_url = self._caption_url(client, base, str(uuid), lang_id)
                account = details.get("account") or {}
                yield FetchedItem(
                    item_id=item_id_for(spec.id, media_url),
                    source_id=spec.id,
                    url=media_url,
                    page_url=details.get("url") or f"{base}/w/{uuid}",
                    title=title,
                    author=account.get("displayName") or account.get("name"),
                    published=published,
                    duration_s=duration,
                    language=language,
                    licence=licence,
                    media_type=media_type,
                    transcript_url=transcript_url,
                    extra={"uuid": str(uuid), "peertube_language": str(lang_id or "")},
                )
            if len(data) < PAGE_SIZE:
                break
            start += PAGE_SIZE

    @staticmethod
    def _caption_url(client: httpx.Client, base: str, uuid: str, lang_id: str | None) -> str | None:
        try:
            captions = get_json(client, f"{base}/api/v1/videos/{uuid}/captions")
        except FetchError:
            return None
        data = captions.get("data") or []
        preferred = [c for c in data if (c.get("language") or {}).get("id") == lang_id]
        for c in preferred or data:
            path = c.get("fileUrl") or c.get("captionPath")
            if path:
                return str(path) if str(path).startswith("http") else f"{base}{path}"
        return None
