"""Wikimedia Commons fetcher (MediaWiki API)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx

from wakewordworld.licences import normalise_spdx
from wakewordworld.manifest.schema import FetchedItem
from wakewordworld.sources.base import (
    item_id_for,
    licence_from_url,
    passes_filters,
    registry,
    resolve_item_licence,
)
from wakewordworld.sources.fetchers import get_json, parse_when, spec_language, strip_html
from wakewordworld.sources.spec import CommonsAccess, SourceSpec

__all__ = ["CommonsFetcher"]

API = "https://commons.wikimedia.org/w/api.php"
SEARCH_LIMIT = 500
INFO_BATCH = 50


def _ext_value(extmeta: dict[str, Any], key: str) -> str | None:
    entry = extmeta.get(key)
    if isinstance(entry, dict) and entry.get("value") is not None:
        return str(entry["value"])
    return None


@registry.register("commons")
class CommonsFetcher:
    """Search Commons by licence statement and resolve file URLs."""

    def _search(self, client: httpx.Client, query: str) -> Iterator[str]:
        offset: int | None = 0
        while offset is not None:
            params: dict[str, str | int] = {
                "action": "query",
                "list": "search",
                "srnamespace": 6,
                "srsearch": query,
                "srlimit": SEARCH_LIMIT,
                "sroffset": offset,
                "format": "json",
            }
            data = get_json(client, API, params=params)
            for hit in (data.get("query") or {}).get("search") or []:
                if hit.get("title"):
                    yield str(hit["title"])
            offset = (data.get("continue") or {}).get("sroffset")

    def fetch(self, spec: SourceSpec, *, client: httpx.Client) -> Iterator[FetchedItem]:
        """Yield one item per file with a recognised licence."""
        access = spec.access
        assert isinstance(access, CommonsAccess)
        language = spec_language(spec)
        titles: list[str] = []
        seen: set[str] = set()
        for qid in access.licence_qids:
            query = f"{access.search} haswbstatement:P275={qid}"
            if access.language_qid:
                query += f" haswbstatement:P407={access.language_qid}"
            for title in self._search(client, query):
                if title not in seen:
                    seen.add(title)
                    titles.append(title)
        for i in range(0, len(titles), INFO_BATCH):
            batch = titles[i : i + INFO_BATCH]
            params: dict[str, str] = {
                "action": "query",
                "prop": "imageinfo",
                "iiprop": "url|size|extmetadata|mime",
                "titles": "|".join(batch),
                "format": "json",
            }
            data = get_json(client, API, params=params)
            pages = (data.get("query") or {}).get("pages") or {}
            for page in pages.values():
                infos = page.get("imageinfo") or []
                if not infos:
                    continue
                info = infos[0]
                media_url = info.get("url")
                if not media_url:
                    continue
                extmeta = info.get("extmetadata") or {}
                title = str(page.get("title", "")).removeprefix("File:")
                duration_raw = info.get("duration")
                duration = float(duration_raw) if duration_raw is not None else None
                published = parse_when(_ext_value(extmeta, "DateTimeOriginal"))
                if not passes_filters(
                    spec.filters, title=title, duration_s=duration, published=published
                ):
                    continue
                lic_url = _ext_value(extmeta, "LicenseUrl")
                lic_short = _ext_value(extmeta, "LicenseShortName")
                spdx = licence_from_url(lic_url) or (
                    normalise_spdx(lic_short) if lic_short else None
                )
                licence = resolve_item_licence(
                    spec,
                    spdx,
                    evidence_quote=f"LicenseShortName: {lic_short}; LicenseUrl: {lic_url}",
                    evidence_url=info.get("descriptionurl"),
                )
                yield FetchedItem(
                    item_id=item_id_for(spec.id, str(media_url)),
                    source_id=spec.id,
                    url=str(media_url),
                    page_url=info.get("descriptionurl"),
                    title=title,
                    author=strip_html(_ext_value(extmeta, "Artist")),
                    published=published,
                    duration_s=duration,
                    language=language,
                    licence=licence,
                    media_type=info.get("mime"),
                    extra={"pageid": str(page.get("pageid", ""))},
                )
