"""RSS 2.0 / Atom podcast feed fetcher.

Reads item-level licences from ``podcast:license`` (Podcasting 2.0 namespace) and
``creativeCommons:license``; falls back to the channel-level tag, then to the spec.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass

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
from wakewordworld.sources.fetchers import (
    FetchError,
    get_text,
    parse_duration,
    parse_when,
    spec_language,
    strip_html,
)
from wakewordworld.sources.spec import RssAccess, SourceSpec

__all__ = ["RssFetcher"]

NS = {
    "podcast": "https://podcastindex.org/namespace/1.0",
    "cc": "http://backend.userland.com/creativeCommonsRssModule",
    "itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
    "atom": "http://www.w3.org/2005/Atom",
    "dc": "http://purl.org/dc/elements/1.1/",
    "media": "http://search.yahoo.com/mrss/",
}


@dataclass(frozen=True)
class _Licence:
    spdx: str
    quote: str
    url: str | None


def _licence_of(node: ET.Element) -> _Licence | None:
    """Read a licence from ``podcast:license`` or ``creativeCommons:license`` children."""
    lic = node.find("podcast:license", NS)
    if lic is not None:
        url = lic.get("url")
        text = (lic.text or "").strip()
        spdx = licence_from_url(url) or (normalise_spdx(text) if text else None)
        if spdx:
            quote = f"<podcast:license url={url!r}>{text}</podcast:license>"
            return _Licence(spdx, quote, url)
    cc = node.find("cc:license", NS)
    if cc is not None and cc.text:
        url = cc.text.strip()
        spdx = licence_from_url(url)
        if spdx:
            return _Licence(spdx, f"<creativeCommons:license>{url}</creativeCommons:license>", url)
    return None


def _text(node: ET.Element, *paths: str) -> str | None:
    for path in paths:
        found = node.find(path, NS)
        if found is not None and found.text and found.text.strip():
            return found.text.strip()
    return None


def _rss_items(channel: ET.Element) -> Iterator[tuple[ET.Element, str, str | None, int | None]]:
    for item in channel.findall("item"):
        enc = item.find("enclosure")
        if enc is None or not enc.get("url"):
            enc = item.find("media:content", NS)
        if enc is None or not enc.get("url"):
            continue
        length = enc.get("length") or enc.get("fileSize")
        yield (
            item,
            str(enc.get("url")),
            enc.get("type"),
            int(length) if length and length.isdigit() else None,
        )


def _atom_entries(feed: ET.Element) -> Iterator[tuple[ET.Element, str, str | None, int | None]]:
    for entry in feed.findall("atom:entry", NS):
        for link in entry.findall("atom:link", NS):
            if link.get("rel") == "enclosure" and link.get("href"):
                length = link.get("length")
                yield (
                    entry,
                    str(link.get("href")),
                    link.get("type"),
                    int(length) if length and length.isdigit() else None,
                )
                break


@registry.register("rss")
class RssFetcher:
    """Fetch items from a podcast feed."""

    def fetch(self, spec: SourceSpec, *, client: httpx.Client) -> Iterator[FetchedItem]:
        """Yield one item per enclosure that passes the spec filters."""
        access = spec.access
        assert isinstance(access, RssAccess)
        url = str(access.url)
        body = get_text(client, url)
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            raise FetchError(url, f"feed is not well-formed XML: {exc}") from exc

        is_atom = root.tag == f"{{{NS['atom']}}}feed"
        channel = root if is_atom else root.find("channel")
        if channel is None:
            raise FetchError(url, "no <channel> element in feed")

        channel_licence = _licence_of(channel) if access.read_item_licence else None
        channel_author = _text(channel, "itunes:author", "atom:author/atom:name", "dc:creator")
        language = spec_language(spec)
        entries = _atom_entries(root) if is_atom else _rss_items(channel)

        for node, media_url, media_type, _length in entries:
            title = _text(node, "title", "atom:title") or media_url.rsplit("/", 1)[-1]
            duration = parse_duration(_text(node, "itunes:duration"))
            published = parse_when(
                _text(node, "pubDate", "atom:published", "atom:updated", "dc:date")
            )
            if not passes_filters(
                spec.filters, title=title, duration_s=duration, published=published
            ):
                continue
            item_licence = _licence_of(node) if access.read_item_licence else None
            chosen = item_licence or channel_licence
            licence = resolve_item_licence(
                spec,
                chosen.spdx if chosen else None,
                evidence_quote=chosen.quote if chosen else None,
                evidence_url=(chosen.url if chosen and chosen.url else url) if chosen else None,
            )
            page = _text(node, "link") if not is_atom else None
            if page is None:
                for link in node.findall("atom:link", NS):
                    if link.get("rel") in (None, "alternate") and link.get("href"):
                        page = link.get("href")
                        break
            transcript = node.find("podcast:transcript", NS)
            transcript_url = transcript.get("url") if transcript is not None else None
            yield FetchedItem(
                item_id=item_id_for(spec.id, media_url),
                source_id=spec.id,
                url=media_url,
                page_url=page,
                title=title,
                author=strip_html(
                    _text(node, "itunes:author", "author", "dc:creator", "atom:author/atom:name")
                    or channel_author
                ),
                published=published,
                duration_s=duration,
                language=language,
                licence=licence,
                media_type=media_type,
                transcript_url=transcript_url,
                extra={"guid": _text(node, "guid", "atom:id") or ""},
            )
