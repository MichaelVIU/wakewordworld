"""Fetcher protocol and registry.

A fetcher turns a :class:`SourceSpec` into a stream of :class:`FetchedItem` objects
without downloading media. It is responsible for reading item-level licence metadata
where the access method exposes it and for applying the spec's filters.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from datetime import date, datetime
from typing import Protocol, runtime_checkable

import httpx

from wakewordworld.licences import LicenceTier, normalise_spdx, tier_for
from wakewordworld.manifest.schema import FetchedItem, ItemLicence
from wakewordworld.sources.spec import SourceFilters, SourceSpec
from wakewordworld.util.hashing import short_id

__all__ = [
    "USER_AGENT",
    "Fetcher",
    "FetcherRegistry",
    "http_client",
    "item_id_for",
    "licence_from_url",
    "passes_filters",
    "resolve_item_licence",
]

USER_AGENT = "wakewordworld/0.1 (+https://github.com/MichaelVIU/wakewordworld)"

_CC_URL_RE = re.compile(
    r"creativecommons\.org/(?:licenses|licences)/(?P<kind>by(?:-nc)?(?:-nd|-sa)?)/(?P<ver>\d\.\d)"
    r"(?:/(?P<juris>[a-z]{2}))?",
    re.IGNORECASE,
)
_CC0_URL_RE = re.compile(r"creativecommons\.org/publicdomain/zero/1\.0", re.IGNORECASE)
_PDM_URL_RE = re.compile(r"creativecommons\.org/publicdomain/mark/1\.0", re.IGNORECASE)


def licence_from_url(url: str | None) -> str | None:
    """Map a Creative Commons licence URL to an SPDX-like identifier.

    Returns ``None`` when the URL is not a recognised CC URL.
    """
    if not url:
        return None
    if _CC0_URL_RE.search(url):
        return "CC0-1.0"
    if _PDM_URL_RE.search(url):
        return "PDM-1.0"
    m = _CC_URL_RE.search(url)
    if not m:
        return None
    kind = m.group("kind").upper()
    ver = m.group("ver")
    juris = m.group("juris")
    spdx = f"CC-{kind}-{ver}"
    if juris and ver.startswith(("2", "3")):
        spdx = f"{spdx}-{juris.upper()}"
    return normalise_spdx(spdx)


def resolve_item_licence(
    spec: SourceSpec,
    item_spdx: str | None,
    evidence_quote: str | None = None,
    evidence_url: str | None = None,
) -> ItemLicence:
    """Combine source-level and item-level licence information.

    Item-level information wins when present. An item-level licence that is *less*
    permissive than the source claim is honoured (tier drops); a *more* permissive
    item-level licence is also honoured, because item metadata is the closer evidence.
    """
    if item_spdx:
        spdx = normalise_spdx(item_spdx)
        tier = tier_for(spdx)
        return ItemLicence(
            spdx=spdx,
            tier=tier,
            origin="item",
            evidence_quote=evidence_quote,
            evidence_url=evidence_url,
        )
    ev = spec.licence.evidence
    return ItemLicence(
        spdx=spec.licence.spdx,
        tier=spec.tier,
        origin="source",
        evidence_quote=ev.quote if ev else None,
        evidence_url=str(ev.url) if ev and ev.url else None,
    )


def item_id_for(source_id: str, url: str) -> str:
    """Stable item id from source id and media URL."""
    return short_id(source_id, url)


def passes_filters(
    filters: SourceFilters,
    *,
    title: str,
    duration_s: float | None,
    published: datetime | date | None,
) -> bool:
    """Apply spec filters to an item; unknown durations pass the duration checks."""
    if duration_s is not None:
        if duration_s < filters.min_duration_s:
            return False
        if filters.max_duration_s is not None and duration_s > filters.max_duration_s:
            return False
    if filters.include_title_regex and not re.search(filters.include_title_regex, title):
        return False
    if filters.exclude_title_regex and re.search(filters.exclude_title_regex, title):
        return False
    if filters.published_after is not None and published is not None:
        pub = published.date() if isinstance(published, datetime) else published
        if pub < filters.published_after:
            return False
    return True


def http_client(timeout: float = 60.0) -> httpx.Client:
    """Shared HTTP client with a descriptive user agent and redirects enabled."""
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
    )


@runtime_checkable
class Fetcher(Protocol):
    """Discover items for a source spec."""

    def fetch(self, spec: SourceSpec, *, client: httpx.Client) -> Iterator[FetchedItem]:
        """Yield items that pass the spec's filters. Must not download media."""
        ...


class FetcherRegistry:
    """Maps an access ``type`` to a fetcher factory."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Fetcher]] = {}

    def register(
        self, access_type: str
    ) -> Callable[[Callable[[], Fetcher]], Callable[[], Fetcher]]:
        """Decorator registering a fetcher factory for an access type."""

        def deco(factory: Callable[[], Fetcher]) -> Callable[[], Fetcher]:
            self._factories[access_type] = factory
            return factory

        return deco

    def for_spec(self, spec: SourceSpec) -> Fetcher:
        """Instantiate the fetcher for a spec."""
        try:
            return self._factories[spec.access.type]()
        except KeyError as exc:
            msg = f"no fetcher registered for access type {spec.access.type!r}"
            raise LookupError(msg) from exc

    def types(self) -> list[str]:
        """Registered access types."""
        return sorted(self._factories)


registry = FetcherRegistry()


def tier_allows_download(licence: ItemLicence) -> bool:
    """Whether an item may be downloaded at all (forbidden items are never fetched)."""
    return licence.tier is not LicenceTier.FORBIDDEN
