"""Concrete fetchers, one per access type.

Importing this package registers every fetcher with :data:`registry`.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import date, datetime
from typing import Any

import httpx

from wakewordworld.manifest.schema import FetchedItem
from wakewordworld.sources.base import http_client, registry
from wakewordworld.sources.spec import Language, SourceSpec

__all__ = [
    "FetchError",
    "fetch_source",
    "get_json",
    "get_text",
    "parse_duration",
    "parse_when",
    "registry",
    "spec_language",
    "strip_html",
]


class FetchError(RuntimeError):
    """A fetcher could not retrieve or parse a remote resource."""

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(f"{reason} ({url})")
        self.url = url
        self.reason = reason


def _request(client: httpx.Client, url: str, **kwargs: Any) -> httpx.Response:
    try:
        resp = client.get(url, **kwargs)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise FetchError(url, f"HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise FetchError(url, f"{type(exc).__name__}: {exc}") from exc
    return resp


def get_json(client: httpx.Client, url: str, **kwargs: Any) -> Any:
    """GET a URL and decode JSON, raising :class:`FetchError` on failure."""
    resp = _request(client, url, **kwargs)
    try:
        return resp.json()
    except ValueError as exc:
        raise FetchError(url, "response is not valid JSON") from exc


def get_text(client: httpx.Client, url: str, **kwargs: Any) -> str:
    """GET a URL and return its text body, raising :class:`FetchError` on failure."""
    return _request(client, url, **kwargs).text


_HMS_RE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)\s*$")
_NUM_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")


def parse_duration(value: str | int | float | None) -> float | None:
    """Parse ``HH:MM:SS``, ``MM:SS`` or plain seconds into seconds."""
    if value is None:
        return None
    if isinstance(value, int | float):
        return float(value) if value >= 0 else None
    text = str(value)
    m = _NUM_RE.match(text)
    if m:
        return float(m.group(1))
    m = _HMS_RE.match(text)
    if m:
        hours = int(m.group(1) or 0)
        return hours * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return None


def parse_when(value: str | None) -> datetime | date | None:
    """Parse an RFC 2822 or ISO 8601 date/time string, ``None`` on failure."""
    if not value:
        return None
    from email.utils import parsedate_to_datetime

    from dateutil import parser as dateparser

    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        pass
    try:
        return dateparser.isoparse(value)
    except (ValueError, OverflowError):
        pass
    try:
        return dateparser.parse(value)
    except (ValueError, OverflowError, TypeError):
        return None


_TAG_RE = re.compile(r"<[^>]+>")


def strip_html(text: str | None) -> str | None:
    """Remove HTML tags and collapse whitespace."""
    if text is None:
        return None
    cleaned = " ".join(_TAG_RE.sub(" ", text).split())
    return cleaned or None


def spec_language(spec: SourceSpec) -> Language | None:
    """The spec's language when it is single-language, else ``None``."""
    return spec.languages[0] if len(spec.languages) == 1 else None


def _capped(items: Iterator[FetchedItem], max_items: int | None) -> Iterator[FetchedItem]:
    if max_items is None:
        yield from items
        return
    for n, item in enumerate(items):
        if n >= max_items:
            break
        yield item


def fetch_source(
    spec: SourceSpec, *, client: httpx.Client | None = None, max_items: int | None = None
) -> list[FetchedItem]:
    """Run the registered fetcher for ``spec`` and return its items as a list.

    Discovery stops as soon as the effective cap (the smaller of ``max_items`` and the
    spec's ``filters.max_items``) is reached, so large catalogues are not enumerated
    for a development slice.
    """
    caps = [c for c in (max_items, spec.filters.max_items) if c is not None]
    cap = min(caps) if caps else None
    fetcher = registry.for_spec(spec)
    if client is None:
        with http_client() as own:
            return list(_capped(fetcher.fetch(spec, client=own), cap))
    return list(_capped(fetcher.fetch(spec, client=client), cap))


# Register all fetchers (import for side effects).
from wakewordworld.sources.fetchers import (  # noqa: E402
    ccc,
    commons,
    http_archive,
    huggingface,
    internet_archive,
    peertube,
    rss,
)

_ = (rss, peertube, internet_archive, ccc, commons, huggingface, http_archive)
