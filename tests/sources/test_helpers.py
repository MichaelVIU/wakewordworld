from __future__ import annotations

from datetime import datetime

import pytest

from wakewordworld.sources.base import licence_from_url
from wakewordworld.sources.fetchers import parse_duration, parse_when, registry, strip_html


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("01:02:03", 3723.0),
        ("12:30", 750.0),
        ("95", 95.0),
        (95, 95.0),
        ("1:02:03.5", 3723.5),
        ("", None),
        (None, None),
        ("n/a", None),
    ],
)
def test_parse_duration(raw: str | int | None, expected: float | None) -> None:
    assert parse_duration(raw) == expected


def test_parse_when_rfc2822_and_iso() -> None:
    d = parse_when("Tue, 10 Sep 2024 08:00:00 +0000")
    assert isinstance(d, datetime)
    assert d.year == 2024
    d2 = parse_when("2024-09-10T08:00:00Z")
    assert isinstance(d2, datetime)
    assert d2.month == 9
    assert parse_when("garbage") is None
    assert parse_when(None) is None


@pytest.mark.parametrize(
    ("url", "spdx"),
    [
        ("http://creativecommons.org/licenses/by-sa/4.0/", "CC-BY-SA-4.0"),
        ("https://creativecommons.org/licenses/by/3.0/de/", "CC-BY-3.0-DE"),
        ("https://creativecommons.org/licenses/by-nc-nd/4.0/", "CC-BY-NC-ND-4.0"),
        ("https://creativecommons.org/publicdomain/zero/1.0/", "CC0-1.0"),
        ("https://creativecommons.org/publicdomain/mark/1.0/", "PDM-1.0"),
        ("https://example.org/terms", None),
        (None, None),
    ],
)
def test_licence_from_url(url: str | None, spdx: str | None) -> None:
    assert licence_from_url(url) == spdx


def test_strip_html() -> None:
    assert strip_html("<a href='x'>Jane  Doe</a> &amp; co") == "Jane Doe &amp; co"
    assert strip_html("") is None


def test_all_access_types_registered() -> None:
    assert registry.types() == sorted(
        ["rss", "peertube", "internet_archive", "ccc", "commons", "huggingface", "http_archive"]
    )
