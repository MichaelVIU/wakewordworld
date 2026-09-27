from __future__ import annotations

import httpx
import respx

from wakewordworld.licences import LicenceTier
from wakewordworld.sources.fetchers import fetch_source

from .conftest import make_spec


@respx.mock
def test_internet_archive_picks_format_and_item_licence(client: httpx.Client) -> None:
    respx.get("https://archive.org/advancedsearch.php").mock(
        return_value=httpx.Response(
            200,
            json={
                "response": {
                    "docs": [
                        {
                            "identifier": "hpr0001",
                            "title": "Episode one",
                            "creator": ["Someone"],
                            "date": "2010-01-01T00:00:00Z",
                            "licenseurl": "http://creativecommons.org/licenses/by-sa/4.0/",
                            "runtime": "00:20:00",
                        },
                        {"identifier": "hpr0002", "title": "Two", "runtime": "10:00"},
                    ]
                }
            },
        )
    )
    respx.get("https://archive.org/metadata/hpr0001").mock(
        return_value=httpx.Response(
            200,
            json={
                "metadata": {
                    "licenseurl": "http://creativecommons.org/licenses/by-sa/4.0/",
                    "creator": "Someone",
                    "language": "eng",
                },
                "files": [
                    {"name": "hpr0001_spectrogram.png", "format": "PNG"},
                    {"name": "hpr0001.ogg", "format": "Ogg Vorbis", "length": "1200.5"},
                    {"name": "hpr0001.mp3", "format": "VBR MP3", "length": "20:01"},
                ],
            },
        )
    )
    respx.get("https://archive.org/metadata/hpr0002").mock(
        return_value=httpx.Response(
            200,
            json={
                "metadata": {"licenseurl": "http://creativecommons.org/licenses/by-nc-nd/4.0/"},
                "files": [{"name": "two.mp3", "format": "VBR MP3"}],
            },
        )
    )
    spec = make_spec(
        {
            "type": "internet_archive",
            "query": "collection:hackerpublicradio",
            "formats": ["VBR MP3", "Ogg Vorbis"],
        },
        languages=["en"],
    )
    items = fetch_source(spec, client=client)
    assert len(items) == 2
    one, two = items
    assert one.url == "https://archive.org/download/hpr0001/hpr0001.mp3"
    assert one.duration_s == 1201.0
    assert one.page_url == "https://archive.org/details/hpr0001"
    assert one.licence.spdx == "CC-BY-SA-4.0"
    assert one.licence.tier is LicenceTier.A
    assert one.media_type == "audio/mpeg"
    assert one.author == "Someone"
    # item-level NC licence demotes even though the source claims BY-SA
    assert two.licence.spdx == "CC-BY-NC-ND-4.0"
    assert two.licence.tier is LicenceTier.B
    assert two.duration_s == 600.0


@respx.mock
def test_internet_archive_skips_items_without_matching_file(client: httpx.Client) -> None:
    respx.get("https://archive.org/advancedsearch.php").mock(
        return_value=httpx.Response(
            200, json={"response": {"docs": [{"identifier": "x", "title": "X"}]}}
        )
    )
    respx.get("https://archive.org/metadata/x").mock(
        return_value=httpx.Response(200, json={"files": [{"name": "x.pdf", "format": "PDF"}]})
    )
    spec = make_spec({"type": "internet_archive", "query": "q"}, languages=["en"])
    assert fetch_source(spec, client=client) == []
