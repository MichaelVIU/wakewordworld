from __future__ import annotations

import httpx
import respx

from wakewordworld.sources.fetchers import fetch_source

from .conftest import make_spec


@respx.mock
def test_http_archive_items(client: httpx.Client) -> None:
    respx.head("https://example.org/corpus.zip").mock(
        return_value=httpx.Response(200, headers={"Content-Length": "12345"})
    )
    respx.head("https://example.org/more.tar.gz").mock(return_value=httpx.Response(500))
    spec = make_spec(
        {
            "type": "http_archive",
            "urls": ["https://example.org/corpus.zip", "https://example.org/more.tar.gz"],
            "sha256": {"corpus.zip": "ab" * 32},
            "audio_glob": "**/*.wav",
            "transcript_glob": "**/*.mrt",
            "transcript_format": "icsi_mrt",
        },
        languages=["en"],
        domain="meeting",
        homepage="https://example.org/",
    )
    items = fetch_source(spec, client=client)
    assert len(items) == 2
    zipped, tarred = items
    assert zipped.media_type == "application/zip"
    assert zipped.extra["sha256"] == "ab" * 32
    assert zipped.extra["content_length"] == "12345"
    assert zipped.extra["transcript_format"] == "icsi_mrt"
    assert zipped.page_url == "https://example.org/"
    assert tarred.media_type == "application/gzip"
    assert tarred.extra["content_length"] == ""
    assert "sha256" not in tarred.extra
