from __future__ import annotations

import httpx
import respx

from wakewordworld.sources.fetchers import fetch_source

from .conftest import make_spec

API = "https://api.media.ccc.de/public"


@respx.mock
def test_ccc_prefers_audio_and_finds_subtitles(client: httpx.Client) -> None:
    respx.get(f"{API}/conferences/38c3").mock(
        return_value=httpx.Response(
            200,
            json={
                "acronym": "38c3",
                "events": [
                    {
                        "guid": "g1",
                        "title": "Ein Vortrag",
                        "original_language": "deu",
                        "length": 3600,
                        "date": "2024-12-27T12:00:00+01:00",
                        "persons": ["Alice", "Bob"],
                        "frontend_link": "https://media.ccc.de/v/38c3-1",
                    },
                    {
                        "guid": "g2",
                        "title": "An English talk",
                        "original_language": "eng",
                        "length": 3600,
                    },
                ],
            },
        )
    )
    respx.get(f"{API}/events/g1").mock(
        return_value=httpx.Response(
            200,
            json={
                "guid": "g1",
                "persons": ["Alice", "Bob"],
                "frontend_link": "https://media.ccc.de/v/38c3-1",
                "recordings": [
                    {
                        "mime_type": "video/mp4",
                        "size": 900,
                        "height": 1080,
                        "recording_url": "https://cdn.media.ccc.de/38c3-1-hd.mp4",
                    },
                    {
                        "mime_type": "audio/opus",
                        "size": 30,
                        "recording_url": "https://cdn.media.ccc.de/38c3-1.opus",
                    },
                    {
                        "mime_type": "application/x-subrip",
                        "recording_url": "https://cdn.media.ccc.de/38c3-1.deu.srt",
                    },
                ],
            },
        )
    )
    spec = make_spec(
        {"type": "ccc", "conferences": ["38c3"], "languages": ["deu"]},
        domain="conference",
    )
    items = fetch_source(spec, client=client)
    assert len(items) == 1
    it = items[0]
    assert it.url == "https://cdn.media.ccc.de/38c3-1.opus"
    assert it.media_type == "audio/opus"
    assert it.transcript_url == "https://cdn.media.ccc.de/38c3-1.deu.srt"
    assert it.author == "Alice, Bob"
    assert it.page_url == "https://media.ccc.de/v/38c3-1"
    assert it.licence.origin == "source"
    assert it.extra["event"] == "38c3"


@respx.mock
def test_ccc_falls_back_to_smallest_video(client: httpx.Client) -> None:
    respx.get(f"{API}/conferences/x").mock(
        return_value=httpx.Response(
            200, json={"events": [{"guid": "g", "title": "T", "length": 100}]}
        )
    )
    respx.get(f"{API}/events/g").mock(
        return_value=httpx.Response(
            200,
            json={
                "recordings": [
                    {"mime_type": "video/mp4", "size": 500, "recording_url": "https://c/hd.mp4"},
                    {"mime_type": "video/mp4", "size": 100, "recording_url": "https://c/sd.mp4"},
                ]
            },
        )
    )
    spec = make_spec({"type": "ccc", "conferences": ["x"]}, domain="conference")
    items = fetch_source(spec, client=client)
    assert items[0].url == "https://c/sd.mp4"


def test_ccc_recording_language_filter() -> None:
    from wakewordworld.sources.fetchers.ccc import _pick_recording

    recs = [
        {"mime_type": "audio/opus", "language": "eng", "recording_url": "https://c/en.opus"},
        {"mime_type": "audio/opus", "language": "deu", "recording_url": "https://c/de.opus"},
        {
            "mime_type": "video/mp4",
            "language": "deu",
            "size": 1,
            "recording_url": "https://c/de.mp4",
        },
    ]
    assert _pick_recording(recs, True, "deu")["recording_url"] == "https://c/de.opus"
    assert _pick_recording(recs, True, "eng")["recording_url"] == "https://c/en.opus"
    # unknown original language: no filtering, first audio wins
    assert _pick_recording(recs, True, None)["recording_url"] == "https://c/en.opus"
    assert _pick_recording(recs, False, "deu")["recording_url"] == "https://c/de.mp4"
