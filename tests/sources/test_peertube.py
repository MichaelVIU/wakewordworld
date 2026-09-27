from __future__ import annotations

import httpx
import respx

from wakewordworld.licences import LicenceTier
from wakewordworld.sources.fetchers import fetch_source

from .conftest import make_spec

BASE = "https://videos.example.org"


def _video(uuid: str, lic: int, lang: str = "fr", duration: int = 3600) -> dict:
    return {
        "uuid": uuid,
        "name": f"Talk {uuid}",
        "duration": duration,
        "publishedAt": "2024-05-01T10:00:00.000Z",
        "licence": {"id": lic, "label": "x"},
        "language": {"id": lang, "label": lang},
    }


def _details(uuid: str) -> dict:
    return {
        "uuid": uuid,
        "url": f"{BASE}/w/{uuid}",
        "account": {"name": "conf", "displayName": "Conf Team"},
        "files": [],
        "streamingPlaylists": [
            {
                "files": [
                    {"resolution": {"id": 1080}, "fileDownloadUrl": f"{BASE}/dl/{uuid}-1080.mp4"},
                    {"resolution": {"id": 360}, "fileDownloadUrl": f"{BASE}/dl/{uuid}-360.mp4"},
                    {"resolution": {"id": 0}, "fileDownloadUrl": f"{BASE}/dl/{uuid}-audio.mp4"},
                ]
            }
        ],
    }


@respx.mock
def test_peertube_filters_by_licence_and_picks_smallest(client: httpx.Client) -> None:
    respx.get(f"{BASE}/api/v1/videos").mock(
        return_value=httpx.Response(
            200,
            json={
                "total": 3,
                "data": [_video("a", 1), _video("b", 5), _video("c", 2, lang="en")],
            },
        )
    )
    respx.get(f"{BASE}/api/v1/videos/a").mock(return_value=httpx.Response(200, json=_details("a")))
    respx.get(f"{BASE}/api/v1/videos/a/captions").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"language": {"id": "en"}, "captionPath": "/lazy-static/captions/a-en.vtt"},
                    {"language": {"id": "fr"}, "captionPath": "/lazy-static/captions/a-fr.vtt"},
                ]
            },
        )
    )
    spec = make_spec(
        {"type": "peertube", "base_url": BASE, "licence_ids": [1, 2, 7], "language_ids": ["fr"]},
        languages=["fr"],
    )
    items = fetch_source(spec, client=client)
    assert [i.extra["uuid"] for i in items] == ["a"]
    it = items[0]
    assert it.url == f"{BASE}/dl/a-360.mp4"
    assert it.licence.spdx == "CC-BY-4.0"
    assert it.licence.tier is LicenceTier.A
    assert it.licence.evidence_quote == "licence.id=1 (Attribution)"
    assert it.transcript_url == f"{BASE}/lazy-static/captions/a-fr.vtt"
    assert it.author == "Conf Team"
    assert it.language == "fr"


@respx.mock
def test_peertube_pagination_and_channel(client: httpx.Client) -> None:
    page1 = [_video(f"v{i}", 2) for i in range(100)]
    page2 = [_video("last", 2)]
    route = respx.get(f"{BASE}/api/v1/video-channels/chan/videos")
    route.side_effect = [
        httpx.Response(200, json={"total": 101, "data": page1}),
        httpx.Response(200, json={"total": 101, "data": page2}),
    ]
    respx.get(url__regex=rf"{BASE}/api/v1/videos/[^/]+$").mock(
        side_effect=lambda req: httpx.Response(200, json=_details(req.url.path.rsplit("/", 1)[-1]))
    )
    respx.get(url__regex=rf"{BASE}/api/v1/videos/[^/]+/captions").mock(
        return_value=httpx.Response(404)
    )
    spec = make_spec({"type": "peertube", "base_url": BASE, "channel": "chan"}, languages=["fr"])
    items = fetch_source(spec, client=client)
    assert len(items) == 101
    assert route.call_count == 2
    assert items[-1].transcript_url is None
