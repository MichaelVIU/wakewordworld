from __future__ import annotations

import httpx
import respx

from wakewordworld.licences import LicenceTier
from wakewordworld.sources.fetchers import fetch_source

from .conftest import make_spec

API = "https://commons.wikimedia.org/w/api.php"


@respx.mock
def test_commons_search_and_imageinfo(client: httpx.Client) -> None:
    def search(request: httpx.Request) -> httpx.Response:
        params = request.url.params
        assert params["list"] == "search"
        assert "haswbstatement:P275=" in params["srsearch"]
        if "sroffset" in params and params["sroffset"] == "500":
            return httpx.Response(200, json={"query": {"search": [{"title": "File:B.webm"}]}})
        return httpx.Response(
            200,
            json={
                "continue": {"sroffset": 500},
                "query": {"search": [{"title": "File:A.ogg"}]},
            },
        )

    def imageinfo(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "query": {
                    "pages": {
                        "1": {
                            "pageid": 1,
                            "title": "File:A.ogg",
                            "imageinfo": [
                                {
                                    "url": "https://upload.wikimedia.org/a.ogg",
                                    "descriptionurl": "https://commons.wikimedia.org/wiki/File:A.ogg",
                                    "mime": "audio/ogg",
                                    "duration": 1234.5,
                                    "extmetadata": {
                                        "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                        "LicenseUrl": {
                                            "value": "https://creativecommons.org/licenses/by-sa/4.0"
                                        },
                                        "Artist": {"value": "<a href='x'>Jane</a>"},
                                    },
                                }
                            ],
                        },
                        "2": {
                            "pageid": 2,
                            "title": "File:B.webm",
                            "imageinfo": [
                                {
                                    "url": "https://upload.wikimedia.org/b.webm",
                                    "mime": "video/webm",
                                    "extmetadata": {"LicenseShortName": {"value": "CC BY-NC 2.0"}},
                                }
                            ],
                        },
                    }
                }
            },
        )

    respx.get(API, params__contains={"list": "search"}).mock(side_effect=search)
    respx.get(API, params__contains={"prop": "imageinfo"}).mock(side_effect=imageinfo)
    spec = make_spec(
        {"type": "commons", "search": "filetype:video Wikimania", "licence_qids": ["Q18199165"]},
        languages=["en"],
    )
    items = fetch_source(spec, client=client)
    assert [i.title for i in items] == ["A.ogg", "B.webm"]
    a, b = items
    assert a.licence.spdx == "CC-BY-SA-4.0"
    assert a.licence.tier is LicenceTier.A
    assert a.author == "Jane"
    assert a.duration_s == 1234.5
    assert a.media_type == "audio/ogg"
    assert b.licence.spdx == "CC-BY-NC-2.0"
    assert b.licence.tier is LicenceTier.B
