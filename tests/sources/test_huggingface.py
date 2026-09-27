from __future__ import annotations

import httpx
import pytest
import respx

from wakewordworld.sources.fetchers import fetch_source
from wakewordworld.sources.fetchers.huggingface import infer_language

from .conftest import make_spec

HUB = "https://huggingface.co"


@pytest.mark.parametrize(
    ("path", "lang"),
    [
        ("data/de/train-00000.parquet", "de"),
        ("audio/fr_test_0.tar", "fr"),
        ("en/validated.parquet", "en"),
        ("data/train.parquet", None),
        ("README.md", None),
    ],
)
def test_infer_language(path: str, lang: str | None) -> None:
    assert infer_language(path) == lang


@respx.mock
def test_huggingface_tree_walk_with_config_and_cursor(
    client: httpx.Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_TOKEN", "secret")
    seen_auth: list[str | None] = []

    def root(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("Authorization"))
        if request.url.params.get("cursor") == "abc":
            return httpx.Response(200, json=[{"type": "directory", "path": "data"}])
        return httpx.Response(
            200,
            json=[{"type": "file", "path": "README.md", "size": 10}],
            headers={"Link": f'<{HUB}/api/datasets/o/r/tree/main?cursor=abc>; rel="next"'},
        )

    respx.get(f"{HUB}/api/datasets/o/r/tree/main").mock(side_effect=root)
    respx.get(f"{HUB}/api/datasets/o/r/tree/main/data").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"type": "directory", "path": "data/de"},
                {"type": "directory", "path": "data/fr"},
            ],
        )
    )
    respx.get(f"{HUB}/api/datasets/o/r/tree/main/data/de").mock(
        return_value=httpx.Response(
            200,
            json=[
                {"type": "file", "path": "data/de/test-00000.parquet", "size": 5},
                {"type": "file", "path": "data/de/train-00000.parquet", "size": 5},
            ],
        )
    )
    respx.get(f"{HUB}/api/datasets/o/r/tree/main/data/fr").mock(
        return_value=httpx.Response(
            200, json=[{"type": "file", "path": "data/fr/test-00000.parquet", "size": 5}]
        )
    )
    spec = make_spec(
        {
            "type": "huggingface",
            "repo_id": "o/r",
            "config": "de",
            "split": "test",
            "text_column": "sentence",
        },
        languages=["de", "fr"],
        domain="read",
    )
    items = fetch_source(spec, client=client)
    assert [i.title for i in items] == ["data/de/test-00000.parquet"]
    it = items[0]
    assert it.url == f"{HUB}/datasets/o/r/resolve/main/data/de/test-00000.parquet"
    assert it.language == "de"
    assert it.media_type == "application/x-parquet"
    assert it.extra["text_column"] == "sentence"
    assert seen_auth
    assert all(a == "Bearer secret" for a in seen_auth)
