from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from wakewordworld.hub.zenodo import SANDBOX, ZenodoClient, ZenodoError, deposition_metadata
from wakewordworld.manifest.schema import ManifestRelease


def _release() -> ManifestRelease:
    return ManifestRelease(
        version="0.1.0",
        created_at=datetime.now(UTC),
        n_chunks=10,
        n_files=4,
        hours_by_language={"de": 1.5},
        hours_by_tier={"A": 1.5},
        sources=["kuechenradio"],
    )


def test_metadata_from_cff(tmp_path: Path) -> None:
    cff = tmp_path / "CITATION.cff"
    cff.write_text("title: Foo\nauthors:\n  - name: Bar Team\n", encoding="utf-8")
    meta = deposition_metadata(
        _release(),
        citation_cff=cff,
        github_tag_url="https://gh/tag",
        hf_dataset_url="https://hf/ds",
    )["metadata"]
    assert meta["title"].startswith("Foo")
    assert meta["creators"] == [{"name": "Bar Team"}]
    assert meta["version"] == "0.1.0"
    assert meta["upload_type"] == "dataset"
    assert {r["identifier"] for r in meta["related_identifiers"]} == {
        "https://gh/tag",
        "https://hf/ds",
    }


@respx.mock
def test_client_flow(tmp_path: Path) -> None:
    client = ZenodoClient(token="secret", base_url=SANDBOX)
    respx.post(f"{SANDBOX}/api/deposit/depositions").mock(
        return_value=httpx.Response(
            201, json={"id": 7, "links": {"bucket": f"{SANDBOX}/api/files/abc", "html": "h"}}
        )
    )
    put = respx.put(f"{SANDBOX}/api/files/abc/SHA256SUMS").mock(
        return_value=httpx.Response(201, json={"key": "SHA256SUMS"})
    )
    respx.post(f"{SANDBOX}/api/deposit/depositions/7/actions/publish").mock(
        return_value=httpx.Response(202, json={"id": 7, "doi": "10.5281/zenodo.7"})
    )
    dep = client.create_deposition({"metadata": {}})
    f = tmp_path / "SHA256SUMS"
    f.write_text("x")
    client.upload_file(dep, f)
    assert put.called
    assert put.calls[0].request.headers["Authorization"] == "Bearer secret"
    assert client.publish(7)["doi"].endswith("zenodo.7")


@respx.mock
def test_client_error_hides_token() -> None:
    client = ZenodoClient(token="secret", base_url=SANDBOX)
    respx.post(f"{SANDBOX}/api/deposit/depositions").mock(
        return_value=httpx.Response(403, text="forbidden")
    )
    with pytest.raises(ZenodoError, match="HTTP 403") as exc:
        client.create_deposition({})
    assert "secret" not in str(exc.value)


def test_from_env_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ZENODO_TOKEN", raising=False)
    with pytest.raises(ZenodoError, match="ZENODO_TOKEN"):
        ZenodoClient.from_env()
