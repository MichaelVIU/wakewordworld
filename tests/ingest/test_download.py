from __future__ import annotations

import httpx
import pytest
import respx

from wakewordworld.ingest.download import (
    DiskSpaceError,
    ForbiddenItemError,
    check_free_space,
    download_item,
    guess_extension,
)
from wakewordworld.licences import LicenceTier
from wakewordworld.manifest.schema import ItemLicence
from wakewordworld.util.hashing import sha256_bytes
from wakewordworld.util.paths import DataRoot

from .conftest import make_item

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


@pytest.mark.parametrize(
    ("url", "ct", "mt", "ext"),
    [
        ("https://x/y/ep.MP3?x=1", None, None, "mp3"),
        ("https://x/file", "audio/mpeg; charset=binary", None, "mp3"),
        ("https://x/file", None, "audio/ogg", "ogg"),
        ("https://x/data.tar.gz", None, None, "tar.gz"),
        ("https://x/blob", "application/x-parquet", None, "parquet"),
        ("https://x/blob", "application/octet-stream", None, "bin"),
    ],
)
def test_guess_extension(url: str, ct: str | None, mt: str | None, ext: str) -> None:
    assert guess_extension(url, ct, mt) == ext


@respx.mock
def test_download_streams_and_hashes(data_root: DataRoot) -> None:
    item = make_item()
    respx.get(item.url).mock(
        return_value=httpx.Response(200, content=PAYLOAD, headers={"content-type": "audio/mpeg"})
    )
    with httpx.Client() as client:
        path, sha, size = download_item(item, data_root, client=client, min_free_gb=0)
    assert path.exists()
    assert sha == sha256_bytes(PAYLOAD)
    assert size == len(PAYLOAD)
    assert path == data_root.original_path(sha, "mp3")
    # idempotent: second call returns same path
    with httpx.Client() as client:
        path2, sha2, _ = download_item(item, data_root, client=client, min_free_gb=0)
    assert (path2, sha2) == (path, sha)


@respx.mock
def test_download_resumes_partial(data_root: DataRoot) -> None:
    item = make_item()
    part = data_root.cache / "downloads" / f"{item.item_id}.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(PAYLOAD[:1000])
    seen_range: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_range.append(request.headers.get("range"))
        return httpx.Response(206, content=PAYLOAD[1000:])

    respx.get(item.url).mock(side_effect=handler)
    with httpx.Client() as client:
        _, sha, size = download_item(item, data_root, client=client, min_free_gb=0)
    assert seen_range == ["bytes=1000-"]
    assert sha == sha256_bytes(PAYLOAD)
    assert size == len(PAYLOAD)


@respx.mock
def test_download_restarts_when_range_ignored(data_root: DataRoot) -> None:
    item = make_item()
    part = data_root.cache / "downloads" / f"{item.item_id}.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"garbage")
    respx.get(item.url).mock(return_value=httpx.Response(200, content=PAYLOAD))
    with httpx.Client() as client:
        _, sha, _ = download_item(item, data_root, client=client, min_free_gb=0)
    assert sha == sha256_bytes(PAYLOAD)


@respx.mock
def test_download_max_bytes(data_root: DataRoot) -> None:
    item = make_item()
    respx.get(item.url).mock(return_value=httpx.Response(200, content=PAYLOAD))
    with httpx.Client() as client, pytest.raises(ValueError, match="max_bytes"):
        download_item(item, data_root, client=client, max_bytes=10, min_free_gb=0)


def test_forbidden_refused(data_root: DataRoot) -> None:
    item = make_item().model_copy(
        update={"licence": ItemLicence(spdx="NO-AI-USE", tier=LicenceTier.FORBIDDEN, origin="item")}
    )
    with httpx.Client() as client, pytest.raises(ForbiddenItemError):
        download_item(item, data_root, client=client, min_free_gb=0)


def test_disk_guard(data_root: DataRoot) -> None:
    with pytest.raises(DiskSpaceError):
        check_free_space(data_root.root, min_free_gb=1e9)
    check_free_space(data_root.root, min_free_gb=0)
