from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import httpx
import pytest
import respx

from wakewordworld.engines.models import ModelChecksumError, download_file, hash_models, unzip_model
from wakewordworld.engines.registry import list_engines, load_engine


def test_registry_lists_three_engines() -> None:
    assert list(list_engines()) == ["microwakeword", "openwakeword", "vosk"]


def test_load_unknown_engine() -> None:
    with pytest.raises(LookupError, match="unknown engine"):
        load_engine("nope")


@respx.mock
def test_download_file_verifies_checksum(tmp_path: Path) -> None:
    payload = b"model-bytes" * 1000
    digest = hashlib.sha256(payload).hexdigest()
    respx.get("https://example.org/m.bin").mock(return_value=httpx.Response(200, content=payload))
    dest = tmp_path / "m.bin"
    assert download_file("https://example.org/m.bin", dest, expected_sha256=digest) == digest
    assert dest.read_bytes() == payload
    # cached: no second request needed
    respx.get("https://example.org/m.bin").mock(return_value=httpx.Response(500))
    assert download_file("https://example.org/m.bin", dest, expected_sha256=digest) == digest
    # mismatch -> error, file removed
    respx.get("https://example.org/bad.bin").mock(return_value=httpx.Response(200, content=b"zzz"))
    with pytest.raises(ModelChecksumError):
        download_file("https://example.org/bad.bin", tmp_path / "bad.bin", expected_sha256=digest)
    assert not (tmp_path / "bad.bin").exists()


def test_unzip_and_hash(tmp_path: Path) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("vosk-model-x/am/final.mdl", b"abc")
        zf.writestr("vosk-model-x/conf/model.conf", b"def")
    archive = tmp_path / "x.zip"
    archive.write_bytes(buf.getvalue())
    out = unzip_model(archive, tmp_path)
    assert out == tmp_path / "vosk-model-x"
    assert (out / "am" / "final.mdl").read_bytes() == b"abc"
    assert unzip_model(archive, tmp_path) == out  # idempotent
    hashes = hash_models(
        {"dir": out, "file": out / "am" / "final.mdl", "missing": tmp_path / "nope"}
    )
    assert set(hashes) == {"dir", "file"}
    assert hashes["file"] == hashlib.sha256(b"abc").hexdigest()
