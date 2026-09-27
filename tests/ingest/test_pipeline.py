from __future__ import annotations

import io
from pathlib import Path

import httpx
import pytest
import respx
import soundfile as sf

from wakewordworld.ingest.pipeline import IngestOptions, IngestPipeline
from wakewordworld.licences import LicenceTier
from wakewordworld.manifest.schema import FetchedItem, ItemLicence
from wakewordworld.sources.spec import SourceSpec
from wakewordworld.util.paths import DataRoot

from .conftest import has_tool, noise

pytestmark = pytest.mark.skipif(
    not (has_tool("ffmpeg") and has_tool("fpcalc")), reason="ffmpeg/fpcalc not installed"
)

SPEC = SourceSpec.model_validate(
    {
        "id": "demo",
        "name": "Demo",
        "languages": ["de"],
        "domain": "podcast",
        "access": {"type": "rss", "url": "https://example.org/feed.xml"},
        "licence": {
            "spdx": "CC-BY-4.0",
            "evidence": {
                "type": "page",
                "url": "https://example.org",
                "quote": "CC BY 4.0",
                "captured_at": "2026-09-26",
            },
        },
    }
)


def wav_bytes(seconds: float, seed: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, noise(seconds, seed), 16_000, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def item(n: int, url: str) -> FetchedItem:
    return FetchedItem(
        item_id=f"item{n}",
        source_id="demo",
        url=url,
        title=f"ep {n}",
        licence=ItemLicence(spdx="CC-BY-4.0", tier=LicenceTier.A, origin="source"),
    )


@respx.mock
def test_pipeline_end_to_end_without_fetch(data_root: DataRoot) -> None:
    a = wav_bytes(5.0, 1)
    b = wav_bytes(5.0, 2)
    respx.get("https://example.org/a.wav").mock(return_value=httpx.Response(200, content=a))
    respx.get("https://example.org/a2.wav").mock(return_value=httpx.Response(200, content=a))
    respx.get("https://example.org/b.wav").mock(return_value=httpx.Response(200, content=b))
    respx.get("https://example.org/broken.wav").mock(return_value=httpx.Response(404))

    pipe = IngestPipeline(
        data_root,
        [SPEC],
        options=IngestOptions(min_free_gb=0, measure_quality=False),
        client=httpx.Client(),
    )
    pipe.items.write(
        "demo",
        [
            item(1, "https://example.org/a.wav"),
            item(2, "https://example.org/a2.wav"),
            item(3, "https://example.org/b.wav"),
            item(4, "https://example.org/broken.wav"),
        ],
    )
    status = pipe.run("demo", do_fetch=False)
    assert status.items == 4
    assert status.downloaded == 3  # broken failed
    assert status.failures == 1
    # a and a2 have identical bytes -> same original sha -> only 2 normalised files
    assert status.files == 2
    assert status.duplicates == 0
    assert status.chunks == 2
    assert abs(status.hours - 10 / 3600) < 1e-6
    failures = pipe.failures("demo")
    assert failures[0].stage == "download"
    # rerun is a no-op
    status2 = pipe.run("demo", do_fetch=False)
    assert (status2.files, status2.chunks) == (2, 2)


@respx.mock
def test_pipeline_max_hours_budget(data_root: DataRoot) -> None:
    for n in range(3):
        respx.get(f"https://example.org/{n}.wav").mock(
            return_value=httpx.Response(200, content=wav_bytes(3600.0 / 1000, n))
        )
    pipe = IngestPipeline(
        data_root,
        [SPEC],
        options=IngestOptions(min_free_gb=0, measure_quality=False, max_hours=1.5 / 1000),
        client=httpx.Client(),
    )
    pipe.items.write("demo", [item(n, f"https://example.org/{n}.wav") for n in range(3)])
    status = pipe.run("demo", do_fetch=False)
    assert status.files == 2  # third exceeds the budget


def test_pipeline_fetch_without_fetchers_module_message(
    data_root: DataRoot, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if name == "wakewordworld.sources.fetchers":
            raise ImportError("nope")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    pipe = IngestPipeline(data_root, [SPEC], options=IngestOptions(min_free_gb=0))
    with pytest.raises(RuntimeError, match="fetch_source"):
        pipe.fetch("demo")


def test_unknown_source(data_root: DataRoot) -> None:
    pipe = IngestPipeline(data_root, [SPEC])
    with pytest.raises(LookupError):
        pipe.status("nope")
    assert Path(pipe.items.path("demo")).parent.exists()
