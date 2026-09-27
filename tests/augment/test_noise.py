from __future__ import annotations

import io
import zipfile
from pathlib import Path

import httpx
import numpy as np
import pytest
import respx
import soundfile as sf

from wakewordworld.augment.noise import (
    NoiseBank,
    NoiseSet,
    download_noise_set,
    load_audio_16k,
    load_noise_sets,
    rng_for,
)
from wakewordworld.licences import LicenceTier, tier_for
from wakewordworld.util.paths import DataRoot


def test_registry_loads_and_is_licensed() -> None:
    sets = load_noise_sets()
    assert {"demand", "musan", "rirs_noises", "but_reverbdb"} <= set(sets)
    for s in sets.values():
        assert s.licence
        assert s.url.startswith("https://")
        assert s.licence_evidence
        if s.download_default:
            assert tier_for(s.licence) is LicenceTier.A, s.id
            assert s.archives, s.id
    demand = sets["demand"]
    assert demand.categories_for("DKITCHEN_16k", "DKITCHEN/ch01.wav") == {"kitchen", "domestic"}
    musan = sets["musan"]
    assert "music" in musan.categories_for("musan", "musan/music/fma/x.wav")


def _wav_bytes(seconds: float, sr: int) -> bytes:
    buf = io.BytesIO()
    x = (np.random.default_rng(0).standard_normal(int(sr * seconds)) * 0.1).astype(np.float32)
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _fake_set() -> dict[str, NoiseSet]:
    s = NoiseSet.model_validate(
        {
            "id": "fake",
            "name": "Fake",
            "kind": "noise",
            "licence": "CC0-1.0",
            "licence_evidence": "test",
            "url": "https://example.org/fake",
            "attribution": "Fake, CC0",
            "archives": [
                {
                    "url": "https://example.org/k.zip",
                    "file": "K_16k.zip",
                    "categories": ["kitchen"],
                },
                {"url": "https://example.org/c.zip", "file": "C_16k.zip", "categories": ["cafe"]},
            ],
        }
    )
    return {"fake": s}


@respx.mock
def test_download_and_bank(tmp_path: Path) -> None:
    respx.get("https://example.org/k.zip").mock(
        return_value=httpx.Response(200, content=_zip_bytes({"K/ch01.wav": _wav_bytes(1.0, 16000)}))
    )
    respx.get("https://example.org/c.zip").mock(
        return_value=httpx.Response(
            200, content=_zip_bytes({"C/ch01.wav": _wav_bytes(1.0, 48000), "../evil.wav": b"x"})
        )
    )
    root = DataRoot(tmp_path)
    dirs = download_noise_set("fake", root, sets=_fake_set())
    assert [d.name for d in dirs] == ["K_16k", "C_16k"]
    assert not (tmp_path / "evil.wav").exists()
    # second call reuses the extracted directories without network
    respx.reset()
    assert len(download_noise_set("fake", root, sets=_fake_set())) == 2

    bank = NoiseBank(root, ["fake"], sets=_fake_set())
    assert len(bank.clips) == 2
    kitchen = bank.candidates(["kitchen"])
    assert len(kitchen) == 1
    assert kitchen[0].categories == {"kitchen"}
    clip = bank.pick("chunk-a", ["kitchen"], seed="s")
    assert clip is bank.pick("chunk-a", ["kitchen"], seed="s")
    audio = bank.load(bank.candidates(["cafe"])[0])
    assert audio.shape[0] == 16000  # resampled from 48 kHz
    assert audio.dtype == np.float32
    with pytest.raises(LookupError, match="no noise clips"):
        bank.pick("x", ["street"], seed="s")


def test_download_refuses_non_default_sets(tmp_path: Path) -> None:
    sets = _fake_set()
    sets["fake"] = sets["fake"].model_copy(update={"download_default": False})
    with pytest.raises(PermissionError, match="not downloaded by default"):
        download_noise_set("fake", DataRoot(tmp_path), sets=sets)
    with pytest.raises(LookupError, match="unknown noise set"):
        download_noise_set("nope", DataRoot(tmp_path), sets=sets)


def test_load_audio_16k_mono(tmp_path: Path) -> None:
    p = tmp_path / "st.wav"
    x = np.stack([np.ones(800), -np.ones(800)], axis=1).astype(np.float32) * 0.5
    sf.write(p, x, 8000)
    y = load_audio_16k(p)
    assert y.shape[0] == 1600
    assert np.allclose(y, 0.0, atol=1e-6)


def test_rng_for_is_stable() -> None:
    a = rng_for("seed", "c1").integers(0, 1 << 30)
    b = rng_for("seed", "c1").integers(0, 1 << 30)
    c = rng_for("seed", "c2").integers(0, 1 << 30)
    assert a == b
    assert a != c
