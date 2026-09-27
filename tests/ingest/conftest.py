from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from wakewordworld.licences import LicenceTier
from wakewordworld.manifest.schema import FetchedItem, ItemLicence
from wakewordworld.util.paths import DataRoot

SR = 16_000


def has_tool(name: str) -> bool:
    return shutil.which(name) is not None or Path(f"/opt/homebrew/bin/{name}").exists()


def tone(duration_s: float, freq: float = 440.0, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def noise(duration_s: float, seed: int, amp: float = 0.2, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(duration_s * sr))).clip(-1, 1).astype(np.float32)


def silence(duration_s: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def write_wav(path: Path, samples: np.ndarray, sr: int = SR, channels: int = 1) -> Path:
    data = samples if channels == 1 else np.stack([samples] * channels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16")
    return path


def write_flac(path: Path, samples: np.ndarray, sr: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sr, subtype="PCM_16", format="FLAC")
    return path


def make_item(source_id: str = "src", url: str = "https://example.org/a.mp3") -> FetchedItem:
    return FetchedItem(
        item_id="item0001",
        source_id=source_id,
        url=url,
        title="Episode 1",
        published=date(2026, 1, 1),
        licence=ItemLicence(spdx="CC-BY-4.0", tier=LicenceTier.A, origin="source"),
    )


@pytest.fixture
def data_root(tmp_path: Path) -> DataRoot:
    root = DataRoot(tmp_path / "data")
    root.ensure()
    return root
