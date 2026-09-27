from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "transcribe"
SR = 16_000


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


def write_flac(path: Path, audio: np.ndarray, sr: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sr, format="FLAC", subtype="PCM_16")
    return path


def tone(seconds: float, freq: float = 440.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return 0.3 * np.sin(2 * np.pi * freq * t)


def speechlike(seconds: float, sr: int = SR, seed: int = 0) -> np.ndarray:
    """Alternating short bursts of harmonic tones at varying pitch and noise, with gaps."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * sr), dtype=np.float64)
    pos = 0
    while pos < out.size:
        dur = int(rng.uniform(0.04, 0.12) * sr)
        kind = rng.integers(0, 3)
        seg = np.zeros(dur)
        if kind == 0:
            f0 = rng.uniform(90, 260)
            t = np.arange(dur) / sr
            for h in range(1, 12):
                seg += np.sin(2 * np.pi * f0 * h * t) / h
            seg *= 0.2
        elif kind == 1:
            seg = rng.normal(0, 0.15, dur)
        end = min(pos + dur, out.size)
        out[pos:end] = seg[: end - pos]
        pos = end
    return out
