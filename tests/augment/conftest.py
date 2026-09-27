from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from wakewordworld.augment.noise import NoiseClip, rng_for

SR = 16000


def speech_like(seconds: float, seed: int = 1) -> NDArray[np.float32]:
    """Band-limited noise bursts with silences, roughly speech-shaped in time."""
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    x = np.zeros(n, dtype=np.float64)
    t = 0
    while t < n:
        burst = int(rng.integers(SR // 5, SR))
        gap = int(rng.integers(SR // 10, SR // 2))
        seg = rng.standard_normal(min(burst, n - t))
        # crude low-pass by moving average to concentrate energy below ~4 kHz
        seg = np.convolve(seg, np.ones(4) / 4, mode="same")
        env = np.hanning(seg.shape[0])
        x[t : t + seg.shape[0]] = seg * env * 0.3
        t += burst + gap
    return x.astype(np.float32)


def steady_noise(seconds: float, seed: int = 2) -> NDArray[np.float32]:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(SR * seconds)) * 0.05).astype(np.float32)


class FakeBank:
    """In-memory noise provider with two categories."""

    def __init__(self, kind: str = "noise") -> None:
        self.kind = kind
        self.clips = [
            NoiseClip(Path("/fake/kitchen.wav"), "fake", frozenset({"kitchen"}), "Fake noise, CC0"),
            NoiseClip(Path("/fake/babble.wav"), "fake", frozenset({"babble"}), "Fake noise, CC0"),
        ]

    def pick(self, chunk_id: str, categories: Sequence[str], *, seed: str) -> NoiseClip:
        cands = [c for c in self.clips if not categories or c.categories & set(categories)]
        rng = rng_for(seed, chunk_id)
        return cands[int(rng.integers(0, len(cands)))]

    def load(self, clip: NoiseClip) -> NDArray[np.float32]:
        if self.kind == "rir":
            h = np.zeros(SR // 2, dtype=np.float32)
            h[0] = 1.0
            h[SR // 100] = 0.5
            h[SR // 50] = 0.25
            return h
        seed = 10 if "kitchen" in clip.path.name else 11
        return steady_noise(3.0, seed)


@pytest.fixture
def fake_bank() -> FakeBank:
    return FakeBank()


@pytest.fixture
def fake_rir_bank() -> FakeBank:
    return FakeBank(kind="rir")
