from __future__ import annotations

import numpy as np

from wakewordworld.augment.rir import apply_rir, fft_convolve, normalise_rir

from .conftest import SR, speech_like


def _rir() -> np.ndarray:
    h = np.zeros(SR, dtype=np.float32)
    h[100] = 0.0  # leading silence before the direct path
    h[400] = 1.0
    h[400 + SR // 50] = 0.4
    h[400 + SR // 10] = 0.1
    return h


def test_normalise_aligns_and_unit_direct_energy() -> None:
    h = normalise_rir(_rir())
    win = round(2.5 * SR / 1000)
    peak = int(np.argmax(np.abs(h)))
    assert peak <= win
    assert h.shape[0] == SR - (400 - win)
    assert abs(float(np.sum(h[: 2 * win + 1].astype(np.float64) ** 2)) - 1.0) < 1e-6


def test_fft_convolve_matches_numpy() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(5000).astype(np.float32)
    h = rng.standard_normal(300).astype(np.float32)
    ref = np.convolve(x.astype(np.float64), h.astype(np.float64))[:5000]
    out = fft_convolve(x, h, block=1024)
    assert out.shape == (5000,)
    assert np.allclose(out, ref, atol=1e-3)


def test_apply_rir_preserves_level_and_length() -> None:
    x = speech_like(3.0)
    y = apply_rir(x, normalise_rir(_rir()))
    assert y.shape == x.shape
    rms_x = np.sqrt(np.mean(x.astype(np.float64) ** 2))
    rms_y = np.sqrt(np.mean(y.astype(np.float64) ** 2))
    assert abs(rms_x - rms_y) / rms_x < 1e-4
    assert not np.array_equal(x, y)
