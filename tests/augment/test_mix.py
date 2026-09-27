from __future__ import annotations

import numpy as np
import pytest

from wakewordworld.augment.mix import active_mask, mix_at_snr, peak_limit, speed_perturb
from wakewordworld.augment.noise import resample_linear, rng_for

from .conftest import SR, speech_like, steady_noise


@pytest.mark.parametrize("snr", [20.0, 10.0, 5.0, 0.0])
def test_snr_accuracy(snr: float) -> None:
    speech = speech_like(6.0)
    noise = steady_noise(2.0)
    res = mix_at_snr(speech, noise, snr, rng=np.random.default_rng(0))
    mask = active_mask(speech)
    p_s = np.mean(speech[mask].astype(np.float64) ** 2)
    p_n = np.mean(res.noise_scaled.astype(np.float64) ** 2)
    measured = 10 * np.log10(p_s / p_n)
    assert abs(measured - snr) < 0.3
    assert abs(res.achieved_snr_db - snr) < 0.3
    assert res.audio.shape == speech.shape
    assert res.audio.dtype == np.float32


def test_active_mask_ignores_silence() -> None:
    speech = speech_like(4.0)
    mask = active_mask(speech)
    assert 0.1 < mask.mean() < 0.95
    silent = np.zeros(SR, dtype=np.float32)
    assert active_mask(silent).any()  # falls back to the loudest frame


def test_peak_limit() -> None:
    x = np.array([0.0, 2.0, -1.5], dtype=np.float32)
    y, gain = peak_limit(x)
    assert np.max(np.abs(y)) <= 10 ** (-1 / 20) + 1e-6
    assert gain < 0
    z, g0 = peak_limit(np.array([0.1, -0.2], dtype=np.float32))
    assert g0 == 0.0
    assert np.array_equal(z, np.array([0.1, -0.2], dtype=np.float32))


def test_mix_limits_peak() -> None:
    speech = (speech_like(2.0) * 3.0).astype(np.float32)
    res = mix_at_snr(speech, steady_noise(1.0), 0.0, rng=np.random.default_rng(1))
    assert np.max(np.abs(res.audio)) <= 10 ** (-1 / 20) + 1e-6
    assert res.gain_db < 0


def test_mix_is_deterministic_by_seed() -> None:
    speech = speech_like(2.0)
    noise = steady_noise(5.0)
    a = mix_at_snr(speech, noise, 10.0, rng=rng_for("s", "chunk-1"))
    b = mix_at_snr(speech, noise, 10.0, rng=rng_for("s", "chunk-1"))
    c = mix_at_snr(speech, noise, 10.0, rng=rng_for("s", "chunk-2"))
    assert a.noise_offset == b.noise_offset
    assert np.array_equal(a.audio, b.audio)
    assert a.noise_offset != c.noise_offset


def test_mix_tiles_short_noise() -> None:
    speech = speech_like(5.0)
    res = mix_at_snr(speech, steady_noise(0.5), 10.0, rng=np.random.default_rng(3))
    assert res.audio.shape == speech.shape


def test_mix_rejects_empty_or_silent_noise() -> None:
    speech = speech_like(1.0)
    with pytest.raises(ValueError, match="empty noise"):
        mix_at_snr(speech, np.zeros(0, dtype=np.float32), 10.0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="silent"):
        mix_at_snr(speech, np.zeros(SR, dtype=np.float32), 10.0, rng=np.random.default_rng(0))


@pytest.mark.parametrize(
    ("factor", "expected"), [(1.1, round(SR * 2 / 1.1)), (0.9, round(SR * 2 / 0.9))]
)
def test_speed_perturb_length(factor: float, expected: int) -> None:
    x = speech_like(2.0)
    y = speed_perturb(x, factor)
    assert y.shape[0] == expected
    assert y.dtype == np.float32


def test_speed_perturb_bounds_and_identity() -> None:
    x = speech_like(1.0)
    assert speed_perturb(x, 1.0) is x
    with pytest.raises(ValueError, match="outside"):
        speed_perturb(x, 1.5)


def test_resample_linear_length() -> None:
    x = np.ones(48000, dtype=np.float32)
    y = resample_linear(x, 48000, 16000)
    assert y.shape[0] == 16000
    assert np.allclose(y, 1.0)
    assert resample_linear(x, 16000, 16000) is x
