"""Level-controlled mixing and speed perturbation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wakewordworld.augment.noise import SAMPLE_RATE

__all__ = ["MixResult", "active_mask", "mix_at_snr", "peak_limit", "speed_perturb"]

PEAK_LIMIT_DBFS = -1.0
FRAME_SAMPLES = SAMPLE_RATE * 20 // 1000  # 20 ms


def active_mask(
    x: NDArray[np.float32], *, frame: int = FRAME_SAMPLES, percentile: float = 20.0
) -> NDArray[np.bool_]:
    """Per-sample mask of speech-active frames.

    A frame is active when its RMS is above the ``percentile`` of all frame RMS values
    and above -60 dBFS, so silences do not drag the measured speech level down.
    """
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.bool_)
    n_frames = int(np.ceil(n / frame))
    padded = np.zeros(n_frames * frame, dtype=np.float64)
    padded[:n] = x
    rms = np.sqrt(np.mean(padded.reshape(n_frames, frame) ** 2, axis=1))
    floor = 10 ** (-60 / 20)
    thr = max(float(np.percentile(rms, percentile)), floor)
    frame_mask = rms > thr
    if not frame_mask.any():
        frame_mask = rms >= rms.max()
    return np.repeat(frame_mask, frame)[:n]


@dataclass(frozen=True)
class MixResult:
    """Output of :func:`mix_at_snr`."""

    audio: NDArray[np.float32]
    noise_scaled: NDArray[np.float32]
    noise_offset: int
    gain_db: float
    """Output gain applied to keep the peak at or below -1 dBFS (0 when not needed)."""
    achieved_snr_db: float


def peak_limit(
    x: NDArray[np.float32], *, limit_dbfs: float = PEAK_LIMIT_DBFS
) -> tuple[NDArray[np.float32], float]:
    """Apply a single gain so the absolute peak does not exceed ``limit_dbfs``."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    limit = 10 ** (limit_dbfs / 20)
    if peak <= limit or peak == 0.0:
        return x, 0.0
    gain = limit / peak
    return (x * gain).astype(np.float32), float(20 * np.log10(gain))


def mix_at_snr(
    speech: NDArray[np.float32],
    noise: NDArray[np.float32],
    snr_db: float,
    *,
    rng: np.random.Generator,
    speech_active_mask: NDArray[np.bool_] | None = None,
) -> MixResult:
    """Add ``noise`` to ``speech`` at ``snr_db`` (speech level over active frames).

    The noise is tiled or cropped to the speech length starting at a random offset.
    """
    if speech.size == 0:
        return MixResult(speech, speech, 0, 0.0, float("nan"))
    if noise.size == 0:
        msg = "empty noise"
        raise ValueError(msg)
    mask = active_mask(speech) if speech_active_mask is None else speech_active_mask
    s64 = speech.astype(np.float64)
    p_speech = float(np.mean(s64[mask] ** 2)) if mask.any() else float(np.mean(s64**2))
    n = speech.shape[0]
    offset = int(rng.integers(0, noise.shape[0]))
    reps = int(np.ceil((offset + n) / noise.shape[0])) + 1
    tiled = np.tile(noise.astype(np.float64), reps)[offset : offset + n]
    p_noise = float(np.mean(tiled**2))
    if p_noise <= 0:
        msg = "noise segment is silent"
        raise ValueError(msg)
    scale = np.sqrt(p_speech / (p_noise * 10 ** (snr_db / 10)))
    noise_scaled = (tiled * scale).astype(np.float32)
    mixed = (s64 + noise_scaled).astype(np.float32)
    limited, gain_db = peak_limit(mixed)
    achieved = 10 * np.log10(p_speech / float(np.mean(noise_scaled.astype(np.float64) ** 2)))
    return MixResult(limited, noise_scaled, offset, gain_db, float(achieved))


def speed_perturb(x: NDArray[np.float32], factor: float) -> NDArray[np.float32]:
    """Change speed (and pitch) by resampling; ``factor`` 1.1 makes speech 10 % faster.

    Linear interpolation is used, which slightly low-passes the signal; the factor is
    restricted to [0.9, 1.1] where the artefact is negligible for this purpose.
    """
    if not 0.9 <= factor <= 1.1:
        msg = f"speed factor {factor} outside [0.9, 1.1]"
        raise ValueError(msg)
    if factor == 1.0 or x.size == 0:
        return x
    n_out = round(x.shape[0] / factor)
    t_out = np.arange(n_out, dtype=np.float64) * factor
    t_in = np.arange(x.shape[0], dtype=np.float64)
    return np.interp(t_out, t_in, x.astype(np.float64)).astype(np.float32)
