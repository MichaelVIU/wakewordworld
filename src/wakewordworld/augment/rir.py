"""Room impulse responses: normalisation, level-preserving convolution, and a bank."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from wakewordworld.augment.noise import SAMPLE_RATE, NoiseBank, NoiseClip, load_audio_16k

__all__ = ["RirBank", "apply_rir", "fft_convolve", "normalise_rir"]


def normalise_rir(
    h: NDArray[np.float32], *, direct_window_ms: float = 2.5, sample_rate: int = SAMPLE_RATE
) -> NDArray[np.float32]:
    """Trim leading silence up to the direct-path window and scale it to unit energy.

    The direct path is the ±``direct_window_ms`` window around the absolute peak; the
    returned response starts at the beginning of that window, so the peak sits at most
    ``direct_window_ms`` after index 0 (pre-ringing of measured responses is kept).
    """
    if h.size == 0:
        msg = "empty impulse response"
        raise ValueError(msg)
    peak = int(np.argmax(np.abs(h)))
    win = max(1, round(direct_window_ms * sample_rate / 1000.0))
    lo, hi = max(0, peak - win), min(h.shape[0], peak + win + 1)
    direct_energy = float(np.sum(h[lo:hi].astype(np.float64) ** 2))
    if direct_energy <= 0:
        msg = "impulse response has no energy"
        raise ValueError(msg)
    aligned = h[lo:].astype(np.float32)
    out: NDArray[np.float32] = (aligned / np.sqrt(direct_energy)).astype(np.float32)
    return out


def fft_convolve(
    x: NDArray[np.float32], h: NDArray[np.float32], *, block: int = 1 << 16
) -> NDArray[np.float32]:
    """Overlap-add FFT convolution returning the first ``len(x)`` samples."""
    n_x, n_h = x.shape[0], h.shape[0]
    if n_x == 0:
        return x.astype(np.float32)
    seg = max(block, 1)
    n_fft = 1
    while n_fft < seg + n_h - 1:
        n_fft <<= 1
    hf = np.fft.rfft(h.astype(np.float64), n_fft)
    out = np.zeros(n_x + n_h - 1, dtype=np.float64)
    for start in range(0, n_x, seg):
        piece = x[start : start + seg].astype(np.float64)
        yf = np.fft.irfft(np.fft.rfft(piece, n_fft) * hf, n_fft)
        end = min(start + piece.shape[0] + n_h - 1, out.shape[0])
        out[start:end] += yf[: end - start]
    return out[:n_x].astype(np.float32)


def apply_rir(x: NDArray[np.float32], h: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convolve with a normalised RIR and rescale so the output RMS equals the input RMS."""
    y = fft_convolve(x, h)
    rms_in = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if x.size else 0.0
    rms_out = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) if y.size else 0.0
    if rms_in > 0 and rms_out > 0:
        y = (y * (rms_in / rms_out)).astype(np.float32)
    return y


class RirBank(NoiseBank):
    """A :class:`NoiseBank` over RIR sets whose ``load`` returns normalised responses."""

    def load(self, clip: NoiseClip) -> NDArray[np.float32]:
        """Load and normalise an impulse response."""
        raw = load_audio_16k(clip.path, resample=self._resample)
        return normalise_rir(raw)

    def pick(self, chunk_id: str, categories: Sequence[str], *, seed: str) -> NoiseClip:
        """Deterministic choice; defaults to real RIRs when no category is given."""
        cats = list(categories) or ["real_rir"]
        if not self.candidates(cats):
            cats = []
        return super().pick(chunk_id, cats, seed=seed + ":rir")
