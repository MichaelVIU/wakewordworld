"""Heuristic music / non-speech tagging for transcript segments.

Deliberately cheap and dependency-free (numpy + soundfile). Over 46 ms frames of a
segment we compute the spectral flatness and the zero-crossing rate (ZCR) and look
at how much they *vary* between frames. Speech alternates voiced (peaky spectrum,
low ZCR) and unvoiced (flat spectrum, high ZCR) frames every few tens of
milliseconds, so both features fluctuate strongly. Sustained music, tones, hum and
broadband noise are comparatively steady. A segment is tagged when

* both the ZCR and the flatness are steady (coefficient of variation below the
  thresholds), or
* the mean flatness is very high (broadband noise), or
* its text is empty or a bare music/applause marker.

It is a heuristic: expect some false tags on very noisy or monotone speech and
misses on rhythmic music with vocals. Tagged segments are excluded from wake word
positives at index time but stay in the negative hours.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf

from wakewordworld.transcribe.schema import Segment

__all__ = ["MusicFeatures", "is_music_marker", "looks_like_music", "music_features", "tag_music"]

FRAME_S = 0.046
_MARKER_RE = re.compile(
    r"^\W*(?:\[(?:musik|music|musique|música|applause|applaus|rires|laughter)\]|♪+|♫+)\W*$",
    re.IGNORECASE,
)


class MusicFeatures(NamedTuple):
    """Frame statistics of one audio excerpt."""

    flatness_mean: float
    flatness_cv: float
    zcr_cv: float
    n_active_frames: int


def is_music_marker(text: str) -> bool:
    """Whether the text is empty or a bare music/applause marker."""
    return not text.strip() or bool(_MARKER_RE.match(text.strip()))


def music_features(audio: np.ndarray, sample_rate: int) -> MusicFeatures:
    """Compute spectral flatness and ZCR statistics over active frames."""
    frame = max(int(FRAME_S * sample_rate), 64)
    n = audio.size // frame
    if n < 3:
        return MusicFeatures(0.0, 1.0, 1.0, 0)
    frames = audio[: n * frame].reshape(n, frame).astype(np.float64)
    frames = frames - frames.mean(axis=1, keepdims=True)
    energy = (frames**2).mean(axis=1)
    active = energy > max(energy.max() * 1e-3, 1e-10)
    if int(active.sum()) < 3:
        return MusicFeatures(0.0, 1.0, 1.0, int(active.sum()))
    frames = frames[active]
    window = np.hanning(frame)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2 + 1e-12
    flatness = np.exp(np.log(spec).mean(axis=1)) / spec.mean(axis=1)
    zcr = (np.abs(np.diff(np.sign(frames), axis=1)) > 0).mean(axis=1)
    return MusicFeatures(
        flatness_mean=float(flatness.mean()),
        flatness_cv=float(flatness.std() / (flatness.mean() + 1e-9)),
        zcr_cv=float(zcr.std() / (zcr.mean() + 1e-9)),
        n_active_frames=int(frames.shape[0]),
    )


def looks_like_music(
    feats: MusicFeatures,
    *,
    zcr_cv_threshold: float = 0.25,
    flatness_cv_threshold: float = 0.5,
    noise_flatness: float = 0.5,
) -> bool:
    """Decision rule on :class:`MusicFeatures`."""
    if feats.n_active_frames < 3:
        return False
    steady = feats.zcr_cv < zcr_cv_threshold and feats.flatness_cv < flatness_cv_threshold
    return steady or feats.flatness_mean > noise_flatness


def tag_music(
    audio_path: Path,
    segments: list[Segment],
    *,
    min_duration_s: float = 1.0,
    **thresholds: float,
) -> None:
    """Set ``is_music`` on segments in place.

    Args:
        audio_path: 16 kHz mono file the segments refer to.
        segments: Segments to tag (mutated).
        min_duration_s: Shorter segments are only tagged by their text.
        **thresholds: Passed to :func:`looks_like_music`.
    """
    info = sf.info(str(audio_path))
    with sf.SoundFile(str(audio_path)) as fh:
        for seg in segments:
            if is_music_marker(seg.text):
                seg.is_music = True
                continue
            if seg.end - seg.start < min_duration_s:
                continue
            start = int(seg.start * info.samplerate)
            n = int((seg.end - seg.start) * info.samplerate)
            if start >= info.frames or n <= 0:
                continue
            fh.seek(start)
            audio = fh.read(n, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if looks_like_music(music_features(audio, info.samplerate), **thresholds):
                seg.is_music = True
