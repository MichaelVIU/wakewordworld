"""Detection semantics, applied identically to every engine.

Given per-frame scores and a threshold:

1. A *detection* fires on a rising edge: ``score[t] >= thr`` and ``score[t-1] < thr``
   (or ``t == 0``).
2. After a detection, further detections are suppressed for ``debounce_s``.
3. A detection is a *hit* for a target occurrence if it fires within
   ``[word_end - pre_s, word_end + post_s]``. Each detection matches at most one
   occurrence and each occurrence at most one detection (earliest-first greedy match).
4. Every unmatched detection is a *false accept*; every unmatched occurrence is a *miss*.
5. Negative time is the chunk duration minus the union of hit windows of the target
   occurrences, so false accepts per hour are computed over time where the target word
   was not spoken.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from wakewordworld.engines.base import FRAME_SAMPLES, SAMPLE_RATE

__all__ = [
    "DetectionConfig",
    "MatchResult",
    "Occurrence",
    "detections_from_scores",
    "match_detections",
    "negative_seconds",
]

FRAME_S = FRAME_SAMPLES / SAMPLE_RATE


@dataclass(frozen=True)
class DetectionConfig:
    """Parameters of the detection semantics (fixed per benchmark release)."""

    debounce_s: float = 1.0
    pre_s: float = 0.5
    post_s: float = 1.0


@dataclass(frozen=True)
class Occurrence:
    """A ground-truth occurrence of the target word/phrase in a chunk."""

    start_s: float
    end_s: float
    confusable: bool = False
    """True for near-miss occurrences (used to categorise false accepts, never hits)."""


@dataclass
class MatchResult:
    """Outcome of matching detections against occurrences for one chunk."""

    hits: list[tuple[float, float]] = field(default_factory=list)
    """(detection time, occurrence end) pairs."""
    misses: list[Occurrence] = field(default_factory=list)
    false_accepts: list[float] = field(default_factory=list)
    """Detection times not matching any target occurrence."""
    confusable_false_accepts: list[float] = field(default_factory=list)
    """Subset of ``false_accepts`` falling inside a confusable occurrence window."""

    @property
    def n_hits(self) -> int:
        """Number of hits."""
        return len(self.hits)

    @property
    def n_misses(self) -> int:
        """Number of misses."""
        return len(self.misses)

    @property
    def n_false_accepts(self) -> int:
        """Number of false accepts (including confusable ones)."""
        return len(self.false_accepts)

    def latencies(self) -> list[float]:
        """Detection latencies (detection time minus word end) for hits."""
        return [t_det - t_end for t_det, t_end in self.hits]


def detections_from_scores(
    scores: NDArray[np.floating], threshold: float, config: DetectionConfig
) -> NDArray[np.float64]:
    """Return detection times (frame end times) for one score series."""
    if scores.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    above = scores >= threshold
    prev = np.concatenate(([False], above[:-1]))
    edges = np.flatnonzero(above & ~prev)
    if edges.size == 0:
        return np.zeros(0, dtype=np.float64)
    times = (edges + 1) * FRAME_S
    kept: list[float] = []
    last = -np.inf
    for t in times:
        if t - last >= config.debounce_s:
            kept.append(float(t))
            last = t
    return np.asarray(kept, dtype=np.float64)


def match_detections(
    detections: NDArray[np.float64],
    occurrences: list[Occurrence],
    config: DetectionConfig,
) -> MatchResult:
    """Greedy earliest-first matching of detections to target occurrences."""
    result = MatchResult()
    targets = sorted((o for o in occurrences if not o.confusable), key=lambda o: o.end_s)
    confusables = [o for o in occurrences if o.confusable]
    used = np.zeros(len(targets), dtype=bool)
    ends = np.asarray([o.end_s for o in targets], dtype=np.float64)
    for t in np.sort(detections):
        lo = t - config.post_s  # occurrence end must be >= lo ...
        hi = t + config.pre_s  # ... and <= hi
        candidates = np.flatnonzero((ends >= lo) & (ends <= hi) & ~used)
        if candidates.size:
            k = int(candidates[0])
            used[k] = True
            result.hits.append((float(t), float(ends[k])))
        else:
            result.false_accepts.append(float(t))
            if any(o.end_s - config.pre_s <= t <= o.end_s + config.post_s for o in confusables):
                result.confusable_false_accepts.append(float(t))
    result.misses = [o for o, u in zip(targets, used, strict=True) if not u]
    return result


def negative_seconds(
    duration_s: float, occurrences: list[Occurrence], config: DetectionConfig
) -> float:
    """Chunk duration minus the union of target hit windows."""
    windows = sorted(
        (max(0.0, o.end_s - config.pre_s), min(duration_s, o.end_s + config.post_s))
        for o in occurrences
        if not o.confusable
    )
    covered = 0.0
    cur_lo, cur_hi = None, None
    for lo, hi in windows:
        if cur_hi is None or lo > cur_hi:
            if cur_hi is not None and cur_lo is not None:
                covered += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
        else:
            cur_hi = max(cur_hi, hi)
    if cur_hi is not None and cur_lo is not None:
        covered += cur_hi - cur_lo
    return max(0.0, duration_s - covered)
