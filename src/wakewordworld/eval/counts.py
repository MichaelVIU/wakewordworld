"""Build count tables from score tables and ground-truth occurrences."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wakewordworld.eval.detect import (
    DetectionConfig,
    Occurrence,
    detections_from_scores,
    match_detections,
    negative_seconds,
)
from wakewordworld.eval.metrics import CountTable

__all__ = ["ChunkScores", "build_count_table", "chunk_counts"]


@dataclass(frozen=True)
class ChunkScores:
    """Scores of one chunk for one wake word plus its grouping unit (file)."""

    chunk_id: str
    unit_id: str
    duration_s: float
    scores: NDArray[np.float32]


def chunk_counts(
    scores: NDArray[np.floating],
    duration_s: float,
    occurrences: list[Occurrence],
    thresholds: NDArray[np.float64],
    config: DetectionConfig,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], float, list[float]]:
    """Hits, misses, false accepts per threshold for one chunk, negative seconds, latencies.

    Latencies are collected at the *lowest* threshold with at least one hit only when
    ``thresholds`` has a single entry; otherwise callers compute latencies separately at
    a chosen operating point.
    """
    t = thresholds.shape[0]
    hits = np.zeros(t, dtype=np.int64)
    misses = np.zeros(t, dtype=np.int64)
    fas = np.zeros(t, dtype=np.int64)
    latencies: list[float] = []
    for k, thr in enumerate(thresholds):
        dets = detections_from_scores(scores, float(thr), config)
        m = match_detections(dets, occurrences, config)
        hits[k] = m.n_hits
        misses[k] = m.n_misses
        fas[k] = m.n_false_accepts
        if t == 1:
            latencies = m.latencies()
    return hits, misses, fas, negative_seconds(duration_s, occurrences, config), latencies


def build_count_table(
    chunks: Iterable[ChunkScores],
    occurrences: Mapping[str, list[Occurrence]],
    thresholds: NDArray[np.float64],
    config: DetectionConfig,
) -> CountTable:
    """Aggregate per-chunk counts into per-unit rows."""
    by_unit: dict[
        str, list[tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], float]]
    ] = {}
    for c in chunks:
        h, m, f, neg_s, _ = chunk_counts(
            c.scores, c.duration_s, occurrences.get(c.chunk_id, []), thresholds, config
        )
        by_unit.setdefault(c.unit_id, []).append((h, m, f, neg_s))
    unit_ids = sorted(by_unit)
    u = len(unit_ids)
    t = thresholds.shape[0]
    hits = np.zeros((u, t), dtype=np.int64)
    misses = np.zeros((u, t), dtype=np.int64)
    fas = np.zeros((u, t), dtype=np.int64)
    negative = np.zeros(u, dtype=np.float64)
    for i, uid in enumerate(unit_ids):
        for h, m, f, n in by_unit[uid]:
            hits[i] += h
            misses[i] += m
            fas[i] += f
            negative[i] += n
    return CountTable(
        thresholds=thresholds,
        hits=hits,
        misses=misses,
        false_accepts=fas,
        negative_s=negative,
        unit_ids=unit_ids,
    )
