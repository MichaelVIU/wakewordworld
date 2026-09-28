from __future__ import annotations

import numpy as np

from wakewordworld.eval.detect import (
    FRAME_S,
    DetectionConfig,
    Occurrence,
    detections_from_scores,
    match_detections,
    negative_seconds,
)

CFG = DetectionConfig(debounce_s=1.0, pre_s=0.5, post_s=1.0)


def test_rising_edge_and_debounce() -> None:
    scores = np.zeros(100, dtype=np.float32)
    scores[10:14] = 0.9  # one plateau -> one detection at frame 10
    scores[15:16] = 0.9  # within debounce of the first -> suppressed
    scores[40:41] = 0.9  # > 1 s later -> second detection
    dets = detections_from_scores(scores, 0.5, CFG)
    assert dets.tolist() == [11 * FRAME_S, 41 * FRAME_S]


def test_first_frame_counts_as_edge() -> None:
    scores = np.array([0.9, 0.9, 0.0], dtype=np.float32)
    assert detections_from_scores(scores, 0.5, CFG).tolist() == [FRAME_S]


def test_empty_scores() -> None:
    assert detections_from_scores(np.zeros(0), 0.5, CFG).size == 0


def test_matching_hits_misses_false_accepts() -> None:
    occ = [Occurrence(10.0, 10.6), Occurrence(30.0, 30.5), Occurrence(50.0, 50.4, confusable=True)]
    dets = np.array([10.9, 20.0, 50.6])  # hit, FA, confusable FA
    m = match_detections(dets, occ, CFG)
    assert m.n_hits == 1
    assert m.hits[0] == (10.9, 10.6)
    assert m.n_misses == 1
    assert m.misses[0].end_s == 30.5
    assert m.false_accepts == [20.0, 50.6]
    assert m.confusable_false_accepts == [50.6]
    assert m.latencies() == [10.9 - 10.6]


def test_hit_window_bounds() -> None:
    occ = [Occurrence(10.0, 10.0)]
    assert match_detections(np.array([9.5]), occ, CFG).n_hits == 1  # word_end - pre
    assert match_detections(np.array([11.0]), occ, CFG).n_hits == 1  # word_end + post
    assert match_detections(np.array([9.4]), occ, CFG).n_hits == 0
    assert match_detections(np.array([11.1]), occ, CFG).n_hits == 0


def test_one_detection_per_occurrence() -> None:
    occ = [Occurrence(10.0, 10.0)]
    m = match_detections(np.array([9.6, 10.8]), occ, CFG)
    assert m.n_hits == 1
    assert m.false_accepts == [10.8]


def test_negative_seconds_merges_overlaps() -> None:
    occ = [Occurrence(10.0, 10.0), Occurrence(10.5, 10.8), Occurrence(50.0, 50.0, confusable=True)]
    # windows: [9.5, 11.0] and [10.3, 11.8] -> union [9.5, 11.8] = 2.3 s; confusable ignored
    assert abs(negative_seconds(100.0, occ, CFG) - (100.0 - 2.3)) < 1e-9
    assert negative_seconds(1.0, [Occurrence(0.0, 0.5)], CFG) == 0.0


def test_unaligned_clip_occurrence_widens_window() -> None:
    """A word spanning a whole clip (no alignment) accepts detections anywhere in it."""
    occ = [Occurrence(0.0, 4.5)]
    m = match_detections(np.array([1.0]), occ, CFG)
    assert m.n_hits == 1
    assert m.false_accepts == []
    assert negative_seconds(4.5, occ, CFG) == 0.0
