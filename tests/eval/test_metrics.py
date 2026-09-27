from __future__ import annotations

import numpy as np

from wakewordworld.eval.counts import ChunkScores, build_count_table
from wakewordworld.eval.detect import DetectionConfig, Occurrence
from wakewordworld.eval.metrics import (
    CountTable,
    bootstrap_summary,
    det_curve,
    summarise,
    threshold_grid,
)

CFG = DetectionConfig()


def _table() -> CountTable:
    thresholds = np.array([0.2, 0.5, 0.8])
    # two units; unit 0 has 10 positives, unit 1 has 10 positives
    hits = np.array([[10, 8, 5], [10, 9, 6]])
    misses = np.array([[0, 2, 5], [0, 1, 4]])
    fas = np.array([[40, 4, 0], [20, 2, 0]])
    neg = np.array([3600.0, 3600.0])
    return CountTable(thresholds, hits, misses, fas, neg, ["a", "b"])


def test_det_curve_and_budgets() -> None:
    curve = det_curve(_table())
    assert curve.n_positives == 20
    assert abs(curve.negative_hours - 2.0) < 1e-9
    fa = [p.fa_per_hour for p in curve.points]
    assert fa == [30.0, 3.0, 0.0]
    frr = [p.frr for p in curve.points]
    assert frr == [0.0, 0.15, 0.45]
    s = summarise(_table(), fa_targets=(0.1, 3.0, 100.0))
    assert s.frr_at_fa[0.1] == 0.45  # only threshold 0.8 reaches <= 0.1 FA/h
    assert s.frr_at_fa[3.0] == 0.15
    assert s.frr_at_fa[100.0] == 0.0
    assert 0.0 <= s.aut <= 1.0


def test_unreachable_budget_is_none() -> None:
    t = _table()
    t.false_accepts[:, :] = 99
    assert summarise(t, fa_targets=(0.1,)).frr_at_fa[0.1] is None


def test_bootstrap_intervals_contain_point_estimate() -> None:
    t = _table()
    s = summarise(t, fa_targets=(3.0,))
    ci = bootstrap_summary(t, fa_targets=(3.0,), n_boot=200)
    lo, hi = ci["frr_at_fa"][3.0]  # type: ignore[index]
    assert lo <= s.frr_at_fa[3.0] <= hi  # type: ignore[operator]
    alo, ahi = ci["aut"]  # type: ignore[misc]
    assert alo <= s.aut <= ahi


def test_threshold_grid_covers_unit_interval() -> None:
    g = threshold_grid(np.array([0.1, 0.2, 0.9], dtype=np.float32), n=10)
    assert g[0] == 0.0
    assert g[-1] > 1.0
    assert np.all(np.diff(g) > 0)


def test_build_count_table_from_scores() -> None:
    # chunk 1: word at 10.0-10.5 s, engine fires at frame ending 10.8 s with score 0.9
    n = 1000
    s1 = np.zeros(n, dtype=np.float32)
    s1[134] = 0.9  # (134+1)*0.08 = 10.8 s
    s1[500] = 0.6  # false accept at 40.08 s for thr <= 0.6
    s2 = np.zeros(n, dtype=np.float32)  # chunk 2: nothing
    chunks = [
        ChunkScores("c1", "file1", 80.0, s1),
        ChunkScores("c2", "file2", 80.0, s2),
    ]
    occ = {"c1": [Occurrence(10.0, 10.5)]}
    table = build_count_table(chunks, occ, np.array([0.5, 0.7, 0.95]), CFG)
    assert table.unit_ids == ["file1", "file2"]
    assert table.hits.tolist() == [[1, 1, 0], [0, 0, 0]]
    assert table.misses.tolist() == [[0, 0, 1], [0, 0, 0]]
    assert table.false_accepts.tolist() == [[1, 0, 0], [0, 0, 0]]
    assert abs(table.negative_s[0] - (80.0 - 1.5)) < 1e-9
    assert table.negative_s[1] == 80.0
