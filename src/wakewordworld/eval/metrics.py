"""Metrics: false accepts per hour, false reject rate, DET curves, confidence intervals.

All metrics are computed from a *count table*: for each unit (a source file) and each
threshold, the number of hits, misses and false accepts and the negative seconds. This
makes threshold sweeps and cluster bootstrap over files cheap array operations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "CountTable",
    "CurvePoint",
    "DetCurve",
    "Summary",
    "bootstrap_summary",
    "det_curve",
    "summarise",
    "threshold_grid",
]

DEFAULT_FA_TARGETS: tuple[float, ...] = (0.1, 0.5, 1.0, 3.0)


@dataclass
class CountTable:
    """Counts per unit and threshold."""

    thresholds: NDArray[np.float64]
    """Shape ``(T,)``, increasing."""
    hits: NDArray[np.int64]
    """Shape ``(U, T)``."""
    misses: NDArray[np.int64]
    false_accepts: NDArray[np.int64]
    negative_s: NDArray[np.float64]
    """Shape ``(U,)``: negative seconds per unit (threshold independent)."""
    unit_ids: list[str]

    def __post_init__(self) -> None:
        u, t = self.hits.shape
        if self.misses.shape != (u, t) or self.false_accepts.shape != (u, t):
            msg = "count arrays must share shape (units, thresholds)"
            raise ValueError(msg)
        if self.negative_s.shape != (u,) or len(self.unit_ids) != u:
            msg = "negative_s and unit_ids must have one entry per unit"
            raise ValueError(msg)
        if self.thresholds.shape != (t,):
            msg = "thresholds must have one entry per threshold column"
            raise ValueError(msg)

    def subset(self, idx: NDArray[np.intp]) -> CountTable:
        """Rows selected by index (with repetition allowed, for bootstrap)."""
        return CountTable(
            thresholds=self.thresholds,
            hits=self.hits[idx],
            misses=self.misses[idx],
            false_accepts=self.false_accepts[idx],
            negative_s=self.negative_s[idx],
            unit_ids=[self.unit_ids[i] for i in idx],
        )


@dataclass(frozen=True)
class CurvePoint:
    """One operating point."""

    threshold: float
    fa_per_hour: float
    frr: float


@dataclass
class DetCurve:
    """Operating points across thresholds."""

    points: list[CurvePoint]
    n_positives: int
    negative_hours: float

    def frr_at(self, fa_target: float) -> CurvePoint | None:
        """Lowest-FRR point whose false accept rate does not exceed ``fa_target``."""
        ok = [p for p in self.points if p.fa_per_hour <= fa_target]
        if not ok:
            return None
        return min(ok, key=lambda p: (p.frr, p.fa_per_hour))

    def eer(self) -> float:
        """Point where FRR ≈ FA rate expressed as a fraction of positives per hour.

        FA/h and FRR have different units; the conventional EER for wake words uses the
        false-accept *rate per positive-length window*. We follow the common practical
        definition: interpolate where FRR equals FA/h divided by positives per hour.
        Returned as FRR at that point, ``nan`` when undefined.
        """
        if self.n_positives == 0 or self.negative_hours <= 0:
            return float("nan")
        pos_per_hour = self.n_positives / self.negative_hours
        best = float("nan")
        best_gap = float("inf")
        for p in self.points:
            far = p.fa_per_hour / pos_per_hour
            gap = abs(far - p.frr)
            if gap < best_gap:
                best_gap = gap
                best = (far + p.frr) / 2.0
        return best

    def aut(self, fa_lo: float = 0.1, fa_hi: float = 3.0) -> float:
        """Area under the DET trade-off, FRR integrated over log10(FA/h) in [fa_lo, fa_hi].

        Normalised by the log-width so the value is a mean FRR across the range;
        points outside the range are clipped, missing regions are filled with the
        nearest available FRR (1.0 when no point reaches ``fa_hi``).
        """
        pts = sorted(self.points, key=lambda p: p.fa_per_hour)
        xs = np.log10(np.clip([p.fa_per_hour for p in pts], 1e-6, None))
        ys = np.asarray([p.frr for p in pts])
        grid = np.linspace(np.log10(fa_lo), np.log10(fa_hi), 200)
        if xs.size == 0:
            return 1.0
        # For each grid FA budget, the best FRR achievable at or below that budget.
        best = np.ones_like(grid)
        for i, g in enumerate(grid):
            mask = xs <= g
            best[i] = ys[mask].min() if mask.any() else 1.0
        return float(np.trapezoid(best, grid) / (grid[-1] - grid[0]))


@dataclass
class Summary:
    """Headline numbers at fixed false-accept budgets."""

    fa_targets: tuple[float, ...]
    frr_at_fa: dict[float, float | None]
    threshold_at_fa: dict[float, float | None]
    eer: float
    aut: float
    n_positives: int
    negative_hours: float


def threshold_grid(scores: NDArray[np.floating], n: int = 200) -> NDArray[np.float64]:
    """Thresholds spanning the observed score distribution plus the unit interval ends."""
    if scores.size == 0:
        return np.linspace(0.0, 1.0, n)
    qs = np.quantile(scores.astype(np.float64), np.linspace(0.0, 1.0, n))
    grid = np.unique(np.concatenate(([0.0], qs, [1.0 + 1e-9])))
    return grid


def det_curve(table: CountTable) -> DetCurve:
    """Aggregate a count table into a DET curve."""
    hits = table.hits.sum(axis=0)
    misses = table.misses.sum(axis=0)
    fas = table.false_accepts.sum(axis=0)
    neg_h = float(table.negative_s.sum()) / 3600.0
    n_pos = int((hits + misses)[0]) if hits.size else 0
    points: list[CurvePoint] = []
    for k, thr in enumerate(table.thresholds):
        total = hits[k] + misses[k]
        frr = float(misses[k] / total) if total else float("nan")
        fa_h = float(fas[k] / neg_h) if neg_h > 0 else float("nan")
        points.append(CurvePoint(threshold=float(thr), fa_per_hour=fa_h, frr=frr))
    return DetCurve(points=points, n_positives=n_pos, negative_hours=neg_h)


def summarise(table: CountTable, fa_targets: tuple[float, ...] = DEFAULT_FA_TARGETS) -> Summary:
    """Headline summary from a count table."""
    curve = det_curve(table)
    frr_at: dict[float, float | None] = {}
    thr_at: dict[float, float | None] = {}
    for target in fa_targets:
        p = curve.frr_at(target)
        frr_at[target] = p.frr if p else None
        thr_at[target] = p.threshold if p else None
    return Summary(
        fa_targets=fa_targets,
        frr_at_fa=frr_at,
        threshold_at_fa=thr_at,
        eer=curve.eer(),
        aut=curve.aut(),
        n_positives=curve.n_positives,
        negative_hours=curve.negative_hours,
    )


def bootstrap_summary(
    table: CountTable,
    *,
    fa_targets: tuple[float, ...] = DEFAULT_FA_TARGETS,
    n_boot: int = 1000,
    seed: int = 20260927,
    ci: float = 0.95,
) -> dict[str, dict[float, tuple[float, float]] | tuple[float, float]]:
    """Cluster bootstrap over units (files) for FRR at each FA target and for AUT.

    Returns ``{"frr_at_fa": {target: (lo, hi)}, "aut": (lo, hi)}``. Targets that are
    unreachable in a resample contribute ``1.0`` (worst case), which keeps intervals
    honest for small negative sets.
    """
    rng = np.random.default_rng(seed)
    u = table.hits.shape[0]
    frr_samples: dict[float, list[float]] = {t: [] for t in fa_targets}
    aut_samples: list[float] = []
    for _ in range(n_boot):
        idx: NDArray[np.intp] = rng.integers(0, u, size=u).astype(np.intp)
        s = summarise(table.subset(idx), fa_targets)
        for t in fa_targets:
            v = s.frr_at_fa[t]
            frr_samples[t].append(1.0 if v is None else v)
        aut_samples.append(s.aut)
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    out: dict[str, dict[float, tuple[float, float]] | tuple[float, float]] = {
        "frr_at_fa": {
            t: (float(np.quantile(v, lo_q)), float(np.quantile(v, hi_q)))
            for t, v in frr_samples.items()
        },
        "aut": (float(np.quantile(aut_samples, lo_q)), float(np.quantile(aut_samples, hi_q))),
    }
    return out
