"""Dependency-free SVG helpers: DET plots and inline confidence-interval bars.

The DET plot draws, for each engine, the best false reject rate achievable at or below
each false-accept budget (a monotone step), on a log10 false-accepts-per-hour axis.
That is the same reading of a curve that :meth:`DetCurve.aut` uses, so the plotted
area is what the ranking metric measures.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from html import escape

__all__ = ["OKABE_ITO", "det_svg", "interval_bar_svg"]

# Colour-blind-safe palette (Okabe & Ito, 2008).
OKABE_ITO: tuple[str, ...] = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
)

Curve = Sequence[tuple[float, float]]
"""(false accepts per hour, false reject rate) points."""

_GRID_FA: tuple[float, ...] = (0.1, 0.5, 1.0, 3.0)


def _monotone_step(points: Curve, fa_lo: float, fa_hi: float) -> list[tuple[float, float]]:
    """Best FRR at or below each FA budget, evaluated at the sorted FA positions."""
    clipped = sorted(
        (min(max(fa, fa_lo), fa_hi), frr)
        for fa, frr in points
        if not (math.isnan(fa) or math.isnan(frr))
    )
    if not clipped:
        return []
    out: list[tuple[float, float]] = []
    best = 1.0
    for fa, frr in clipped:
        best = min(best, frr)
        out.append((fa, best))
    # Extend to the right edge so the step is visible over the whole budget range.
    if out[-1][0] < fa_hi:
        out.append((fa_hi, out[-1][1]))
    return out


def det_svg(
    curves: Sequence[tuple[str, Curve]],
    *,
    width: int = 720,
    height: int = 440,
    fa_range: tuple[float, float] = (0.01, 10.0),
    frr_range: tuple[float, float] = (0.0, 1.0),
    title: str | None = None,
) -> str:
    """Render a DET plot (FA/h on log10 x axis, FRR in percent on the y axis)."""
    fa_lo, fa_hi = fa_range
    if fa_lo <= 0 or fa_hi <= fa_lo:
        msg = "fa_range must be positive and increasing"
        raise ValueError(msg)
    frr_lo, frr_hi = frr_range
    margin_l, margin_r, margin_t, margin_b = 64, 20, 36 if title else 16, 48
    legend_h = 18 * len(curves) + 8 if curves else 0
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b - legend_h
    lx_lo, lx_hi = math.log10(fa_lo), math.log10(fa_hi)

    def x(fa: float) -> float:
        return margin_l + (math.log10(fa) - lx_lo) / (lx_hi - lx_lo) * plot_w

    def y(frr: float) -> float:
        return margin_t + (frr_hi - frr) / (frr_hi - frr_lo) * plot_h

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" class="det">',
    ]
    if title:
        parts.append(
            f'<title>{escape(title)}</title><text x="{margin_l}" y="22" class="det-title">'
            f"{escape(title)}</text>"
        )
    parts.append(
        f'<rect x="{margin_l}" y="{margin_t}" width="{plot_w}" height="{plot_h}" class="det-bg"/>'
    )
    # Vertical gridlines at the reporting budgets and decades.
    decade = math.ceil(lx_lo)
    grid_positions = set(_GRID_FA)
    while decade <= lx_hi:
        grid_positions.add(10.0**decade)
        decade += 1
    for fa in sorted(grid_positions):
        if fa < fa_lo or fa > fa_hi:
            continue
        gx = x(fa)
        emphasis = " det-grid-budget" if fa in _GRID_FA else ""
        parts.append(
            f'<line x1="{gx:.1f}" y1="{margin_t}" x2="{gx:.1f}" y2="{margin_t + plot_h}" '
            f'class="det-grid{emphasis}"/>'
            f'<text x="{gx:.1f}" y="{margin_t + plot_h + 16}" text-anchor="middle" '
            f'class="det-tick">{fa:g}</text>'
        )
    # Horizontal gridlines every 10 percent.
    step = (frr_hi - frr_lo) / 10.0
    for k in range(11):
        frr = frr_lo + k * step
        gy = y(frr)
        parts.append(
            f'<line x1="{margin_l}" y1="{gy:.1f}" x2="{margin_l + plot_w}" y2="{gy:.1f}" '
            f'class="det-grid"/>'
            f'<text x="{margin_l - 6}" y="{gy + 4:.1f}" text-anchor="end" class="det-tick">'
            f"{frr * 100:.0f}%</text>"
        )
    parts.append(
        f'<text x="{margin_l + plot_w / 2:.1f}" y="{margin_t + plot_h + 36}" '
        f'text-anchor="middle" class="det-label">false accepts per hour</text>'
        f'<text transform="translate(14 {margin_t + plot_h / 2:.1f}) rotate(-90)" '
        f'text-anchor="middle" class="det-label">false reject rate</text>'
    )
    for i, (label, points) in enumerate(curves):
        colour = OKABE_ITO[i % len(OKABE_ITO)]
        stepped = _monotone_step(points, fa_lo, fa_hi)
        if stepped:
            d: list[str] = []
            prev_y: float | None = None
            for fa, frr in stepped:
                px, py = x(fa), y(min(max(frr, frr_lo), frr_hi))
                if prev_y is None:
                    d.append(f"M{px:.1f},{py:.1f}")
                else:
                    d.append(f"H{px:.1f}V{py:.1f}")
                prev_y = py
            dash = "" if i < len(OKABE_ITO) else ' stroke-dasharray="6 3"'
            parts.append(
                f'<path d="{" ".join(d)}" fill="none" stroke="{colour}" stroke-width="2"{dash}>'
                f"<title>{escape(label)}</title></path>"
            )
        ly = margin_t + plot_h + margin_b + 18 * i
        parts.append(
            f'<line x1="{margin_l}" y1="{ly}" x2="{margin_l + 24}" y2="{ly}" stroke="{colour}" '
            f'stroke-width="3"/><text x="{margin_l + 30}" y="{ly + 4}" class="det-legend">'
            f"{escape(label)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def interval_bar_svg(
    value: float | None,
    lo: float | None,
    hi: float | None,
    *,
    width: int = 160,
    height: int = 14,
    vmax: float = 1.0,
) -> str:
    """Inline bar: a point estimate with its interval, scaled to ``[0, vmax]``."""
    if value is None or math.isnan(value):
        return f'<svg width="{width}" height="{height}" class="ci" aria-hidden="true"></svg>'
    lo_v = value if lo is None or math.isnan(lo) else lo
    hi_v = value if hi is None or math.isnan(hi) else hi

    def sx(v: float) -> float:
        return min(max(v, 0.0), vmax) / vmax * (width - 4) + 2

    mid = height / 2
    label = f"{value:.3f} [{lo_v:.3f}, {hi_v:.3f}]"
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" class="ci" '
        f'role="img"><title>{escape(label)}</title>'
        f'<line x1="2" y1="{mid}" x2="{width - 2}" y2="{mid}" class="ci-axis"/>'
        f'<line x1="{sx(lo_v):.1f}" y1="{mid}" x2="{sx(hi_v):.1f}" y2="{mid}" class="ci-range"/>'
        f'<circle cx="{sx(value):.1f}" cy="{mid}" r="3.5" class="ci-point"/></svg>'
    )
