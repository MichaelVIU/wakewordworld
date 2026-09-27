"""Render a self-contained HTML report from evaluation runs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from jinja2 import Environment, PackageLoader, select_autoescape
from markupsafe import Markup

from wakewordworld import __version__
from wakewordworld.report.build import (
    MIN_POSITIVES,
    MIN_UNITS,
    Run,
    leaderboard,
    per_domain,
    per_language,
    sealed_gap,
)
from wakewordworld.report.svg import det_svg, interval_bar_svg

__all__ = ["build_context", "render_report"]

_BUDGETS: tuple[tuple[float, str], ...] = (
    (0.1, "frr_at_0_1"),
    (0.5, "frr_at_0_5"),
    (1.0, "frr_at_1"),
    (3.0, "frr_at_3"),
)


def _fmt(v: object, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        if v != v:  # NaN
            return "-"
        return f"{v:.{digits}f}"
    return str(v)


def _fmt_ci(v: object, lo: object, hi: object, digits: int = 3) -> str:
    if v is None:
        return "-"
    return f"{_fmt(v, digits)} [{_fmt(lo, digits)}, {_fmt(hi, digits)}]"


def _short_hashes(hashes: Mapping[str, str] | None) -> str:
    if not hashes:
        return "-"
    return ", ".join(f"{k}:{v[:10]}" for k, v in sorted(hashes.items()))


def _curves_for(
    runs: Sequence[Run], *, wake_word: str, slice_type: str, slice_value: str
) -> list[tuple[str, list[tuple[float, float]]]]:
    out: list[tuple[str, list[tuple[float, float]]]] = []
    for r in runs:
        sub = r.curves.filter(
            (pl.col("wake_word") == wake_word)
            & (pl.col("slice_type") == slice_type)
            & (pl.col("slice_value") == slice_value)
        )
        if sub.is_empty():
            continue
        pts = list(
            zip(
                sub.get_column("fa_per_hour").to_list(),
                sub.get_column("frr").to_list(),
                strict=True,
            )
        )
        out.append((r.engine_label, pts))
    return out


def _engine_language_curves(
    run: Run, wake_word: str
) -> list[tuple[str, list[tuple[float, float]]]]:
    sub = run.curves.filter(
        (pl.col("wake_word") == wake_word) & (pl.col("slice_type") == "language")
    )
    out: list[tuple[str, list[tuple[float, float]]]] = []
    for (lang,), g in sub.group_by("slice_value", maintain_order=True):
        pts = list(
            zip(g.get_column("fa_per_hour").to_list(), g.get_column("frr").to_list(), strict=True)
        )
        out.append((str(lang), pts))
    return sorted(out)


def _pivot_table(df: pl.DataFrame) -> dict[str, Any]:
    if df.is_empty():
        return {"columns": [], "rows": []}
    value_cols = [c for c in df.columns if c not in ("engine", "wake_word")]
    rows = [
        {
            "engine": r["engine"],
            "wake_word": r["wake_word"],
            "values": [_fmt(r[c]) for c in value_cols],
        }
        for r in df.iter_rows(named=True)
    ]
    return {"columns": value_cols, "rows": rows}


def build_context(runs: Sequence[Run], *, title: str, fa_budget: float = 0.5) -> dict[str, Any]:
    """Assemble everything the template needs."""
    run_list = list(runs)
    board = leaderboard(run_list, fa_budget=fa_budget)
    wake_words: list[str] = (
        sorted(board.get_column("wake_word").unique().to_list()) if not board.is_empty() else []
    )
    boards: list[dict[str, Any]] = []
    for w in wake_words:
        sub = board.filter(pl.col("wake_word") == w)
        rows = []
        for rec in sub.iter_rows(named=True):
            rows.append(
                {
                    "rank": "-" if rec.get("rank") is None else str(rec["rank"]),
                    "engine": rec["engine"],
                    "verified": bool(rec["verified"]),
                    "preliminary": bool(rec["preliminary"]),
                    "aut": _fmt_ci(rec["aut"], rec["aut_lo"], rec["aut_hi"]),
                    "aut_bar": Markup(interval_bar_svg(rec["aut"], rec["aut_lo"], rec["aut_hi"])),
                    "budgets": [
                        _fmt_ci(rec[c], rec[f"{c}_lo"], rec[f"{c}_hi"]) for _, c in _BUDGETS
                    ],
                    "confusable": _fmt(rec.get("confusable_fa_per_1000"), 1),
                    "latency": _fmt(rec.get("latency_p50_s"), 2),
                    "rtf": _fmt(rec.get("rtf_median"), 3),
                    "n_positives": rec["n_positives"],
                    "n_units": rec["n_units"],
                    "negative_hours": _fmt(rec["negative_hours"], 1),
                    "manifest_version": rec["manifest_version"],
                }
            )
        boards.append(
            {
                "wake_word": w,
                "rows": rows,
                "det": Markup(
                    det_svg(
                        _curves_for(run_list, wake_word=w, slice_type="all", slice_value="all"),
                        title=f"DET: “{w}”, all engines, pooled",
                    )
                ),
            }
        )
    engine_plots: list[dict[str, Any]] = []
    for run in run_list:
        if run.curves.is_empty():
            continue
        for w in sorted(run.curves.get_column("wake_word").unique().to_list()):
            curves = _engine_language_curves(run, w)
            if not curves:
                continue
            plot_title = f"DET: {run.engine_label}, “{w}”, per language"
            engine_plots.append(
                {
                    "engine": run.engine_label,
                    "wake_word": w,
                    "det": Markup(det_svg(curves, title=plot_title)),
                }
            )
    gap = sealed_gap(run_list)
    provenance = [
        {
            "run_id": run.run_id,
            "engine": run.engine_label,
            "manifest_version": run.meta.get("manifest_version", "-"),
            "harness_version": run.meta.get("harness_version", "-"),
            "created_at": run.meta.get("created_at", "-"),
            "hardware": run.meta.get("hardware", "-"),
            "container_digest": (str(run.meta.get("container_digest") or "-"))[:19],
            "model_hashes": _short_hashes(run.meta.get("model_hashes")),  # type: ignore[arg-type]
            "detection": json.dumps(run.meta.get("detection", {}), sort_keys=True),
            "n_chunks": run.meta.get("n_chunks", "-"),
            "audio_hours": _fmt(run.meta.get("audio_hours"), 1),
            "verified": run.verified,
        }
        for run in run_list
    ]
    manifest_versions = sorted({str(r.meta.get("manifest_version", "?")) for r in run_list})
    return {
        "title": title,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "harness_version": __version__,
        "manifest_versions": manifest_versions,
        "fa_budget": fa_budget,
        "budget_labels": [f"FRR @ {b:g} FA/h" for b, _ in _BUDGETS],
        "min_positives": MIN_POSITIVES,
        "min_units": MIN_UNITS,
        "boards": boards,
        "per_language": _pivot_table(per_language(run_list, fa_budget=fa_budget)),
        "per_domain": _pivot_table(per_domain(run_list, fa_budget=fa_budget)),
        "engine_plots": engine_plots,
        "sealed_gap": []
        if gap.is_empty()
        else [
            {
                **row,
                "aut_public": _fmt(row["aut_public"]),
                "aut_sealed": _fmt(row["aut_sealed"]),
                "gap": _fmt(row["gap"]),
            }
            for row in gap.iter_rows(named=True)
        ],
        "provenance": provenance,
        "n_runs": len(run_list),
    }


def render_report(
    runs: Iterable[Run],
    *,
    title: str = "WakeWordWorld results",
    out_path: Path,
    fa_budget: float = 0.5,
) -> Path:
    """Render the HTML report to ``out_path`` and return it."""
    env = Environment(
        loader=PackageLoader("wakewordworld.report", "templates"),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    html = env.get_template("report.html.j2").render(
        **build_context(list(runs), title=title, fa_budget=fa_budget)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
