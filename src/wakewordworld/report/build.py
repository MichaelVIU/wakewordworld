"""Load evaluation runs and build leaderboard tables.

Ranking rule (see ``docs/methodology/metrics.md``): engines are ordered by the
pessimistic end of their AUT interval (``aut_hi`` ascending, then ``aut``). A row is
*preliminary* when it has fewer than 100 positives or fewer than 50 parent files for
the wake word in question; preliminary rows are shown but not ranked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import polars as pl

__all__ = [
    "MIN_POSITIVES",
    "MIN_UNITS",
    "Run",
    "export_json",
    "leaderboard",
    "load_run",
    "load_runs",
    "per_domain",
    "per_language",
    "sealed_gap",
]

MIN_POSITIVES = 100
MIN_UNITS = 50

_BUDGET_COLS: dict[float, str] = {
    0.1: "frr_at_0_1",
    0.5: "frr_at_0_5",
    1.0: "frr_at_1",
    3.0: "frr_at_3",
}


@dataclass(frozen=True)
class Run:
    """One evaluation run directory, loaded."""

    path: Path
    meta: dict[str, object]
    summary: pl.DataFrame
    curves: pl.DataFrame
    chunks: pl.DataFrame

    @property
    def run_id(self) -> str:
        """Run identifier."""
        return str(self.meta.get("run_id", self.path.name))

    @property
    def engine_id(self) -> str:
        """Engine identifier."""
        return str(self.meta.get("engine_id", "?"))

    @property
    def engine_version(self) -> str:
        """Engine version string."""
        return str(self.meta.get("engine_version", "?"))

    @property
    def engine_label(self) -> str:
        """Engine id and version for tables and legends."""
        return f"{self.engine_id} {self.engine_version}"

    @property
    def verified(self) -> bool:
        """Whether maintainers marked the run verified (``run.json`` key ``verified``)."""
        return bool(self.meta.get("verified", False))


def load_run(path: Path) -> Run:
    """Load one run directory."""
    meta = json.loads((path / "run.json").read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        msg = f"{path}/run.json must contain an object"
        raise ValueError(msg)
    chunks_path = path / "chunks.parquet"
    return Run(
        path=path,
        meta=meta,
        summary=pl.read_parquet(path / "summary.parquet"),
        curves=pl.read_parquet(path / "curves.parquet"),
        chunks=pl.read_parquet(chunks_path) if chunks_path.exists() else pl.DataFrame(),
    )


def load_runs(results_root: Path) -> list[Run]:
    """Load every run directory (one containing ``run.json``) under a root."""
    runs = [load_run(p.parent) for p in sorted(results_root.glob("*/run.json"))]
    return sorted(runs, key=lambda r: (r.engine_id, r.engine_version, r.run_id))


def _budget_col(fa_budget: float) -> str:
    try:
        return _BUDGET_COLS[fa_budget]
    except KeyError as exc:
        msg = f"fa_budget must be one of {sorted(_BUDGET_COLS)}"
        raise ValueError(msg) from exc


def _rows(runs: list[Run], slice_type: str, slice_value: str | None) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for r in runs:
        sub = r.summary.filter(pl.col("slice_type") == slice_type)
        if slice_value is not None:
            sub = sub.filter(pl.col("slice_value") == slice_value)
        if sub.is_empty():
            continue
        frames.append(
            sub.with_columns(
                pl.lit(r.run_id).alias("run_id"),
                pl.lit(r.engine_id).alias("engine_id"),
                pl.lit(r.engine_version).alias("engine_version"),
                pl.lit(r.engine_label).alias("engine"),
                pl.lit(str(r.meta.get("manifest_version", "?"))).alias("manifest_version"),
                pl.lit(r.verified).alias("verified"),
            )
        )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def leaderboard(
    runs: list[Run],
    *,
    slice_type: str = "all",
    slice_value: str = "all",
    fa_budget: float = 0.5,
) -> pl.DataFrame:
    """One row per (engine, version, wake word), ranked by ``aut_hi`` then ``aut``.

    Adds ``frr_budget``/``frr_budget_lo``/``frr_budget_hi`` for the requested budget,
    ``preliminary`` and ``rank`` (null for preliminary rows).
    """
    df = _rows(runs, slice_type, slice_value)
    if df.is_empty():
        return df
    col = _budget_col(fa_budget)
    df = df.with_columns(
        pl.col(col).alias("frr_budget"),
        pl.col(f"{col}_lo").alias("frr_budget_lo"),
        pl.col(f"{col}_hi").alias("frr_budget_hi"),
        ((pl.col("n_positives") < MIN_POSITIVES) | (pl.col("n_units") < MIN_UNITS)).alias(
            "preliminary"
        ),
    ).sort(["wake_word", "preliminary", "aut_hi", "aut", "engine"], nulls_last=True)
    ranked = (
        df.filter(~pl.col("preliminary"))
        .with_columns(pl.int_range(1, pl.len() + 1).over("wake_word").alias("rank"))
        .select(["wake_word", "engine", "rank"])
    )
    return df.join(ranked, on=["wake_word", "engine"], how="left")


def per_language(runs: list[Run], *, fa_budget: float = 0.5) -> pl.DataFrame:
    """Pivot: rows engine x wake word, columns languages, values FRR at the budget."""
    return _pivot(runs, "language", fa_budget)


def per_domain(runs: list[Run], *, fa_budget: float = 0.5) -> pl.DataFrame:
    """Pivot: rows engine x wake word, columns domains, values FRR at the budget."""
    return _pivot(runs, "domain", fa_budget)


def _pivot(runs: list[Run], slice_type: str, fa_budget: float) -> pl.DataFrame:
    df = _rows(runs, slice_type, None)
    if df.is_empty():
        return df
    col = _budget_col(fa_budget)
    return (
        df.select(["engine", "wake_word", "slice_value", col])
        .pivot(
            on="slice_value", index=["engine", "wake_word"], values=col, aggregate_function="first"
        )
        .sort(["wake_word", "engine"])
    )


def sealed_gap(runs: list[Run]) -> pl.DataFrame:
    """AUT on sealed minus AUT on public chunks, per engine and wake word."""
    frames: list[pl.DataFrame] = []
    for r in runs:
        s = r.summary.filter(pl.col("slice_type") == "sealed")
        if s.is_empty():
            continue
        wide = s.select(["wake_word", "slice_value", "aut"]).pivot(
            on="slice_value", index="wake_word", values="aut", aggregate_function="first"
        )
        if "yes" not in wide.columns or "no" not in wide.columns:
            continue
        frames.append(
            wide.select(
                pl.lit(r.engine_label).alias("engine"),
                "wake_word",
                pl.col("no").alias("aut_public"),
                pl.col("yes").alias("aut_sealed"),
                (pl.col("yes") - pl.col("no")).alias("gap"),
            )
        )
    return pl.concat(frames) if frames else pl.DataFrame()


def export_json(runs: list[Run], path: Path, *, fa_budget: float = 0.5) -> None:
    """Write a machine-readable leaderboard (all slices) as JSON."""
    board = leaderboard(runs, fa_budget=fa_budget)
    lang = per_language(runs, fa_budget=fa_budget)
    dom = per_domain(runs, fa_budget=fa_budget)
    payload = {
        "schema": "wakewordworld-leaderboard/1",
        "fa_budget": fa_budget,
        "min_positives": MIN_POSITIVES,
        "min_units": MIN_UNITS,
        "runs": [
            {
                "run_id": r.run_id,
                "engine_id": r.engine_id,
                "engine_version": r.engine_version,
                "manifest_version": r.meta.get("manifest_version"),
                "harness_version": r.meta.get("harness_version"),
                "created_at": r.meta.get("created_at"),
                "hardware": r.meta.get("hardware"),
                "container_digest": r.meta.get("container_digest"),
                "model_hashes": r.meta.get("model_hashes", {}),
                "detection": r.meta.get("detection", {}),
                "verified": r.verified,
            }
            for r in runs
        ],
        "leaderboard": [] if board.is_empty() else board.to_dicts(),
        "per_language": [] if lang.is_empty() else lang.to_dicts(),
        "per_domain": [] if dom.is_empty() else dom.to_dicts(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
