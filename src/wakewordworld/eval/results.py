"""Result artefacts of an evaluation run.

A run directory (``results/<run_id>/``) contains:

* ``run.json`` — :class:`RunMeta`: what was evaluated, with what, on which hardware.
* ``summary.parquet`` — one row per (wake word, slice): headline metrics with intervals.
* ``curves.parquet`` — DET points per (wake word, slice, threshold).
* ``chunks.parquet`` — per-chunk timing (real-time factor) and counts at the 0.5 FA/h
  operating point.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from wakewordworld import __version__

__all__ = ["RunMeta", "SummaryRow", "hardware_string", "write_run"]


def hardware_string() -> str:
    """Short description of the machine the run executed on."""
    return f"{platform.system()} {platform.machine()} {platform.processor() or ''}".strip()


@dataclass
class RunMeta:
    """Provenance of a run."""

    run_id: str
    created_at: str
    harness_version: str
    manifest_version: str
    manifest_dir: str
    engine_id: str
    engine_version: str
    model_hashes: dict[str, str]
    engine_config: dict[str, object]
    wake_words: dict[str, str]
    """engine wake word key -> target phrase in the word index."""
    detection: dict[str, float]
    n_chunks: int
    audio_hours: float
    hardware: str
    container_digest: str | None = None
    notes: str | None = None
    languages: list[str] = field(default_factory=list)

    @classmethod
    def new(cls, **kwargs: object) -> RunMeta:
        """Create with timestamp, harness version and hardware filled in."""
        return cls(
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            harness_version=__version__,
            hardware=hardware_string(),
            **kwargs,  # type: ignore[arg-type]
        )


@dataclass
class SummaryRow:
    """Headline metrics for one wake word on one slice."""

    wake_word: str
    slice_type: str
    slice_value: str
    n_units: int
    n_chunks: int
    n_positives: int
    n_confusable: int
    negative_hours: float
    frr_at_0_1: float | None
    frr_at_0_5: float | None
    frr_at_1: float | None
    frr_at_3: float | None
    frr_at_0_1_lo: float | None
    frr_at_0_1_hi: float | None
    frr_at_0_5_lo: float | None
    frr_at_0_5_hi: float | None
    frr_at_1_lo: float | None
    frr_at_1_hi: float | None
    frr_at_3_lo: float | None
    frr_at_3_hi: float | None
    threshold_at_0_5: float | None
    eer: float
    aut: float
    aut_lo: float
    aut_hi: float
    confusable_fa_per_1000: float | None
    latency_p50_s: float | None
    latency_p90_s: float | None
    latency_p99_s: float | None
    rtf_median: float | None


def write_run(
    out_dir: Path,
    meta: RunMeta,
    summary: list[SummaryRow],
    curves: pl.DataFrame,
    chunks: pl.DataFrame,
) -> None:
    """Write all run artefacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
    pl.DataFrame([asdict(r) for r in summary]).write_parquet(out_dir / "summary.parquet")
    curves.write_parquet(out_dir / "curves.parquet")
    chunks.write_parquet(out_dir / "chunks.parquet")
    # A human-readable copy of the summary for quick inspection and diffs.
    pl.DataFrame([asdict(r) for r in summary]).write_csv(out_dir / "summary.csv")
