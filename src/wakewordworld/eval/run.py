"""Orchestrate an evaluation run: manifest + word index + engine -> results.

The run scores every chunk once (cached per engine version), then derives counts,
curves, slices, intervals and latencies from the stored scores.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray

from wakewordworld.engines.base import Engine
from wakewordworld.eval.counts import chunk_counts
from wakewordworld.eval.detect import (
    DetectionConfig,
    detections_from_scores,
    match_detections,
)
from wakewordworld.eval.metrics import (
    DEFAULT_FA_TARGETS,
    CountTable,
    bootstrap_summary,
    det_curve,
    summarise,
    threshold_grid,
)
from wakewordworld.eval.protocol import ScoreTable, StreamStats, stream_chunk
from wakewordworld.eval.results import RunMeta, SummaryRow, write_run
from wakewordworld.eval.targets import find_occurrences, occurrences_by_chunk
from wakewordworld.manifest.schema import ChunkRow
from wakewordworld.sources.spec import Language
from wakewordworld.util.hashing import short_id
from wakewordworld.util.paths import DataRoot

__all__ = ["EvalPlan", "load_manifest_rows", "run_evaluation", "target_phrase"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalPlan:
    """What to evaluate."""

    manifest_dir: Path
    engine_id: str
    engine_config: Mapping[str, object]
    wake_words: Mapping[str, str]
    """engine wake word key -> target phrase (normalised tokens separated by spaces)."""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    languages: tuple[str, ...] = ()
    n_boot: int = 1000
    force_rescore: bool = False
    limit_chunks: int | None = None
    confusables: Mapping[str, Sequence[str]] | None = None
    """target phrase -> near-miss words; computed from the phonetic index when None."""


def target_phrase(engine_key: str) -> str:
    """Default mapping from an engine's wake word key to index tokens."""
    key = engine_key.lower()
    for suffix in ("_v0.1", "_v1", "_v2"):
        key = key.removesuffix(suffix)
    return " ".join(part for part in key.replace("-", "_").split("_") if part)


def load_manifest_rows(manifest_dir: Path) -> list[ChunkRow]:
    """Read ``chunks.jsonl`` (and ``sealed/chunks.jsonl`` when present)."""
    rows: list[ChunkRow] = []
    for name in ("chunks.jsonl", "sealed/chunks.jsonl"):
        p = manifest_dir / name
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as fh:
            rows.extend(ChunkRow.model_validate_json(line) for line in fh if line.strip())
    if not rows:
        msg = f"no chunks found in {manifest_dir}"
        raise FileNotFoundError(msg)
    return rows


def _manifest_version(manifest_dir: Path) -> str:
    p = manifest_dir / "release.json"
    if p.exists():
        return str(json.loads(p.read_text(encoding="utf-8")).get("version", manifest_dir.name))
    return manifest_dir.name


def _chunk_audio(data_root: DataRoot, row: ChunkRow) -> Path:
    return data_root.chunks / row.source_id / f"{row.chunk_id}.flac"


def _load_index(data_root: DataRoot, source_ids: Iterable[str]) -> pl.DataFrame:
    frames = []
    for sid in sorted(set(source_ids)):
        p = data_root.index / f"{sid}.parquet"
        if p.exists():
            frames.append(pl.read_parquet(p, columns=["chunk_id", "word", "start_s", "end_s"]))
    if not frames:
        return pl.DataFrame(
            schema={
                "chunk_id": pl.Utf8,
                "word": pl.Utf8,
                "start_s": pl.Float64,
                "end_s": pl.Float64,
            }
        )
    return pl.concat(frames)


def _confusables_for(
    data_root: DataRoot, phrase: str, language: str, index: pl.DataFrame
) -> list[str]:
    """Near-miss words from the phonetic index; empty when espeak-ng is unavailable."""
    if " " in phrase:
        return []
    try:
        from wakewordworld.index.phonetic import near_misses
    except ImportError:  # pragma: no cover - module built in another milestone
        return []
    vocab = index.get_column("word").unique().to_list()
    try:
        pairs = near_misses(phrase, vocab, Language(language), data_root=data_root)
    except (RuntimeError, OSError, TypeError) as exc:
        log.warning("near-miss lookup unavailable for %r (%s)", phrase, exc)
        return []
    return [w for w, _ in pairs]


def _scores_dir(data_root: DataRoot, engine: Engine) -> Path:
    ident = short_id(
        engine.info.engine_id,
        engine.info.version,
        json.dumps(dict(engine.info.model_hashes), sort_keys=True),
        json.dumps(dict(engine.info.config), sort_keys=True, default=str),
    )
    return data_root.scores / engine.info.engine_id / ident


def _score_all(
    engine: Engine, rows: list[ChunkRow], data_root: DataRoot, *, force: bool
) -> tuple[dict[str, ScoreTable], dict[str, StreamStats]]:
    out_dir = _scores_dir(data_root, engine)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables: dict[str, ScoreTable] = {}
    stats: dict[str, StreamStats] = {}
    stats_path = out_dir / "stats.jsonl"
    cached_stats: dict[str, StreamStats] = {}
    if stats_path.exists():
        for line in stats_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                cached_stats[d["chunk_id"]] = StreamStats(**d)
    with stats_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            p = out_dir / f"{row.chunk_id}.parquet"
            if p.exists() and not force and row.chunk_id in cached_stats:
                tables[row.chunk_id] = ScoreTable.load(p, row.chunk_id)
                stats[row.chunk_id] = cached_stats[row.chunk_id]
                continue
            audio = _chunk_audio(data_root, row)
            if not audio.exists():
                log.warning("missing audio for chunk %s (%s)", row.chunk_id, audio)
                continue
            table, st = stream_chunk(engine, audio, chunk_id=row.chunk_id)
            table.save(p)
            fh.write(json.dumps(st.__dict__) + "\n")
            fh.flush()
            tables[row.chunk_id] = table
            stats[row.chunk_id] = st
    return tables, stats


def _slices(rows: list[ChunkRow]) -> dict[tuple[str, str], set[str]]:
    """Slice definitions: (type, value) -> set of chunk ids."""
    out: dict[tuple[str, str], set[str]] = {("all", "all"): {r.chunk_id for r in rows}}
    for r in rows:
        out.setdefault(("language", str(r.language)), set()).add(r.chunk_id)
        out.setdefault(("domain", str(r.domain)), set()).add(r.chunk_id)
        out.setdefault(("tier", str(r.licence_tier)), set()).add(r.chunk_id)
        out.setdefault(("source", r.source_id), set()).add(r.chunk_id)
        out.setdefault(("sealed", "yes" if r.sealed else "no"), set()).add(r.chunk_id)
    return out


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def run_evaluation(data_root: DataRoot, engine: Engine, plan: EvalPlan, out_dir: Path) -> Path:
    """Execute a plan and write results; returns the run directory."""
    rows = load_manifest_rows(plan.manifest_dir)
    if plan.languages:
        rows = [r for r in rows if r.language in plan.languages]
    if plan.limit_chunks is not None:
        rows = rows[: plan.limit_chunks]
    if not rows:
        msg = "no chunks selected"
        raise ValueError(msg)
    row_by_id = {r.chunk_id: r for r in rows}
    index = _load_index(data_root, (r.source_id for r in rows))

    tables, stats = _score_all(engine, rows, data_root, force=plan.force_rescore)
    rows = [r for r in rows if r.chunk_id in tables]
    slices = _slices(rows)
    words = engine.info.wake_words
    continuous = engine.info.capabilities.continuous_scores

    summary_rows: list[SummaryRow] = []
    curve_frames: list[pl.DataFrame] = []
    chunk_frames: list[dict[str, object]] = []

    for key, phrase in plan.wake_words.items():
        if key not in words:
            log.warning("engine has no wake word %r; skipping", key)
            continue
        j = words.index(key)
        if plan.confusables is not None:
            confusables = list(plan.confusables.get(phrase, []))
        else:
            lang = str(rows[0].language) if len({r.language for r in rows}) == 1 else "en"
            confusables = _confusables_for(data_root, phrase, lang, index)
        occ_df = find_occurrences(index, phrase, confusables=confusables)
        occ = occurrences_by_chunk(occ_df)
        all_scores = np.concatenate([t.scores[:, j] for t in tables.values()])
        thresholds = threshold_grid(all_scores) if continuous else np.array([0.5])

        # Per-chunk counts once; slices aggregate them.
        per_chunk: dict[
            str, tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], float]
        ] = {}
        for cid, t in tables.items():
            h, m, f, neg, _ = chunk_counts(
                t.scores[:, j],
                row_by_id[cid].duration_s,
                occ.get(cid, []),
                thresholds,
                plan.detection,
            )
            per_chunk[cid] = (h, m, f, neg)

        for (stype, svalue), chunk_ids in sorted(slices.items()):
            ids = [c for c in chunk_ids if c in per_chunk]
            if not ids:
                continue
            units = sorted({row_by_id[c].file_id for c in ids})
            uidx = {u: i for i, u in enumerate(units)}
            hits = np.zeros((len(units), thresholds.size), dtype=np.int64)
            misses = np.zeros_like(hits)
            fas = np.zeros_like(hits)
            neg_s = np.zeros(len(units), dtype=np.float64)
            for c in ids:
                h, m, f, n = per_chunk[c]
                i = uidx[row_by_id[c].file_id]
                hits[i] += h
                misses[i] += m
                fas[i] += f
                neg_s[i] += n
            table = CountTable(thresholds, hits, misses, fas, neg_s, units)
            summ = summarise(table, DEFAULT_FA_TARGETS)
            ci = bootstrap_summary(table, fa_targets=DEFAULT_FA_TARGETS, n_boot=plan.n_boot)
            frr_ci = ci["frr_at_fa"]
            aut_ci = ci["aut"]
            assert isinstance(frr_ci, dict)
            assert isinstance(aut_ci, tuple)

            # Latencies and confusable false accepts at the 0.5 FA/h operating point;
            # when no threshold reaches that budget (typical for ASR-grammar baselines),
            # fall back to the operating point with the fewest false accepts so the
            # latency column is still informative.
            thr05 = summ.threshold_at_fa[0.5]
            if thr05 is None:
                curve_pts = det_curve(table).points
                finite = [p for p in curve_pts if p.fa_per_hour == p.fa_per_hour]
                if finite:
                    thr05 = min(finite, key=lambda p: (p.fa_per_hour, p.frr)).threshold
            latencies: list[float] = []
            n_conf_fa = 0
            n_conf = sum(1 for c in ids for o in occ.get(c, []) if o.confusable)
            if thr05 is not None:
                for c in ids:
                    dets = detections_from_scores(tables[c].scores[:, j], thr05, plan.detection)
                    m_res = match_detections(dets, occ.get(c, []), plan.detection)
                    latencies.extend(m_res.latencies())
                    n_conf_fa += len(m_res.confusable_false_accepts)
            rtfs = [stats[c].rtf for c in ids if c in stats]

            summary_rows.append(
                SummaryRow(
                    wake_word=phrase,
                    slice_type=stype,
                    slice_value=svalue,
                    n_units=len(units),
                    n_chunks=len(ids),
                    n_positives=summ.n_positives,
                    n_confusable=n_conf,
                    negative_hours=summ.negative_hours,
                    frr_at_0_1=summ.frr_at_fa[0.1],
                    frr_at_0_5=summ.frr_at_fa[0.5],
                    frr_at_1=summ.frr_at_fa[1.0],
                    frr_at_3=summ.frr_at_fa[3.0],
                    frr_at_0_1_lo=frr_ci[0.1][0],
                    frr_at_0_1_hi=frr_ci[0.1][1],
                    frr_at_0_5_lo=frr_ci[0.5][0],
                    frr_at_0_5_hi=frr_ci[0.5][1],
                    frr_at_1_lo=frr_ci[1.0][0],
                    frr_at_1_hi=frr_ci[1.0][1],
                    frr_at_3_lo=frr_ci[3.0][0],
                    frr_at_3_hi=frr_ci[3.0][1],
                    threshold_at_0_5=thr05,
                    eer=summ.eer,
                    aut=summ.aut,
                    aut_lo=aut_ci[0],
                    aut_hi=aut_ci[1],
                    confusable_fa_per_1000=(1000.0 * n_conf_fa / n_conf) if n_conf else None,
                    latency_p50_s=_percentile(latencies, 50),
                    latency_p90_s=_percentile(latencies, 90),
                    latency_p99_s=_percentile(latencies, 99),
                    rtf_median=_percentile(rtfs, 50),
                )
            )
            curve = det_curve(table)
            curve_frames.append(
                pl.DataFrame(
                    {
                        "wake_word": [phrase] * len(curve.points),
                        "slice_type": [stype] * len(curve.points),
                        "slice_value": [svalue] * len(curve.points),
                        "threshold": [p.threshold for p in curve.points],
                        "fa_per_hour": [p.fa_per_hour for p in curve.points],
                        "frr": [p.frr for p in curve.points],
                    }
                )
            )

        for c in per_chunk:
            r = row_by_id[c]
            chunk_frames.append(
                {
                    "wake_word": phrase,
                    "chunk_id": c,
                    "file_id": r.file_id,
                    "source_id": r.source_id,
                    "language": str(r.language),
                    "duration_s": r.duration_s,
                    "n_occurrences": sum(1 for o in occ.get(c, []) if not o.confusable),
                    "rtf": stats[c].rtf if c in stats else None,
                }
            )

    run_id = short_id(
        _manifest_version(plan.manifest_dir),
        engine.info.engine_id,
        engine.info.version,
        json.dumps(dict(engine.info.model_hashes), sort_keys=True),
        json.dumps(dict(plan.wake_words), sort_keys=True),
        json.dumps(plan.detection.__dict__, sort_keys=True),
        length=12,
    )
    meta = RunMeta.new(
        run_id=run_id,
        manifest_version=_manifest_version(plan.manifest_dir),
        manifest_dir=str(plan.manifest_dir),
        engine_id=engine.info.engine_id,
        engine_version=engine.info.version,
        model_hashes=dict(engine.info.model_hashes),
        engine_config=dict(engine.info.config),
        wake_words=dict(plan.wake_words),
        detection=dict(plan.detection.__dict__),
        n_chunks=len(rows),
        audio_hours=sum(r.duration_s for r in rows) / 3600.0,
        languages=sorted({str(r.language) for r in rows}),
    )
    run_dir = out_dir / run_id
    curves = (
        pl.concat(curve_frames)
        if curve_frames
        else pl.DataFrame(
            schema={
                "wake_word": pl.Utf8,
                "slice_type": pl.Utf8,
                "slice_value": pl.Utf8,
                "threshold": pl.Float64,
                "fa_per_hour": pl.Float64,
                "frr": pl.Float64,
            }
        )
    )
    write_run(run_dir, meta, summary_rows, curves, pl.DataFrame(chunk_frames))
    return run_dir
