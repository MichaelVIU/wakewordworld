from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from wakewordworld.engines.base import FRAME_SAMPLES, Capabilities, EngineInfo
from wakewordworld.eval.protocol import ScoreTable, iter_frames, stream_chunk


class EnergyEngine:
    """Toy engine: score = normalised RMS of the frame."""

    def __init__(self) -> None:
        self._info = EngineInfo(
            engine_id="toy",
            version="0",
            wake_words=("beep",),
            capabilities=Capabilities(continuous_scores=True),
        )
        self.resets = 0

    @property
    def info(self) -> EngineInfo:
        return self._info

    def reset(self) -> None:
        self.resets += 1

    def process(self, frame: np.ndarray) -> dict[str, float]:
        assert frame.shape == (FRAME_SAMPLES,)
        assert frame.dtype == np.int16
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))
        return {"beep": min(1.0, rms / 10000.0)}

    def close(self) -> None:
        pass


def _write(path: Path, seconds: float) -> None:
    n = int(16000 * seconds)
    x = np.zeros(n, dtype=np.int16)
    x[16000:32000] = 8000  # 1 s "beep" from 1.0 s to 2.0 s
    sf.write(path, x, 16000, subtype="PCM_16")


def test_iter_frames_pads_last(tmp_path: Path) -> None:
    p = tmp_path / "a.flac"
    _write(p, 2.55)  # 40800 samples = 31 full frames + 1120 remainder
    frames = list(iter_frames(p))
    assert len(frames) == 32
    assert all(f.shape == (FRAME_SAMPLES,) for f in frames)


def test_stream_chunk_scores_and_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "a.flac"
    _write(p, 3.2)
    eng = EnergyEngine()
    table, stats = stream_chunk(eng, p, chunk_id="c1")
    assert eng.resets == 1
    assert table.n_frames == 40
    assert stats.n_frames == 40
    assert abs(stats.audio_s - 3.2) < 1e-9
    s = table.scores[:, 0]
    assert s[5] == 0.0
    assert s[15] > 0.7
    assert s[30] == 0.0
    out = tmp_path / "scores.parquet"
    table.save(out)
    again = ScoreTable.load(out, "c1")
    assert again.wake_words == ("beep",)
    assert np.allclose(again.scores, table.scores)


def test_run_evaluation_end_to_end(tmp_path: Path) -> None:
    """A toy engine over a two-chunk manifest produces results with the expected layout."""
    import json

    import polars as pl

    from wakewordworld.eval.run import EvalPlan, run_evaluation
    from wakewordworld.util.paths import DataRoot

    root = DataRoot(tmp_path / "data")
    root.ensure()
    src = "toy"
    (root.chunks / src).mkdir(parents=True)
    rows = []
    for i, cid in enumerate(("c1", "c2")):
        _write(root.chunks / src / f"{cid}.flac", 30.0)
        rows.append(
            {
                "chunk_id": cid,
                "file_id": f"f{i}",
                "item_id": f"i{i}",
                "source_id": src,
                "language": "en",
                "licence_spdx": "CC0-1.0",
                "licence_tier": "A",
                "attribution": "toy",
                "item_url": "https://example.org/x",
                "start_s": 0.0,
                "duration_s": 30.0,
                "domain": "podcast",
                "microphone": "close",
            }
        )
    mdir = tmp_path / "manifests" / "0.0.1"
    mdir.mkdir(parents=True)
    (mdir / "chunks.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (mdir / "release.json").write_text(json.dumps({"version": "0.0.1"}), encoding="utf-8")
    # word index: the "beep" occurs 1.0-2.0 s in c1 only
    pl.DataFrame(
        {"chunk_id": ["c1"], "word": ["beep"], "start_s": [1.0], "end_s": [2.0]}
    ).write_parquet(root.index / f"{src}.parquet")
    plan = EvalPlan(
        manifest_dir=mdir,
        engine_id="toy",
        engine_config={},
        wake_words={"beep": "beep"},
        n_boot=20,
        confusables={"beep": []},
    )
    run_dir = run_evaluation(root, EnergyEngine(), plan, tmp_path / "results")
    summary = pl.read_parquet(run_dir / "summary.parquet")
    all_row = summary.filter(pl.col("slice_type") == "all").row(0, named=True)
    assert all_row["n_positives"] == 1
    assert all_row["n_units"] == 2
    assert (run_dir / "run.json").exists()
    assert (run_dir / "curves.parquet").exists()
    assert (root.scores / "toy").exists()
