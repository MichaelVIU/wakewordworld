"""Streaming protocol: feed a chunk to an engine and record raw per-frame scores.

The score table is the primary artefact of an evaluation run. Everything else
(detections, curves, metrics) is derived from it and can be recomputed with different
thresholds or detection settings without touching the engine again.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import soundfile as sf
from numpy.typing import NDArray

from wakewordworld.engines.base import FRAME_SAMPLES, SAMPLE_RATE, Engine, pad_frame

__all__ = ["ScoreTable", "StreamStats", "iter_frames", "score_stream", "stream_chunk"]

SCORE_SCHEMA: dict[str, type[pl.DataType]] = {
    "chunk_id": pl.Utf8,
    "wake_word": pl.Utf8,
    "frame": pl.Int32,
    "t_end_s": pl.Float32,
    "score": pl.Float32,
}


@dataclass(frozen=True)
class StreamStats:
    """Timing of one chunk through one engine."""

    chunk_id: str
    audio_s: float
    wall_s: float
    n_frames: int

    @property
    def rtf(self) -> float:
        """Real-time factor (wall time / audio time)."""
        return self.wall_s / self.audio_s if self.audio_s > 0 else float("nan")


@dataclass
class ScoreTable:
    """Per-frame scores of one chunk, columnar."""

    chunk_id: str
    wake_words: tuple[str, ...]
    scores: NDArray[np.float32]
    """Shape ``(n_frames, n_wake_words)``."""

    @property
    def n_frames(self) -> int:
        """Number of frames."""
        return int(self.scores.shape[0])

    def frame_end_times(self) -> NDArray[np.float32]:
        """End time in seconds of each frame."""
        t = (np.arange(self.n_frames, dtype=np.float64) + 1.0) * FRAME_SAMPLES / SAMPLE_RATE
        return t.astype(np.float32)

    def to_polars(self) -> pl.DataFrame:
        """Long-format frame with one row per (frame, wake word)."""
        t = self.frame_end_times()
        frames: list[pl.DataFrame] = []
        for j, w in enumerate(self.wake_words):
            frames.append(
                pl.DataFrame(
                    {
                        "chunk_id": pl.Series([self.chunk_id] * self.n_frames, dtype=pl.Utf8),
                        "wake_word": pl.Series([w] * self.n_frames, dtype=pl.Utf8),
                        "frame": pl.Series(np.arange(self.n_frames, dtype=np.int32)),
                        "t_end_s": pl.Series(t),
                        "score": pl.Series(self.scores[:, j].astype(np.float32)),
                    }
                )
            )
        if not frames:
            return pl.DataFrame(schema=SCORE_SCHEMA)
        return pl.concat(frames)

    @classmethod
    def from_polars(cls, df: pl.DataFrame, chunk_id: str) -> ScoreTable:
        """Rebuild from long format."""
        sub = df.filter(pl.col("chunk_id") == chunk_id).sort(["wake_word", "frame"])
        words = tuple(sub.get_column("wake_word").unique(maintain_order=True).to_list())
        n = int(sub.get_column("frame").to_numpy().max()) + 1 if sub.height else 0
        arr = np.zeros((n, len(words)), dtype=np.float32)
        for j, w in enumerate(words):
            s = sub.filter(pl.col("wake_word") == w)
            arr[s.get_column("frame").to_numpy(), j] = s.get_column("score").to_numpy()
        return cls(chunk_id=chunk_id, wake_words=words, scores=arr)

    def save(self, path: Path) -> None:
        """Write Parquet."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_polars().write_parquet(path, compression="zstd")

    @classmethod
    def load(cls, path: Path, chunk_id: str) -> ScoreTable:
        """Read Parquet."""
        return cls.from_polars(pl.read_parquet(path), chunk_id)


def iter_frames(path: Path) -> Iterator[NDArray[np.int16]]:
    """Yield 80 ms int16 frames from a 16 kHz mono file; the last frame is zero-padded."""
    with sf.SoundFile(path) as fh:
        if fh.samplerate != SAMPLE_RATE:
            msg = f"{path}: expected {SAMPLE_RATE} Hz, got {fh.samplerate}"
            raise ValueError(msg)
        if fh.channels != 1:
            msg = f"{path}: expected mono, got {fh.channels} channels"
            raise ValueError(msg)
        while True:
            block = fh.read(FRAME_SAMPLES, dtype="int16", always_2d=False)
            if block.shape[0] == 0:
                break
            yield pad_frame(np.ascontiguousarray(block, dtype=np.int16))
            if block.shape[0] < FRAME_SAMPLES:
                break


def score_stream(
    engine: Engine, frames: Iterator[NDArray[np.int16]], *, chunk_id: str
) -> tuple[ScoreTable, int]:
    """Run frames through a (reset) engine; returns the score table and frame count."""
    words = engine.info.wake_words
    idx = {w: j for j, w in enumerate(words)}
    rows: list[NDArray[np.float32]] = []
    engine.reset()
    for frame in frames:
        result = engine.process(frame)
        row = np.zeros(len(words), dtype=np.float32)
        for w, s in result.items():
            j = idx.get(w)
            if j is not None:
                row[j] = float(min(1.0, max(0.0, s)))
        rows.append(row)
    scores = np.vstack(rows) if rows else np.zeros((0, len(words)), dtype=np.float32)
    return ScoreTable(chunk_id=chunk_id, wake_words=words, scores=scores), len(rows)


def stream_chunk(
    engine: Engine, audio_path: Path, *, chunk_id: str
) -> tuple[ScoreTable, StreamStats]:
    """Score one chunk file and measure wall time."""
    t0 = time.perf_counter()
    table, n = score_stream(engine, iter_frames(audio_path), chunk_id=chunk_id)
    wall = time.perf_counter() - t0
    audio_s = n * FRAME_SAMPLES / SAMPLE_RATE
    return table, StreamStats(chunk_id=chunk_id, audio_s=audio_s, wall_s=wall, n_frames=n)
