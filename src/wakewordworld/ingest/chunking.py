"""Silence-aware chunking of normalised files into evaluation chunks.

Policy (all values configurable):

* files shorter than ``max_len_s`` (20 min) become one chunk;
* otherwise, within the window ``[min_len_s, max_len_s]`` after the previous cut, cut
  at the middle of the longest silence; if there is none, cut at ``max_len_s``;
* a trailing piece shorter than ``min_tail_s`` (60 s) is merged into the previous chunk.

Silence is detected on 20 ms RMS frames: a frame is silent when its level is below
``min(max(abs_floor_dbfs, noise_floor + margin), median - 10 dB)`` where the noise
floor is the 5th percentile of frame levels; a silence must last at least
``min_silence_s``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from wakewordworld.ingest.records import ChunkRecord, chunk_id_for
from wakewordworld.util.paths import DataRoot

__all__ = [
    "ChunkPolicy",
    "Silence",
    "cut_points",
    "detect_silences",
    "frame_levels",
    "write_chunks",
]


@dataclass(frozen=True)
class ChunkPolicy:
    """Chunking parameters in seconds / dBFS."""

    min_len_s: float = 10 * 60
    target_len_s: float = 15 * 60
    max_len_s: float = 20 * 60
    min_tail_s: float = 60.0
    frame_s: float = 0.02
    abs_floor_dbfs: float = -45.0
    noise_margin_db: float = 6.0
    min_silence_s: float = 0.4


@dataclass(frozen=True)
class Silence:
    """A silent interval."""

    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        """Length in seconds."""
        return self.end_s - self.start_s

    @property
    def mid_s(self) -> float:
        """Midpoint in seconds."""
        return (self.start_s + self.end_s) / 2


@dataclass
class _Cut:
    at_s: float
    reason: str
    silences: list[Silence] = field(default_factory=list)


def frame_levels(path: Path, frame_s: float = 0.02, block_frames: int = 1 << 20) -> np.ndarray:
    """RMS level in dBFS per frame, streamed from disk."""
    levels: list[np.ndarray] = []
    carry = np.zeros(0, dtype=np.float32)
    with sf.SoundFile(str(path)) as f:
        n = max(1, round(f.samplerate * frame_s))
        block_frames = (block_frames // n) * n or n
        for block in f.blocks(blocksize=block_frames, dtype="float32", always_2d=True):
            x = np.concatenate([carry, block[:, 0]])
            usable = (x.shape[0] // n) * n
            frames = x[:usable].reshape(-1, n)
            carry = x[usable:]
            if frames.size:
                rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
                levels.append(20 * np.log10(np.maximum(rms, 1e-9)))
    if carry.size:
        tail_rms = float(np.sqrt(np.mean(carry.astype(np.float64) ** 2)))
        levels.append(np.array([20 * np.log10(max(tail_rms, 1e-9))]))
    return np.concatenate(levels) if levels else np.zeros(0)


def detect_silences(levels: np.ndarray, policy: ChunkPolicy) -> list[Silence]:
    """Silent intervals from per-frame levels."""
    if levels.size == 0:
        return []
    noise_floor = float(np.percentile(levels, 5))
    median = float(np.median(levels))
    threshold = max(policy.abs_floor_dbfs, noise_floor + policy.noise_margin_db)
    # Never classify the bulk of the signal as silence: stay well below the median.
    threshold = min(threshold, median - 10.0)
    silent = levels < threshold
    min_frames = max(1, round(policy.min_silence_s / policy.frame_s))
    out: list[Silence] = []
    i = 0
    n = silent.shape[0]
    while i < n:
        if not silent[i]:
            i += 1
            continue
        j = i
        while j < n and silent[j]:
            j += 1
        if j - i >= min_frames:
            out.append(Silence(i * policy.frame_s, j * policy.frame_s))
        i = j
    return out


def cut_points(duration_s: float, silences: list[Silence], policy: ChunkPolicy) -> list[_Cut]:
    """Chunk boundaries as ``(start, reason)`` pairs; the first chunk starts at 0."""
    cuts: list[_Cut] = [_Cut(0.0, "start")]
    if duration_s <= policy.max_len_s:
        return cuts
    pos = 0.0
    while duration_s - pos > policy.max_len_s:
        lo, hi = pos + policy.min_len_s, pos + policy.max_len_s
        window = [s for s in silences if s.mid_s >= lo and s.mid_s <= hi]
        if window:
            best = max(
                window, key=lambda s: (s.duration_s, -abs(s.mid_s - pos - policy.target_len_s))
            )
            cut = _Cut(best.mid_s, "silence")
        else:
            cut = _Cut(hi, "max_length")
        # Avoid leaving a tail shorter than min_tail_s.
        if duration_s - cut.at_s < policy.min_tail_s:
            break
        cuts.append(cut)
        pos = cut.at_s
    return cuts


def _sha256_chunk(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_chunks(
    audio_path: Path,
    *,
    file_id: str,
    source_id: str,
    data_root: DataRoot,
    policy: ChunkPolicy | None = None,
) -> list[ChunkRecord]:
    """Cut a normalised file into chunk FLACs and return their records.

    Existing chunk files are reused (not rewritten) when present.
    """
    policy = policy or ChunkPolicy()
    levels = frame_levels(audio_path, policy.frame_s)
    silences = detect_silences(levels, policy)
    with sf.SoundFile(str(audio_path)) as f:
        sr = f.samplerate
        total_frames = f.frames
    duration_s = total_frames / sr
    cuts = cut_points(duration_s, silences, policy)
    starts = [c.at_s for c in cuts]
    ends = [*starts[1:], duration_s]
    reasons = [*(c.reason for c in cuts[1:]), "file_end"]
    out_dir = data_root.chunks / source_id
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[ChunkRecord] = []
    with sf.SoundFile(str(audio_path)) as f:
        for start, end, reason in zip(starts, ends, reasons, strict=True):
            start_frame = round(start * sr)
            end_frame = round(end * sr)
            if end_frame <= start_frame:
                continue
            cid = chunk_id_for(file_id, round(start * 1000))
            out = out_dir / f"{cid}.flac"
            if not out.exists():
                tmp = out.with_suffix(".tmp.flac")
                f.seek(start_frame)
                remaining = end_frame - start_frame
                with sf.SoundFile(
                    str(tmp), mode="w", samplerate=sr, channels=1, subtype="PCM_16", format="FLAC"
                ) as w:
                    while remaining > 0:
                        block = f.read(min(remaining, 1 << 20), dtype="int16", always_2d=True)
                        if block.shape[0] == 0:
                            break
                        w.write(block[:, :1])
                        remaining -= block.shape[0]
                tmp.replace(out)
            records.append(
                ChunkRecord(
                    chunk_id=cid,
                    file_id=file_id,
                    source_id=source_id,
                    start_s=float(start),
                    duration_s=float((end_frame - start_frame) / sr),
                    audio_path=str(out.relative_to(data_root.root)),
                    audio_sha256=_sha256_chunk(out),
                    cut_reason=reason if len(cuts) > 1 or reason == "file_end" else "file_end",
                )
            )
    return records
