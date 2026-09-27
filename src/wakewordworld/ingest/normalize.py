"""Audio normalisation with ffmpeg: 16 kHz mono 16-bit FLAC plus quality statistics."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from wakewordworld.ingest.records import TARGET_SAMPLE_RATE, FileRecord, file_id_for
from wakewordworld.util.hashing import sha256_file
from wakewordworld.util.paths import DataRoot

__all__ = [
    "FfmpegError",
    "MediaInfo",
    "clipping_ratio",
    "find_tool",
    "measure_loudness",
    "normalize_file",
    "probe",
]

log = logging.getLogger(__name__)

_EXTRA_PATHS = ("/opt/homebrew/bin", "/usr/local/bin")
_LUFS_RE = re.compile(r"I:\s+(-?[\d.]+|-inf)\s+LUFS")
_PEAK_RE = re.compile(r"Peak:\s+(-?[\d.]+|-inf)\s+dBFS")


class FfmpegError(RuntimeError):
    """ffmpeg/ffprobe failed; the message carries the stderr tail."""


def find_tool(name: str) -> str:
    """Locate an executable, also looking in common Homebrew paths."""
    found = shutil.which(name)
    if found:
        return found
    for base in _EXTRA_PATHS:
        candidate = Path(base) / name
        if candidate.exists():
            return str(candidate)
    msg = f"{name} not found on PATH (install ffmpeg / chromaprint)"
    raise FfmpegError(msg)


def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        msg = f"{cmd[0]} timed out after {timeout:.0f}s: {' '.join(cmd[:6])}"
        raise FfmpegError(msg) from exc
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-8:]
        msg = f"{cmd[0]} failed ({proc.returncode}): " + " | ".join(tail)
        raise FfmpegError(msg)
    return proc


@dataclass(frozen=True)
class MediaInfo:
    """Subset of ffprobe output."""

    duration_s: float
    sample_rate: int | None
    channels: int | None
    codec: str | None
    has_audio: bool


def probe(path: Path, *, timeout: float = 120.0) -> MediaInfo:
    """Read duration, sample rate and channel count with ffprobe."""
    cmd = [
        find_tool("ffprobe"),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    out = json.loads(_run(cmd, timeout).stdout or "{}")
    streams = [s for s in out.get("streams", []) if s.get("codec_type") == "audio"]
    fmt = out.get("format", {})
    duration = float(fmt.get("duration") or 0.0)
    if not streams:
        return MediaInfo(duration, None, None, None, False)
    s0 = streams[0]
    if not duration and s0.get("duration"):
        duration = float(s0["duration"])
    sr = int(s0["sample_rate"]) if s0.get("sample_rate") else None
    ch = int(s0["channels"]) if s0.get("channels") else None
    return MediaInfo(duration, sr, ch, s0.get("codec_name"), True)


def _soxr_available(ffmpeg: str) -> bool:
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-buildconf"], capture_output=True, text=True, check=False
        )
    except OSError:
        return False
    return "--enable-libsoxr" in proc.stdout


def measure_loudness(path: Path, *, timeout: float = 600.0) -> tuple[float | None, float | None]:
    """Integrated loudness (LUFS) and true peak (dBFS) via ffmpeg's ebur128 filter."""
    cmd = [
        find_tool("ffmpeg"),
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-af",
        "ebur128=peak=true",
        "-f",
        "null",
        "-",
    ]
    proc = _run(cmd, timeout)
    text = proc.stderr
    # The summary block is at the end; take the last matches.
    lufs_m = _LUFS_RE.findall(text)
    peak_m = _PEAK_RE.findall(text)

    def _num(v: str | None) -> float | None:
        if v is None or v == "-inf":
            return None
        return float(v)

    return _num(lufs_m[-1] if lufs_m else None), _num(peak_m[-1] if peak_m else None)


def clipping_ratio(path: Path, *, threshold: float = 0.999, block_frames: int = 1 << 20) -> float:
    """Fraction of samples at or above ``threshold`` of full scale."""
    clipped = 0
    total = 0
    with sf.SoundFile(str(path)) as f:
        for block in f.blocks(blocksize=block_frames, dtype="float32", always_2d=True):
            mono = block[:, 0]
            clipped += int(np.count_nonzero(np.abs(mono) >= threshold))
            total += mono.shape[0]
    return clipped / total if total else 0.0


def normalize_file(
    original: Path,
    *,
    item_id: str,
    source_id: str,
    original_sha256: str,
    data_root: DataRoot,
    member_path: str | None = None,
    measure: bool = True,
    timeout: float = 3600.0,
) -> FileRecord:
    """Transcode an original into the 16 kHz mono FLAC working copy.

    Args:
        original: Path to the downloaded (or extracted) media file.
        item_id: Fetched item id.
        source_id: Source id (used for the output directory).
        original_sha256: Checksum of ``original``.
        data_root: Data directory.
        member_path: Archive member path or parquet row marker, if any.
        measure: Whether to run the loudness pass and clipping analysis.
        timeout: Per-ffmpeg-call timeout in seconds.

    Returns:
        A :class:`FileRecord` describing the working copy.
    """
    info = probe(original)
    if not info.has_audio:
        msg = f"{original} has no audio stream"
        raise FfmpegError(msg)
    file_id = file_id_for(item_id, original_sha256)
    out_dir = data_root.audio / source_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{file_id}.flac"
    ffmpeg = find_tool("ffmpeg")
    if not out.exists():
        tmp = out.with_suffix(".flac.tmp.flac")
        af = ["-af", "aresample=resampler=soxr"] if _soxr_available(ffmpeg) else []
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(original),
            "-vn",
            "-sn",
            "-dn",
            "-map",
            "0:a:0",
            *af,
            "-ac",
            "1",
            "-ar",
            str(TARGET_SAMPLE_RATE),
            "-sample_fmt",
            "s16",
            "-c:a",
            "flac",
            "-compression_level",
            "8",
            str(tmp),
        ]
        _run(cmd, timeout)
        tmp.replace(out)
    with sf.SoundFile(str(out)) as f:
        duration = f.frames / f.samplerate
        sr = f.samplerate
    lufs: float | None = None
    peak: float | None = None
    clip: float | None = None
    if measure:
        lufs, peak = measure_loudness(out, timeout=timeout)
        clip = clipping_ratio(out)
    ext = "".join(original.suffixes[-2:]).lstrip(".") if original.suffix else "bin"
    if ext.count(".") > 1 or len(ext) > 8:
        ext = original.suffix.lstrip(".") or "bin"
    return FileRecord(
        file_id=file_id,
        item_id=item_id,
        source_id=source_id,
        original_sha256=original_sha256,
        original_ext=ext,
        original_bytes=original.stat().st_size,
        audio_path=str(out.relative_to(data_root.root)),
        audio_sha256=sha256_file(out),
        duration_s=float(duration),
        sample_rate=sr,
        channels_original=info.channels,
        sample_rate_original=info.sample_rate,
        loudness_lufs=lufs,
        peak_dbfs=peak,
        clipping_ratio=clip,
        member_path=member_path,
    )
