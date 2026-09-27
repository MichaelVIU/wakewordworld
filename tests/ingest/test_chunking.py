from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from wakewordworld.ingest.chunking import (
    ChunkPolicy,
    Silence,
    cut_points,
    detect_silences,
    frame_levels,
    write_chunks,
)
from wakewordworld.util.paths import DataRoot

from .conftest import SR, silence, tone, write_flac


def test_detect_silences_from_levels() -> None:
    policy = ChunkPolicy(frame_s=0.02, min_silence_s=0.4)
    levels = np.full(1000, -20.0)
    levels[100:130] = -80.0  # 0.6 s -> silence
    levels[500:510] = -80.0  # 0.2 s -> too short
    sil = detect_silences(levels, policy)
    assert len(sil) == 1
    assert sil[0].start_s == 2.0
    assert sil[0].end_s == 2.6


def test_cut_points_short_file_single_chunk() -> None:
    policy = ChunkPolicy()
    cuts = cut_points(19 * 60, [Silence(300, 301)], policy)
    assert [c.at_s for c in cuts] == [0.0]


def test_cut_points_prefers_longest_silence_in_window() -> None:
    policy = ChunkPolicy()
    silences = [
        Silence(11 * 60, 11 * 60 + 1),
        Silence(17 * 60, 17 * 60 + 3),
        Silence(25 * 60, 25 * 60 + 2),
    ]
    cuts = cut_points(45 * 60, silences, policy)
    ats = [c.at_s for c in cuts]
    assert ats[0] == 0.0
    assert ats[1] == 17 * 60 + 1.5  # longest silence in [10, 20] min
    assert cuts[1].reason == "silence"
    # second window [27, 37] min has no silence -> max_length cut at 37 min
    assert ats[2] == 17 * 60 + 1.5 + 20 * 60
    assert cuts[2].reason == "max_length"
    assert len(cuts) == 3  # remaining 45 - 37.03 = ~8 min < 20 -> file end


def test_cut_points_avoids_short_tail() -> None:
    policy = ChunkPolicy()
    # 20.5 min file with a silence at 20 min: cutting there would leave a 30 s tail
    cuts = cut_points(20.5 * 60, [Silence(20 * 60 - 0.5, 20 * 60 + 0.5)], policy)
    assert [c.at_s for c in cuts] == [0.0]


def test_frame_levels_and_write_chunks(data_root: DataRoot, tmp_path: Path) -> None:
    # 45 minutes: tone, with 2 s silences at 12:30 and 30:30 (and 1 s at 16 min).
    minute = tone(60.0, amp=0.2)
    parts: list[np.ndarray] = []
    for m in range(45):
        if m == 12:
            parts.append(np.concatenate([tone(29.0, amp=0.2), silence(2.0), tone(29.0, amp=0.2)]))
        elif m == 16:
            parts.append(np.concatenate([tone(29.5, amp=0.2), silence(1.0), tone(29.5, amp=0.2)]))
        elif m == 30:
            parts.append(np.concatenate([tone(29.0, amp=0.2), silence(2.0), tone(29.0, amp=0.2)]))
        else:
            parts.append(minute)
    samples = np.concatenate(parts)
    path = write_flac(tmp_path / "long.flac", samples)

    levels = frame_levels(path, 0.02)
    assert abs(levels.shape[0] - 45 * 60 * 50) <= 1

    records = write_chunks(path, file_id="f1", source_id="s", data_root=data_root)
    starts = [round(r.start_s) for r in records]
    assert starts == [0, 12 * 60 + 30, 30 * 60 + 30]
    assert [r.cut_reason for r in records] == ["silence", "silence", "file_end"]
    total = sum(r.duration_s for r in records)
    assert abs(total - 45 * 60) < 0.01
    for r in records:
        p = data_root.root / r.audio_path
        assert p.exists()
        with sf.SoundFile(str(p)) as f:
            assert f.samplerate == SR
            assert f.channels == 1
            assert abs(f.frames / SR - r.duration_s) < 1e-6
    # idempotent
    again = write_chunks(path, file_id="f1", source_id="s", data_root=data_root)
    assert [r.chunk_id for r in again] == [r.chunk_id for r in records]


def test_write_chunks_short_file(data_root: DataRoot, tmp_path: Path) -> None:
    path = write_flac(tmp_path / "short.flac", tone(5.0))
    records = write_chunks(path, file_id="f2", source_id="s", data_root=data_root)
    assert len(records) == 1
    assert records[0].start_s == 0.0
    assert records[0].cut_reason == "file_end"
    assert abs(records[0].duration_s - 5.0) < 1e-6
