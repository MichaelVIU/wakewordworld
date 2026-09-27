from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf

from wakewordworld.ingest.normalize import clipping_ratio, normalize_file, probe
from wakewordworld.util.hashing import sha256_file
from wakewordworld.util.paths import DataRoot

from .conftest import has_tool, tone, write_flac, write_wav

pytestmark = pytest.mark.skipif(not has_tool("ffmpeg"), reason="ffmpeg not installed")


def test_probe_and_normalize_stereo_44k(data_root: DataRoot, tmp_path: Path) -> None:
    src = write_wav(tmp_path / "in.wav", tone(3.0, sr=44_100), sr=44_100, channels=2)
    info = probe(src)
    assert info.has_audio
    assert info.sample_rate == 44_100
    assert info.channels == 2
    assert abs(info.duration_s - 3.0) < 0.05

    rec = normalize_file(
        src,
        item_id="it",
        source_id="s",
        original_sha256=sha256_file(src),
        data_root=data_root,
    )
    out = data_root.root / rec.audio_path
    assert out.exists()
    with sf.SoundFile(str(out)) as f:
        assert f.samplerate == 16_000
        assert f.channels == 1
        assert f.subtype == "PCM_16"
    assert abs(rec.duration_s - 3.0) < 0.05
    assert rec.channels_original == 2
    assert rec.sample_rate_original == 44_100
    assert rec.loudness_lufs is not None
    assert rec.loudness_lufs < 0
    assert rec.peak_dbfs is not None
    assert rec.clipping_ratio == 0.0
    assert rec.audio_sha256 == sha256_file(out)
    assert rec.original_ext == "wav"
    # idempotent: same file id and checksum
    rec2 = normalize_file(
        src, item_id="it", source_id="s", original_sha256=sha256_file(src), data_root=data_root
    )
    assert rec2.file_id == rec.file_id
    assert rec2.audio_sha256 == rec.audio_sha256


def test_clipping_ratio(tmp_path: Path) -> None:
    clean = write_flac(tmp_path / "clean.flac", tone(1.0, amp=0.5))
    assert clipping_ratio(clean) == 0.0
    hot = write_flac(tmp_path / "hot.flac", tone(1.0, amp=1.0))
    assert clipping_ratio(hot) > 0.0
