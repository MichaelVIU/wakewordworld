from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from wakewordworld.ingest.archive import expand_archive, is_archive
from wakewordworld.ingest.parquet_audio import expand_parquet
from wakewordworld.util.paths import DataRoot

from .conftest import tone, write_wav


def test_expand_zip(data_root: DataRoot, tmp_path: Path) -> None:
    a = write_wav(tmp_path / "a.wav", tone(0.5))
    b = write_wav(tmp_path / "sub" / "b.wav", tone(0.5, freq=880))
    (tmp_path / "notes.txt").write_text("x")
    z = tmp_path / "corpus.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(a, "a.wav")
        zf.write(b, "sub/b.wav")
        zf.write(tmp_path / "notes.txt", "notes.txt")
        zf.writestr("../evil.wav", b"RIFF")
    assert is_archive(z)
    members = list(expand_archive(z, item_id="it", data_root=data_root, audio_glob="**/*.wav"))
    assert [m.member_path for m in members] == ["a.wav", "sub/b.wav"]
    assert all(m.path.exists() for m in members)
    assert len({m.sha256 for m in members}) == 2
    # idempotent
    again = list(expand_archive(z, item_id="it", data_root=data_root, audio_glob="**/*.wav"))
    assert [m.sha256 for m in again] == [m.sha256 for m in members]


def test_expand_tar_gz(data_root: DataRoot, tmp_path: Path) -> None:
    a = write_wav(tmp_path / "a.wav", tone(0.2))
    t = tmp_path / "c.tar.gz"
    with tarfile.open(t, "w:gz") as tf:
        tf.add(a, "deep/a.wav")
    members = list(expand_archive(t, item_id="it2", data_root=data_root))
    assert [m.member_path for m in members] == ["deep/a.wav"]


def test_expand_parquet_hf_layout(data_root: DataRoot) -> None:
    def wav_bytes(freq: float) -> bytes:
        buf = io.BytesIO()
        sf.write(buf, tone(0.3, freq=freq), 16_000, subtype="PCM_16", format="WAV")
        return buf.getvalue()

    audio = pa.array(
        [
            {"bytes": wav_bytes(440), "path": "clip1.wav"},
            {"bytes": wav_bytes(660), "path": None},
            {"bytes": None, "path": "missing.wav"},
        ],
        type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
    )
    table = pa.table({"audio": audio, "sentence": ["hallo michael", "servus", "leer"]})
    p = data_root.cache / "x.parquet"
    pq.write_table(table, p)
    rows = list(
        expand_parquet(
            p,
            item_id="pq",
            source_id="cv",
            data_root=data_root,
            audio_column="audio",
            text_column="sentence",
            max_rows=None,
        )
    )
    assert [r.row_index for r in rows] == [0, 1]
    assert rows[0].path.suffix == ".wav"
    assert rows[1].path.suffix == ".wav"
    assert rows[0].text == "hallo michael"
    ref = data_root.cache / "reftext" / "cv" / f"{rows[0].sha256}.txt"
    assert ref.read_text() == "hallo michael"
    data, sr = sf.read(str(rows[1].path))
    assert sr == 16_000
    assert len(data) == int(0.3 * 16_000)
    capped = list(expand_parquet(p, item_id="pq", source_id="cv", data_root=data_root, max_rows=1))
    assert len(capped) == 1
    assert np.isfinite(capped[0].row_index)
