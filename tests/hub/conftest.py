from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from wakewordworld.licences import LicenceTier
from wakewordworld.manifest.freeze import freeze
from wakewordworld.manifest.schema import ChunkRow, ManifestRelease
from wakewordworld.util.paths import DataRoot


def _row(
    chunk_id: str, source_id: str, lang: str, spdx: str, tier: LicenceTier, **kw: object
) -> ChunkRow:
    return ChunkRow(
        chunk_id=chunk_id,
        file_id=f"file-{chunk_id}",
        item_id=f"item-{chunk_id}",
        source_id=source_id,
        language=lang,  # type: ignore[arg-type]
        licence_spdx=spdx,
        licence_tier=tier,
        attribution=f"{source_id}, {spdx}, https://example.org/{chunk_id}",
        item_url=f"https://example.org/{chunk_id}.mp3",
        page_url=f"https://example.org/{chunk_id}",
        start_s=0.0,
        duration_s=600.0,
        domain="podcast",  # type: ignore[arg-type]
        microphone="close",  # type: ignore[arg-type]
        audio_sha256=None if tier is LicenceTier.B else "0" * 64,
        **kw,  # type: ignore[arg-type]
    )


@pytest.fixture
def rows() -> list[ChunkRow]:
    return [
        _row("c-cc0", "voxpopuli", "de", "CC0-1.0", LicenceTier.A),
        _row("c-ccby", "kuechenradio", "de", "CC-BY-3.0-DE", LicenceTier.A),
        _row("c-ccbysa", "hackerpublicradio", "en", "CC-BY-SA-4.0", LicenceTier.A),
        _row("c-nc", "freakshow", "de", "CC-BY-NC-ND-3.0-DE", LicenceTier.B),
    ]


@pytest.fixture
def manifest_dir(tmp_path: Path, rows: list[ChunkRow]) -> Path:
    release = ManifestRelease(
        version="0.1.0",
        created_at=datetime.now(UTC),
        n_chunks=len(rows),
        n_files=len(rows),
        hours_by_language={"de": 0.5, "en": 0.17},
        hours_by_tier={"A": 0.5, "B": 0.17},
        sources=sorted({r.source_id for r in rows}),
        notes="test release",
    )
    return freeze(rows, release, tmp_path / "manifests" / "0.1.0")


@pytest.fixture
def data_root(tmp_path: Path, rows: list[ChunkRow]) -> DataRoot:
    root = DataRoot(tmp_path / "data")
    root.ensure()
    for r in rows:
        d = root.chunks / r.source_id
        d.mkdir(parents=True, exist_ok=True)
        x = (np.sin(np.linspace(0, 200, 16000)) * 8000).astype(np.int16)
        sf.write(d / f"{r.chunk_id}.flac", x, 16000, subtype="PCM_16")
    import polars as pl

    pl.DataFrame(
        {
            "chunk_id": ["c-ccby", "c-ccby", "c-other"],
            "word": ["hallo", "michael", "x"],
            "start_s": [0.1, 0.5, 0.0],
            "end_s": [0.3, 0.9, 0.1],
        }
    ).write_parquet(root.index / "kuechenradio.parquet")
    (root.transcripts / "kuechenradio").mkdir(parents=True)
    (root.transcripts / "kuechenradio" / "file-c-ccby.json").write_text(
        '{"file_id": "file-c-ccby"}'
    )
    (root.transcripts / "freakshow").mkdir(parents=True)
    (root.transcripts / "freakshow" / "file-c-nc.json").write_text('{"file_id": "file-c-nc"}')
    return root
