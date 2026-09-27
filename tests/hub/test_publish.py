from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from wakewordworld.hub.publish import build_release_files
from wakewordworld.manifest.schema import ChunkRow
from wakewordworld.util.paths import DataRoot


def test_build_release_files(manifest_dir: Path, data_root: DataRoot, tmp_path: Path) -> None:
    out = tmp_path / "dist"
    build = build_release_files(manifest_dir, data_root, out, repo_id="org/bench")
    assert build.n_rows == 4
    assert build.n_audio == 3
    assert build.families == {"cc0": 1, "cc-by": 1, "cc-by-sa": 1}
    assert build.missing_audio == []

    meta = pl.read_parquet(out / "metadata.parquet")
    assert meta.height == 4
    nc = meta.filter(pl.col("chunk_id") == "c-nc").row(0, named=True)
    assert nc["licence_tier"] == "B"
    assert nc["audio_sha256"] is None

    for fam in ("cc0", "cc-by", "cc-by-sa"):
        shards = sorted((out / "data" / fam).glob("*.parquet"))
        assert len(shards) == 1
        t = pq.read_table(shards[0])
        assert t.schema.field("audio").type.num_fields == 2
        row = t.to_pylist()[0]
        assert row["audio"]["path"].endswith(".flac")
        assert row["audio"]["bytes"][:4] == b"fLaC"
        assert row["licence_tier"] == "A"
    # No tier B audio anywhere.
    all_ids = [
        r["chunk_id"]
        for fam in ("cc0", "cc-by", "cc-by-sa")
        for r in pq.read_table(next((out / "data" / fam).glob("*.parquet"))).to_pylist()
    ]
    assert "c-nc" not in all_ids

    idx = pl.read_parquet(out / "index" / "kuechenradio.parquet")
    assert idx.get_column("chunk_id").unique().to_list() == ["c-ccby"]
    assert (out / "transcripts" / "kuechenradio.jsonl").exists()
    assert not (out / "transcripts" / "freakshow.jsonl").exists()

    assert (out / "README.md").read_text(encoding="utf-8").startswith("---")
    assert (out / "SHA256SUMS").exists()
    b = json.loads((out / "build.json").read_text())
    assert b["n_audio"] == 3


def test_build_release_files_no_audio(
    manifest_dir: Path, data_root: DataRoot, tmp_path: Path
) -> None:
    out = tmp_path / "dist"
    build = build_release_files(manifest_dir, data_root, out, include_audio=False)
    assert build.n_audio == 0
    assert not (out / "data").exists()
    assert (out / "metadata.parquet").exists()


def test_refuses_sealed_rows(manifest_dir: Path, data_root: DataRoot, tmp_path: Path) -> None:
    p = manifest_dir / "chunks.jsonl"
    rows = [
        ChunkRow.model_validate_json(line) for line in p.read_text().splitlines() if line.strip()
    ]
    rows[0] = rows[0].model_copy(update={"sealed": True})
    p.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    with pytest.raises(ValueError, match="sealed"):
        build_release_files(manifest_dir, data_root, tmp_path / "dist")
