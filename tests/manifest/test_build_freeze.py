from __future__ import annotations

from pathlib import Path

import pytest

from tests.index.conftest import make_chunk, make_item, make_spec, write_source
from wakewordworld.licences import LicenceTier
from wakewordworld.manifest.build import build_manifest, is_sealed, render_attribution
from wakewordworld.manifest.freeze import freeze
from wakewordworld.manifest.validate import validate_dir
from wakewordworld.transcribe.schema import Transcript
from wakewordworld.util.paths import DataRoot


def _populate(root: DataRoot, transcript: Transcript) -> list:
    spec_a = make_spec("src_en", "en")
    spec_b = make_spec("src_de", "de", spdx="CC-BY-NC-SA-4.0", background_tags=["studio"])
    write_source(
        root,
        spec_a,
        transcript=transcript,
        chunks=[
            make_chunk("f1", spec_a.id, "c1", 0.0, 30.0),
            make_chunk("f1", spec_a.id, "c2", 30.0, 30.0),
        ],
    )
    write_source(
        root,
        spec_b,
        transcript=None,
        chunks=[make_chunk("f2", spec_b.id, "c3", 0.0, 60.0)],
        file_id="f2",
        item_id="i2",
    )
    return [spec_a, spec_b]


def test_render_attribution() -> None:
    spec = make_spec("src_en", "en")
    item = make_item(spec, "i1")
    text = render_attribution(spec, item, "CC-BY-4.0")
    assert text == "Source src_en: Episode i1 by Host, CC-BY-4.0, https://example.org/src_en/i1"
    spec_nt = make_spec(
        "src_x",
        "en",
        licence={"spdx": "CC0-1.0", "evidence": spec.licence.evidence.model_dump(mode="json")},
    )
    assert render_attribution(spec_nt, make_item(spec_nt, "i1"), "CC0-1.0") == (
        "Source src_x, CC0-1.0, https://example.org/src_x/i1"
    )


def test_is_sealed_deterministic() -> None:
    ids = [f"file{i}" for i in range(2000)]
    sealed = [f for f in ids if is_sealed(f, fraction=0.1, seed="s")]
    assert 150 < len(sealed) < 250
    assert sealed == [f for f in ids if is_sealed(f, fraction=0.1, seed="s")]
    assert sealed != [f for f in ids if is_sealed(f, fraction=0.1, seed="other")]
    assert not any(is_sealed(f, fraction=0.0, seed="s") for f in ids)


def test_build_public_masks_tier_b_and_splits_sealed(
    data_root: DataRoot, sample_transcript: Transcript
) -> None:
    specs = _populate(data_root, sample_transcript)
    rows, sealed, release = build_manifest(
        data_root, specs, version="0.1.0", public=True, sealed_fraction=0.0
    )
    assert sealed == []
    assert release.n_chunks == 3
    assert release.n_files == 2
    assert release.sources == ["src_de", "src_en"]
    assert release.hours_by_language == {"de": round(60 / 3600, 4), "en": round(60 / 3600, 4)}
    assert release.hours_by_tier == {"A": round(60 / 3600, 4), "B": round(60 / 3600, 4)}
    by_id = {r.chunk_id: r for r in rows}
    assert by_id["c1"].licence_tier is LicenceTier.A
    assert by_id["c1"].audio_sha256 == "2" * 64
    assert by_id["c3"].licence_tier is LicenceTier.B
    assert by_id["c3"].audio_sha256 is None
    assert by_id["c3"].background_tags == ["studio"]
    assert by_id["c1"].engine_training_overlap == ["openwakeword"]
    assert by_id["c1"].transcript_backend == "test:1"
    assert not by_id["c1"].has_reference_transcript
    assert by_id["c3"].transcript_backend is None
    assert by_id["c1"].attribution.startswith("Source src_en: Episode i1 by Host, CC-BY-4.0")

    # With everything sealed, public rows are empty and sealed rows carry all chunks of a file.
    rows2, sealed2, release2 = build_manifest(
        data_root, specs, version="0.1.0", public=True, sealed_fraction=1.0
    )
    assert rows2 == []
    assert len(sealed2) == 3
    assert release2.n_chunks == 0
    assert all(r.sealed for r in sealed2)

    # Internal build keeps tier B checksums and sealed rows inline.
    rows3, sealed3, _ = build_manifest(
        data_root, specs, version="0.1.0", public=False, sealed_fraction=1.0
    )
    assert sealed3 == []
    assert len(rows3) == 3
    assert {r.chunk_id: r.audio_sha256 for r in rows3}["c3"] == "2" * 64
    assert all(r.sealed for r in rows3)


def test_freeze_validate_roundtrip(
    data_root: DataRoot, sample_transcript: Transcript, tmp_path: Path
) -> None:
    specs = _populate(data_root, sample_transcript)
    rows, sealed, release = build_manifest(
        data_root, specs, version="0.1.0", public=True, sealed_fraction=0.5
    )
    out = tmp_path / "manifests" / "0.1.0"
    freeze(rows, release, out, sealed_rows=sealed)
    for name in ("chunks.jsonl", "chunks.parquet", "release.json", "sources.md", "SHA256SUMS"):
        assert (out / name).exists(), name
    if sealed:
        assert (out / "sealed" / "chunks.jsonl").exists()
        assert (out / "sealed" / "README.md").exists()
    assert validate_dir(out) == []
    assert "src_en" in (out / "sources.md").read_text()

    with pytest.raises(FileExistsError):
        freeze(rows, release, out, sealed_rows=sealed)
    freeze(rows, release, out, sealed_rows=sealed, force=True)

    # Tampering is detected.
    (out / "sources.md").write_text("tampered", encoding="utf-8")
    assert any("checksum mismatch" in p for p in validate_dir(out))

    # Tier B row with checksum is detected.
    bad = out / "chunks.jsonl"
    text = bad.read_text().replace(
        '"licence_tier":"B","attribution"', '"licence_tier":"B","attribution"'
    )
    lines = []
    for line in text.splitlines():
        if '"licence_tier":"B"' in line:
            line = line.replace('"audio_sha256":null', '"audio_sha256":"' + "9" * 64 + '"')
        lines.append(line)
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
    problems = validate_dir(out)
    assert any("tier B row carries audio_sha256" in p for p in problems) or not any(
        '"licence_tier":"B"' in ln for ln in lines
    )


def test_validate_missing(tmp_path: Path) -> None:
    assert validate_dir(tmp_path / "nope")[0].endswith("missing")
