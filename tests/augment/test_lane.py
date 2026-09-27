from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import soundfile as sf
from typer.testing import CliRunner

from wakewordworld.augment.cli import augment_app
from wakewordworld.augment.lane import (
    DEFAULT_CONDITIONS,
    Condition,
    build_lane,
    data_root_for_condition,
    parse_condition,
)
from wakewordworld.licences import LicenceTier
from wakewordworld.manifest.freeze import freeze
from wakewordworld.manifest.schema import ChunkRow, ManifestRelease
from wakewordworld.util.paths import DataRoot

from .conftest import SR, FakeBank, speech_like


def _row(chunk_id: str, source: str, tier: LicenceTier, sha: str | None) -> ChunkRow:
    return ChunkRow(
        chunk_id=chunk_id,
        file_id=f"file-{chunk_id}",
        item_id=f"item-{chunk_id}",
        source_id=source,
        language="de",
        licence_spdx="CC-BY-4.0" if tier is LicenceTier.A else "CC-BY-NC-4.0",
        licence_tier=tier,
        attribution=f"{source} attribution",
        item_url="https://example.org/x",
        start_s=0.0,
        duration_s=4.0,
        domain="podcast",
        microphone="close",
        audio_sha256=sha,
    )


@pytest.fixture
def base(tmp_path: Path) -> tuple[DataRoot, Path]:
    root = DataRoot(tmp_path / "data")
    root.ensure()
    rows = [_row("c1", "src_a", LicenceTier.A, "abc"), _row("c2", "src_b", LicenceTier.B, None)]
    for r in rows:
        p = root.chunks / r.source_id / f"{r.chunk_id}.flac"
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(p, (speech_like(4.0) * 32767).astype(np.int16), SR, subtype="PCM_16")
    pl.DataFrame(
        {
            "chunk_id": ["c1", "c1"],
            "word": ["hallo", "michael"],
            "start_s": [1.0, 2.0],
            "end_s": [1.4, 2.5],
        }
    ).write_parquet(root.index / "src_a.parquet")
    release = ManifestRelease(
        version="0.1.0",
        created_at=datetime.now(UTC),
        n_chunks=2,
        n_files=2,
        hours_by_language={"de": 8 / 3600},
        hours_by_tier={"A": 4 / 3600, "B": 4 / 3600},
        sources=["src_a", "src_b"],
    )
    mdir = tmp_path / "manifests" / "0.1.0"
    freeze(rows, release, mdir)
    return root, mdir


def test_parse_condition() -> None:
    c = parse_condition("snr10:kitchen,babble")
    assert c == Condition(
        "snr10-kitchen+babble", snr_db=10.0, noise_categories=("kitchen", "babble")
    )
    assert parse_condition("snr5").noise_categories
    assert parse_condition("rir:rirs_noises").rir_set == "rirs_noises"
    assert parse_condition("speed:1.1").speed == 1.1
    assert parse_condition("clean").is_identity
    combo = parse_condition("rir:rirs_noises+snr10:kitchen")
    assert combo.rir_set == "rirs_noises"
    assert combo.snr_db == 10.0
    with pytest.raises(ValueError, match="unknown condition"):
        parse_condition("loud")
    with pytest.raises(ValueError, match="speed"):
        Condition("x", speed=2.0)


def test_default_conditions_cover_plan() -> None:
    names = {c.name for c in DEFAULT_CONDITIONS}
    assert {"clean", "snr20", "snr10", "snr5", "snr0", "speed-0.9", "speed-1.1"} <= names
    assert any(c.rir_set for c in DEFAULT_CONDITIONS)


def test_data_root_for_condition_layout(base: tuple[DataRoot, Path]) -> None:
    root, _ = base
    aug = data_root_for_condition(root, Condition("snr10", snr_db=10.0))
    assert aug.chunks.resolve() == (root.root / "chunks_aug" / "snr10").resolve()
    assert aug.index.resolve() == root.index.resolve()
    assert aug.audio.resolve() == root.audio.resolve()
    fast = data_root_for_condition(root, Condition("speed-1.1", speed=1.1))
    assert fast.index.is_dir()
    assert not fast.index.is_symlink()
    # idempotent
    assert data_root_for_condition(root, "snr10").root == aug.root


def test_build_lane_snr_and_speed(
    base: tuple[DataRoot, Path], fake_bank: FakeBank, fake_rir_bank: FakeBank
) -> None:
    root, mdir = base
    conds = [
        Condition("clean"),
        Condition("snr10-kitchen", snr_db=10.0, noise_categories=("kitchen",)),
        Condition("rir-fake", rir_set="fake"),
        Condition("speed-1.1", speed=1.1),
    ]
    out = mdir.parent
    results = build_lane(
        root,
        mdir,
        conditions=conds,
        out_manifests_dir=out,
        noise_bank=fake_bank,
        rir_bank=fake_rir_bank,
        seed="t",
    )
    assert [r.n_chunks for r in results] == [2, 2, 2, 2]
    by_name = {r.condition.name: r for r in results}

    snr = by_name["snr10-kitchen"]
    assert snr.manifest_dir == out / "0.1.0-aug-snr10-kitchen"
    assert (root.root / "chunks_aug" / "snr10-kitchen" / "src_a" / "c1.flac").exists()
    rows = [
        ChunkRow.model_validate_json(line)
        for line in (snr.manifest_dir / "chunks.jsonl").read_text().splitlines()
    ]
    assert all("aug:snr10-kitchen" in r.background_tags for r in rows)
    assert all("augmented with: Fake noise, CC0" in r.attribution for r in rows)
    tier_a = next(r for r in rows if r.chunk_id == "c1")
    tier_b = next(r for r in rows if r.chunk_id == "c2")
    assert tier_a.audio_sha256 not in (None, "abc")
    assert tier_b.audio_sha256 is None
    release = json.loads((snr.manifest_dir / "release.json").read_text())
    notes = json.loads(release["notes"])
    assert notes["condition"]["snr_db"] == 10.0
    assert notes["seed"] == "t"
    prov = [
        json.loads(line)
        for line in (snr.data_root.root / "provenance.jsonl").read_text().splitlines()
    ]
    assert all(p["noise_set"] == "fake" and abs(p["achieved_snr_db"] - 10.0) < 0.3 for p in prov)
    assert all("kitchen" in p["noise_file"] for p in prov)
    # audio differs from the clean copy but has the same length
    a = sf.read(root.chunks / "src_a" / "c1.flac", dtype="float32")[0]
    b = sf.read(snr.data_root.chunks / "src_a" / "c1.flac", dtype="float32")[0]
    assert a.shape == b.shape
    assert not np.allclose(a, b)

    clean = by_name["clean"]
    c = sf.read(clean.data_root.chunks / "src_a" / "c1.flac", dtype="float32")[0]
    assert np.allclose(a, c, atol=1e-4)

    fast = by_name["speed-1.1"]
    f = sf.read(fast.data_root.chunks / "src_a" / "c1.flac", dtype="float32")[0]
    assert abs(f.shape[0] - round(a.shape[0] / 1.1)) <= 1
    idx = pl.read_parquet(fast.data_root.index / "src_a.parquet")
    assert idx.get_column("start_s").to_list() == pytest.approx([1.0 / 1.1, 2.0 / 1.1])
    fast_rows = [
        ChunkRow.model_validate_json(line)
        for line in (fast.manifest_dir / "chunks.jsonl").read_text().splitlines()
    ]
    assert all(abs(r.duration_s - 4.0 / 1.1) < 0.01 for r in fast_rows)

    rir = by_name["rir-fake"]
    r_audio = sf.read(rir.data_root.chunks / "src_a" / "c1.flac", dtype="float32")[0]
    assert r_audio.shape == a.shape
    assert not np.allclose(a, r_audio)

    # rerun reuses existing chunks
    again = build_lane(
        root, mdir, conditions=conds[:2], out_manifests_dir=out, noise_bank=fake_bank, seed="t"
    )
    assert all(p.get("reused") for p in again[1].provenance)


def test_build_lane_requires_banks(base: tuple[DataRoot, Path]) -> None:
    root, mdir = base
    with pytest.raises(ValueError, match="needs a noise bank"):
        build_lane(
            root, mdir, conditions=[Condition("snr5", snr_db=5.0)], out_manifests_dir=mdir.parent
        )
    with pytest.raises(ValueError, match="needs an RIR bank"):
        build_lane(
            root, mdir, conditions=[Condition("rir-x", rir_set="x")], out_manifests_dir=mdir.parent
        )


def test_cli_smoke(base: tuple[DataRoot, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    root, mdir = base
    import wakewordworld.augment.noise as noise_mod
    import wakewordworld.augment.rir as rir_mod

    monkeypatch.setattr(noise_mod, "NoiseBank", lambda *a, **k: FakeBank())
    monkeypatch.setattr(rir_mod, "RirBank", lambda *a, **k: FakeBank(kind="rir"))
    runner = CliRunner()
    res = runner.invoke(
        augment_app,
        [
            "lane",
            "build",
            "--manifest",
            str(mdir),
            "--condition",
            "snr5:babble",
            "--condition",
            "rir:fake",
            "--out-manifests",
            str(mdir.parent),
            "--data-root",
            str(root.root),
            "--limit",
            "1",
        ],
    )
    assert res.exit_code == 0, res.output
    assert "snr5-babble" in res.output
    assert "eval run <engine> --manifest" in res.output
    res = runner.invoke(augment_app, ["lane", "list", "--data-root", str(root.root)])
    assert res.exit_code == 0, res.output
    assert "snr5-babble" in res.output
    res = runner.invoke(augment_app, ["noise", "list", "--data-root", str(root.root)])
    assert res.exit_code == 0, res.output
    assert "demand" in res.output
