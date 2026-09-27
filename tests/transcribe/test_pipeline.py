from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from wakewordworld.cli import app
from wakewordworld.ingest.records import FileRecord, write_jsonl
from wakewordworld.sources.spec import Language, SourceSpec
from wakewordworld.transcribe.backends.base import BackendUnavailable
from wakewordworld.transcribe.pipeline import (
    find_reference,
    transcribe_file,
    transcribe_source,
    transcript_path,
)
from wakewordworld.transcribe.schema import Segment, Transcript, Word
from wakewordworld.util.paths import DataRoot

from .conftest import speechlike, write_flac

SPEC = SourceSpec.model_validate(
    {
        "id": "demo",
        "name": "Demo",
        "languages": ["de"],
        "domain": "podcast",
        "access": {"type": "rss", "url": "https://example.org/feed.xml"},
        "licence": {
            "spdx": "CC-BY-4.0",
            "evidence": {"type": "page", "quote": "CC BY 4.0", "captured_at": "2026-09-26"},
        },
    }
)

ARCHIVE_SPEC = SourceSpec.model_validate(
    {
        "id": "icsidemo",
        "name": "ICSI demo",
        "languages": ["en"],
        "domain": "meeting",
        "access": {
            "type": "http_archive",
            "urls": ["https://example.org/a.zip"],
            "transcript_glob": "**/*.mrt",
            "transcript_format": "icsi_mrt",
        },
        "licence": {
            "spdx": "CC-BY-4.0",
            "evidence": {"type": "page", "quote": "CC BY 4.0", "captured_at": "2026-09-26"},
        },
    }
)


class FakeBackend:
    name = "fake:1"

    def __init__(self) -> None:
        self.calls: list[tuple[Path, Language | None]] = []

    def transcribe(self, audio_path: Path, language: Language | None) -> Transcript:
        self.calls.append((audio_path, language))
        return Transcript(
            file_id="",
            source_id="",
            language=Language.EN,
            backend=self.name,
            origin="asr",
            duration_s=2.0,
            segments=[
                Segment(
                    start=0.0,
                    end=2.0,
                    text="hallo Michael",
                    words=[
                        Word(text="hallo", start=0.1, end=0.5, confidence=0.9),
                        Word(text="Michael", start=0.6, end=1.2, confidence=0.8),
                    ],
                )
            ],
        )


def _make_root(tmp_path: Path, source_id: str, n: int = 2) -> tuple[DataRoot, list[FileRecord]]:
    root = DataRoot(tmp_path / "data")
    root.ensure()
    recs: list[FileRecord] = []
    for i in range(n):
        rel = Path("audio") / source_id / f"f{i}.flac"
        write_flac(root.root / rel, speechlike(2.0, seed=i))
        recs.append(
            FileRecord(
                file_id=f"f{i}",
                item_id=f"i{i}",
                source_id=source_id,
                original_sha256="0" * 64,
                original_ext="mp3",
                original_bytes=1,
                audio_path=str(rel),
                audio_sha256="1" * 64,
                duration_s=2.0,
            )
        )
    write_jsonl(root.audio / source_id / "files.jsonl", recs, key="file_id")
    return root, recs


def test_asr_path_fills_ids_and_language(tmp_path: Path) -> None:
    root, recs = _make_root(tmp_path, "demo")
    backend = FakeBackend()
    t = transcribe_file(root, SPEC, recs[0], backend=backend)
    assert t.file_id == "f0"
    assert t.source_id == "demo"
    assert t.language == Language.DE  # spec language wins over backend guess
    assert t.origin == "asr"
    assert [w.text for w in t.words()] == ["hallo", "Michael"]
    assert backend.calls[0][1] == Language.DE


def test_reference_uniform_path_and_skip_and_force(tmp_path: Path) -> None:
    root, recs = _make_root(tmp_path, "demo")
    ref = root.cache / "reftext" / "demo" / "f1.txt"
    ref.parent.mkdir(parents=True)
    ref.write_text("Guten Morgen Michael", encoding="utf-8")
    hit = find_reference(root, SPEC, recs[1])
    assert hit is not None
    assert (hit.fmt, hit.how) == ("txt", "reftext")
    assert find_reference(root, SPEC, recs[0]) is None

    backend = FakeBackend()
    summary = transcribe_source(root, SPEC, backend=backend, align_backend="uniform")
    assert (summary.done, summary.skipped, summary.failed) == (2, 0, 0)
    assert summary.origins == {"asr": 1, "reference": 1}
    assert summary.hours == pytest.approx(4 / 3600)
    t1 = Transcript.load(transcript_path(root, "demo", "f1"))
    assert (t1.origin, t1.backend) == ("reference", "uniform")
    assert [w.text for w in t1.words()] == ["Guten", "Morgen", "Michael"]
    assert t1.words()[-1].end == pytest.approx(2.0)

    again = transcribe_source(root, SPEC, backend=backend, align_backend="uniform")
    assert (again.done, again.skipped) == (0, 2)
    forced = transcribe_source(
        root, SPEC, backend=backend, align_backend="uniform", force=True, limit=1
    )
    assert forced.done == 1


def test_reference_from_archive_by_stem(tmp_path: Path, fixtures_dir: Path) -> None:
    root, recs = _make_root(tmp_path, "icsidemo", n=1)
    arch = root.cache / "archives" / "icsidemo" / "Bmr001"
    arch.mkdir(parents=True)
    (arch / "f0.mrt").write_text((fixtures_dir / "sample.mrt").read_text(), encoding="utf-8")
    hit = find_reference(root, ARCHIVE_SPEC, recs[0])
    assert hit is not None
    assert (hit.fmt, hit.how) == ("icsi_mrt", "archive")
    t = transcribe_file(root, ARCHIVE_SPEC, recs[0], backend=None, align_backend="none")
    assert (t.origin, t.backend) == ("reference", "none")
    assert t.segments[0].speaker == "me011"
    assert t.words() == []


def test_failures_are_recorded_not_raised(tmp_path: Path) -> None:
    root, _ = _make_root(tmp_path, "demo", n=1)
    summary = transcribe_source(root, SPEC, backend=None, align_backend="none")
    assert (summary.done, summary.failed) == (0, 1)
    failures = root.transcripts / "demo" / "failures.jsonl"
    rows = [json.loads(line) for line in failures.read_text().splitlines()]
    assert rows[0]["stage"] == "asr"
    assert "no ASR backend" in rows[0]["error"]
    other_root, other_recs = _make_root(tmp_path / "b", "demo", 1)
    with pytest.raises(BackendUnavailable):
        transcribe_file(other_root, SPEC, other_recs[0], backend=None)


def test_duplicates_are_skipped(tmp_path: Path) -> None:
    root, recs = _make_root(tmp_path, "demo", n=2)
    dup = recs[1].model_copy(update={"duplicate_of": "f0"})
    write_jsonl(root.audio / "demo" / "files.jsonl", [dup], key="file_id")
    summary = transcribe_source(root, SPEC, backend=FakeBackend())
    assert summary.done == 1


def test_cli_status_show_and_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root, _ = _make_root(tmp_path, "demo", n=1)
    transcribe_source(root, SPEC, backend=FakeBackend())
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    import yaml

    (sources_dir / "demo.yaml").write_text(
        yaml.safe_dump(json.loads(SPEC.model_dump_json(exclude_none=True))), encoding="utf-8"
    )
    runner = CliRunner()
    res = runner.invoke(
        app,
        ["transcribe", "status", "--data-root", str(root.root), "--sources-dir", str(sources_dir)],
    )
    assert res.exit_code == 0, res.output
    assert "demo" in res.output
    res = runner.invoke(
        app, ["transcribe", "show", "f0", "--source", "demo", "--data-root", str(root.root)]
    )
    assert res.exit_code == 0, res.output
    assert "Michael" in res.output
    res = runner.invoke(app, ["transcribe", "check-backends"])
    assert res.exit_code == 0, res.output
    res = runner.invoke(
        app,
        [
            "transcribe",
            "run",
            "demo",
            "--backend",
            "none",
            "--align",
            "bogus",
            "--data-root",
            str(root.root),
            "--sources-dir",
            str(sources_dir),
        ],
    )
    assert res.exit_code == 2
