from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from wakewordworld.ingest.records import ChunkRecord, FileRecord, write_jsonl
from wakewordworld.licences import LicenceTier
from wakewordworld.manifest.schema import FetchedItem, ItemLicence
from wakewordworld.sources.spec import Language, SourceSpec
from wakewordworld.transcribe.schema import Segment, Transcript, Word
from wakewordworld.util.paths import DataRoot


def make_spec(
    source_id: str, language: str, *, spdx: str = "CC-BY-4.0", **kw: object
) -> SourceSpec:
    base: dict[str, object] = {
        "id": source_id,
        "name": f"Source {source_id}",
        "languages": [language],
        "domain": "podcast",
        "microphone": "close",
        "access": {"type": "rss", "url": f"https://example.org/{source_id}.xml"},
        "licence": {
            "spdx": spdx,
            "evidence": {
                "type": "page",
                "url": "https://example.org/about",
                "quote": "All episodes are licensed under " + spdx,
                "captured_at": "2026-09-26",
            },
            "attribution": "{name}: {title} by {author}, {licence}, {url}",
        },
        "background_tags": ["home"],
        "engine_training_overlap": ["openwakeword"],
    }
    base.update(kw)
    return SourceSpec.model_validate(base)


def make_item(spec: SourceSpec, item_id: str, *, tier: LicenceTier | None = None) -> FetchedItem:
    return FetchedItem(
        item_id=item_id,
        source_id=spec.id,
        url=f"https://example.org/{spec.id}/{item_id}.mp3",
        page_url=f"https://example.org/{spec.id}/{item_id}",
        title=f"Episode {item_id}",
        author="Host",
        published=datetime(2026, 1, 1, tzinfo=UTC),
        duration_s=1200.0,
        language=spec.languages[0],
        licence=ItemLicence(
            spdx=spec.licence.spdx, tier=tier or spec.tier, origin="source", evidence_quote="x"
        ),
    )


def make_file(item: FetchedItem, file_id: str, duration: float) -> FileRecord:
    return FileRecord(
        file_id=file_id,
        item_id=item.item_id,
        source_id=item.source_id,
        original_sha256="0" * 64,
        original_ext="mp3",
        original_bytes=1,
        audio_path=f"audio/{item.source_id}/{file_id}.flac",
        audio_sha256="1" * 64,
        duration_s=duration,
    )


def make_chunk(
    file_id: str, source_id: str, chunk_id: str, start: float, dur: float
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        file_id=file_id,
        source_id=source_id,
        start_s=start,
        duration_s=dur,
        audio_path=f"chunks/{source_id}/{chunk_id}.flac",
        audio_sha256="2" * 64,
        cut_reason="silence",
    )


def words(spec: list[tuple[str, float, float]], speaker: str | None = None) -> list[Word]:
    return [Word(text=t, start=s, end=e, confidence=0.9, speaker=speaker) for t, s, e in spec]


def write_source(
    root: DataRoot,
    spec: SourceSpec,
    *,
    transcript: Transcript | None,
    chunks: list[ChunkRecord],
    file_id: str = "f1",
    item_id: str = "i1",
    tier: LicenceTier | None = None,
    duration: float = 60.0,
) -> None:
    item = make_item(spec, item_id, tier=tier)
    write_jsonl(root.items / f"{spec.id}.jsonl", [item], key="item_id")
    write_jsonl(
        root.audio / spec.id / "files.jsonl", [make_file(item, file_id, duration)], key="file_id"
    )
    write_jsonl(root.chunks / spec.id / "chunks.jsonl", chunks, key="chunk_id")
    if transcript is not None:
        transcript.save(root.transcripts / spec.id / f"{file_id}.json")


@pytest.fixture
def data_root(tmp_path: Path) -> DataRoot:
    root = DataRoot(tmp_path / "data")
    root.ensure()
    return root


@pytest.fixture
def sample_transcript() -> Transcript:
    """A 60 s file: greeting segment, music, and a second utterance crossing 30 s."""
    return Transcript(
        file_id="f1",
        source_id="src_en",
        language=Language.EN,
        backend="test:1",
        origin="asr",
        duration_s=60.0,
        segments=[
            Segment(
                start=0.0,
                end=5.0,
                text="Jane, can you hear me?",
                speaker="A",
                words=words(
                    [
                        ("Jane,", 0.5, 0.9),
                        ("can", 1.5, 1.7),
                        ("you", 1.7, 1.9),
                        ("hear", 1.9, 2.2),
                        ("me?", 2.2, 2.6),
                    ],
                    "A",
                ),
            ),
            Segment(
                start=10.0,
                end=20.0,
                text="[music]",
                is_music=True,
                words=words([("Michael", 12.0, 12.5)]),
            ),
            Segment(
                start=28.0,
                end=33.0,
                text="Yes Michael I do",
                speaker="B",
                words=words(
                    [
                        ("Yes", 28.5, 28.8),
                        ("Michael", 29.8, 30.4),
                        ("I", 31.0, 31.1),
                        ("do", 31.1, 31.4),
                    ],
                    "B",
                ),
            ),
        ],
    )
