from __future__ import annotations

from datetime import UTC, datetime

import pytest

from wakewordworld.licences import LicenceTier
from wakewordworld.manifest.schema import (
    CANARY,
    ChunkRow,
    ItemLicence,
    ManifestRelease,
    WordOccurrence,
)


def test_chunk_row_roundtrip() -> None:
    row = ChunkRow(
        chunk_id="c1",
        file_id="f1",
        item_id="i1",
        source_id="kuechenradio",
        language="de",
        licence_spdx="cc-by-3.0-de",
        licence_tier=LicenceTier.A,
        attribution="Küchenradio, CC BY 3.0 DE",
        item_url="https://example.org/ep1.mp3",
        start_s=0.0,
        duration_s=600.0,
        domain="podcast",
        microphone="close",
    )
    dumped = row.model_dump_json()
    again = ChunkRow.model_validate_json(dumped)
    assert again == row
    assert again.canary == CANARY


def test_item_licence_normalises() -> None:
    lic = ItemLicence(spdx="CC BY-SA 4.0", tier=LicenceTier.A, origin="item")
    assert lic.spdx == "CC-BY-SA-4.0"


def test_word_occurrence_bounds() -> None:
    with pytest.raises(ValueError, match=r"confidence|end_s"):
        WordOccurrence(
            chunk_id="c", word="a", raw="a", start_s=1.0, end_s=0.0, backend="x", confidence=2.0
        )


def test_release_version_pattern() -> None:
    ok = ManifestRelease(
        version="0.1.0",
        created_at=datetime.now(UTC),
        n_chunks=1,
        n_files=1,
        hours_by_language={"de": 1.0},
        hours_by_tier={"A": 1.0},
        sources=["x"],
    )
    assert ok.schema_version == 1
    with pytest.raises(ValueError, match="pattern"):
        ManifestRelease(
            version="v0.1",
            created_at=datetime.now(UTC),
            n_chunks=1,
            n_files=1,
            hours_by_language={},
            hours_by_tier={},
            sources=[],
        )
