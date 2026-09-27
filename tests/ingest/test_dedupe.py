from __future__ import annotations

from wakewordworld.ingest.dedupe import mark_duplicates
from wakewordworld.ingest.records import FileRecord


def rec(file_id: str, sha: str, fp: str | None) -> FileRecord:
    return FileRecord(
        file_id=file_id,
        item_id="i",
        source_id="s",
        original_sha256="o" + file_id,
        original_ext="mp3",
        original_bytes=1,
        audio_path=f"audio/s/{file_id}.flac",
        audio_sha256=sha,
        duration_s=1.0,
        fingerprint=fp,
    )


def test_marks_by_sha_and_fingerprint() -> None:
    out = mark_duplicates(
        [rec("c", "s1", "fp1"), rec("a", "s1", "fpA"), rec("b", "s2", "fpA"), rec("d", "s3", None)]
    )
    by_id = {r.file_id: r.duplicate_of for r in out}
    assert by_id == {"a": None, "b": "a", "c": "a", "d": None}


def test_existing_duplicate_kept() -> None:
    r = rec("b", "x", None).model_copy(update={"duplicate_of": "zzz"})
    out = mark_duplicates([r])
    assert out[0].duplicate_of == "zzz"
